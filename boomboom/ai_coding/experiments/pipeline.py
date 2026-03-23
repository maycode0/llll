from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ai_coding.attack.base import ReplacementGenerator
from ai_coding.attack.mock_replacer import MockReplacementGenerator
from ai_coding.attack.search import run_local_group_attack
from ai_coding.calibration.evaluator import evaluate_probe_sets
from ai_coding.calibration.probes import select_probe_sets
from ai_coding.calibration.state import initialize_calibration_state
from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import CalibrationState, TextSample, TokenInfo
from ai_coding.models.base import SurrogateModel, VictimModel
from ai_coding.monte_carlo.estimator import estimate_group_score
from ai_coding.ranking.sorter import rank_group_scores
from ai_coding.seed.aggregator import aggregate_word_level_shap
from ai_coding.seed.selector import select_seed_indices
from ai_coding.groups.builder import build_local_groups


def run_single_sample_pipeline(
    sample: TextSample,
    tokens: list[TokenInfo],
    surrogate: SurrogateModel,
    victim: VictimModel,
    replacer: ReplacementGenerator,
    config: ExperimentConfig,
    previous_state: CalibrationState | None = None,
) -> tuple[dict[str, Any], CalibrationState]:
    effective_replacer = replacer
    if sample.replacement_candidates and isinstance(replacer, MockReplacementGenerator):
        merged = {key: list(value) for key, value in replacer.candidates.items()}
        for key, value in sample.replacement_candidates.items():
            merged[key] = list(value)
        effective_replacer = MockReplacementGenerator(candidates=merged)

    word_infos = aggregate_word_level_shap(tokens, sample.words)
    seed_indices = select_seed_indices(word_infos, config.seed)
    groups = build_local_groups(sample.words, seed_indices, config.groups)
    group_scores = [estimate_group_score(sample, word_infos, group, surrogate, config) for group in groups]
    ranked = rank_group_scores(group_scores, config.ranking)

    state_before = initialize_calibration_state(config, previous_state)
    joint_probes, syn_probes = select_probe_sets(ranked, config.calibration)
    calibration_result = evaluate_probe_sets(sample, joint_probes, syn_probes, victim, config, state_before)

    rescored = [
        estimate_group_score(
            sample,
            word_infos,
            item.group,
            surrogate,
            config,
            alpha=calibration_result.state_after.alpha,
            lambda_value=calibration_result.state_after.lambda_value,
        )
        for item in ranked
    ]
    reranked = rank_group_scores(rescored, config.ranking)
    attack_result, trace_steps = run_local_group_attack(
        sample,
        reranked,
        surrogate,
        victim,
        effective_replacer,
        candidate_rerank=config.attack.candidate_rerank,
        candidate_eval_limit=config.attack.candidate_eval_limit,
        cascade_step2_candidate_eval_limit=config.attack.cascade_step2_candidate_eval_limit,
        enable_joint_replacement=config.attack.enable_joint_replacement,
        joint_candidate_limit_per_position=config.attack.joint_candidate_limit_per_position,
        joint_eval_limit=config.attack.joint_eval_limit,
        cascade_step2_joint_candidate_limit_per_position=config.attack.cascade_step2_joint_candidate_limit_per_position,
        cascade_step2_joint_eval_limit=config.attack.cascade_step2_joint_eval_limit,
        enable_cascade_replacement=config.attack.enable_cascade_replacement,
        cascade_group_limit=config.attack.cascade_group_limit,
        cascade_beam_width=config.attack.cascade_beam_width,
        cascade_min_word_count=config.attack.cascade_min_word_count,
    )

    payload: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "seed_indices": seed_indices,
        "raw_text": sample.raw_text,
        "metadata": sample.metadata,
        "calibration": {
            "alpha_before": calibration_result.state_before.alpha,
            "lambda_before": calibration_result.state_before.lambda_value,
            "alpha_after": calibration_result.state_after.alpha,
            "lambda_after": calibration_result.state_after.lambda_value,
            "r_joint": calibration_result.r_joint,
            "r_syn": calibration_result.r_syn,
            "update_reason": calibration_result.update_reason,
            "joint_probe_pairs": [list(item.group.member_indices) for item in calibration_result.probe_joint],
            "syn_probe_pairs": [list(item.group.member_indices) for item in calibration_result.probe_syn],
        },
        "groups_before_attack": [
            {
                "member_indices": list(item.group.member_indices),
                "joint_mc": item.joint_mc,
                "variance": item.variance,
                "synergy": item.synergy,
                "score": item.score,
            }
            for item in reranked
        ],
        "attack": {
            "status": attack_result.status.value,
            "total_queries": attack_result.total_queries,
            "final_words": attack_result.final_words,
            "successful_group": list(attack_result.successful_group.member_indices) if attack_result.successful_group else None,
            "successful_replacement": list(attack_result.successful_replacement) if attack_result.successful_replacement else None,
            "trace": [
                {
                    "query_index": step.query_index,
                    "text": step.text_snapshot,
                    "predicted_label": step.predicted_label,
                    "notes": step.notes,
                }
                for step in trace_steps
            ],
        },
    }
    return payload, calibration_result.state_after


def summarize_payloads(mode: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(payloads)
    success_count = sum(1 for item in payloads if item["attack"]["status"] == "success")
    avg_queries = sum(item["attack"]["total_queries"] for item in payloads) / total if total else 0.0
    avg_alpha = sum(item["calibration"]["alpha_after"] for item in payloads) / total if total else 0.0
    avg_lambda = sum(item["calibration"]["lambda_after"] for item in payloads) / total if total else 0.0
    return {
        "mode": mode,
        "sample_count": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "avg_queries": avg_queries,
        "avg_alpha_after": avg_alpha,
        "avg_lambda_after": avg_lambda,
    }


def payload_to_table_row(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "sample_id": payload["sample_id"],
        "status": payload["attack"]["status"],
        "total_queries": payload["attack"]["total_queries"],
        "alpha_before": payload["calibration"]["alpha_before"],
        "alpha_after": payload["calibration"]["alpha_after"],
        "lambda_before": payload["calibration"]["lambda_before"],
        "lambda_after": payload["calibration"]["lambda_after"],
        "r_joint": payload["calibration"]["r_joint"],
        "r_syn": payload["calibration"]["r_syn"],
        "update_reason": payload["calibration"]["update_reason"],
        "successful_group": payload["attack"]["successful_group"],
        "successful_replacement": payload["attack"]["successful_replacement"],
    }


def write_markdown_summary(path: str | Path, title: str, summaries: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Summary", ""]
    for item in summaries:
        lines.append(
            f"- `{item['mode']}`: success_rate={item['success_rate']:.2f}, avg_queries={item['avg_queries']:.2f}, "
            f"avg_alpha_after={item['avg_alpha_after']:.2f}, avg_lambda_after={item['avg_lambda_after']:.2f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

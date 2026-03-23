from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_coding.attack.mlm_replacer import RobertaMlmReplacementGenerator
from ai_coding.core.config import ExperimentConfig
from ai_coding.data.io import build_replacement_generator, load_text_samples
from ai_coding.models.factory import build_surrogate, build_victim


def build_demo_components(
    *,
    samples_path: str | Path = "ai_inputs/demo_samples.jsonl",
    replacements_path: str | Path = "ai_inputs/demo_replacements.json",
    surrogate_path: str | Path = "ai_inputs/demo_surrogate.json",
    victim_path: str | Path = "ai_inputs/demo_victim.json",
    surrogate_kind: str = "mock",
    victim_kind: str = "mock",
    device: str = "cuda",
    surrogate_max_length: int = 128,
    victim_max_length: int = 128,
    seed_ratio: float = 0.3,
    min_seed_count: int = 1,
    seed_max_ratio: float | None = None,
    window_radius: int = 2,
    keep_function_words: bool = True,
    monte_carlo_sample_count: int = 20,
    monte_carlo_keep_probability: float = 0.5,
    alpha_init: float = 1.0,
    lambda_init: float = 1.0,
    ranking_beta: float = 0.1,
    top_k_groups: int | None = None,
    calibration_probe_count: int = 2,
    calibration_local_query_budget: int = 2,
    calibration_tau: float = 0.05,
    calibration_eta: float = 0.2,
    replacer_kind: str = "static",
    mlm_model_path: str | Path = r"E:\modelHub\roberta-base",
    mlm_top_k: int = 10,
    mlm_min_score: float = 0.01,
    mlm_relative_min_score: float = 0.2,
    mlm_filter_stopwords: bool = True,
    candidate_rerank: str = "none",
    candidate_eval_limit: int | None = None,
    cascade_step2_candidate_eval_limit: int | None = None,
    enable_joint_replacement: bool = False,
    joint_candidate_limit_per_position: int = 2,
    joint_eval_limit: int | None = 4,
    cascade_step2_joint_candidate_limit_per_position: int | None = None,
    cascade_step2_joint_eval_limit: int | None = None,
    enable_cascade_replacement: bool = False,
    cascade_group_limit: int = 2,
    cascade_beam_width: int = 1,
    cascade_min_word_count: int = 50,
    max_samples: int | None = None,
) -> tuple[ExperimentConfig, list[Any], dict[str, Any], Any, Any, Any]:
    config = ExperimentConfig()
    config.seed.seed_ratio = seed_ratio
    config.seed.min_seed_count = min_seed_count
    config.seed.max_seed_ratio = seed_max_ratio
    config.groups.window_radius = window_radius
    config.groups.keep_function_words = keep_function_words
    config.monte_carlo.sample_count = monte_carlo_sample_count
    config.monte_carlo.keep_probability = monte_carlo_keep_probability
    config.ranking.alpha_init = alpha_init
    config.ranking.lambda_init = lambda_init
    config.ranking.beta = ranking_beta
    config.ranking.top_k_groups = top_k_groups
    config.calibration.probe_count = calibration_probe_count
    config.calibration.local_query_budget = calibration_local_query_budget
    config.calibration.tau = calibration_tau
    config.calibration.eta = calibration_eta
    config.attack.candidate_rerank = candidate_rerank
    config.attack.candidate_eval_limit = candidate_eval_limit
    config.attack.cascade_step2_candidate_eval_limit = cascade_step2_candidate_eval_limit
    config.attack.enable_joint_replacement = enable_joint_replacement
    config.attack.joint_candidate_limit_per_position = joint_candidate_limit_per_position
    config.attack.joint_eval_limit = joint_eval_limit
    config.attack.cascade_step2_joint_candidate_limit_per_position = cascade_step2_joint_candidate_limit_per_position
    config.attack.cascade_step2_joint_eval_limit = cascade_step2_joint_eval_limit
    config.attack.enable_cascade_replacement = enable_cascade_replacement
    config.attack.cascade_group_limit = cascade_group_limit
    config.attack.cascade_beam_width = cascade_beam_width
    config.attack.cascade_min_word_count = cascade_min_word_count
    samples, token_map = load_text_samples(samples_path, max_samples=max_samples)
    surrogate = build_surrogate(
        surrogate_kind,
        surrogate_path,
        config=config,
        device=device,
        max_length=surrogate_max_length,
    )
    victim = build_victim(
        victim_kind,
        victim_path,
        config=config,
        device=device,
        max_length=victim_max_length,
    )
    if replacer_kind == "mlm":
        replacer = RobertaMlmReplacementGenerator(
            model_path=str(mlm_model_path),
            device_name=device,
            top_k=mlm_top_k,
            max_length=max(surrogate_max_length, victim_max_length),
            min_score=mlm_min_score,
            relative_min_score=mlm_relative_min_score,
            filter_stopwords=mlm_filter_stopwords,
        )
    elif replacer_kind == "static":
        replacer = build_replacement_generator(replacements_path, samples[0] if samples else None)
    else:
        raise ValueError(f"Unsupported replacer kind '{replacer_kind}'")
    return config, samples, token_map, surrogate, victim, replacer

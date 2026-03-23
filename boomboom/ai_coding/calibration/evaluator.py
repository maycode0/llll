from __future__ import annotations

from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import CalibrationResult, CalibrationState, GroupScore, ProbeEvaluation, TextSample
from ai_coding.models.base import VictimModel
from ai_coding.models.mock_victim import MockVictimModel


def _probe_success(sample: TextSample, group_score: GroupScore, victim: VictimModel, config: ExperimentConfig, strategy: str) -> ProbeEvaluation:
    attempts = 0
    success = False
    words = list(sample.words)
    target = sample.original_label
    i, j = group_score.group.member_indices
    for _ in range(config.calibration.local_query_budget):
        attempts += 1
        candidate = list(words)
        candidate[i] = config.mask_token
        if strategy == "joint":
            candidate[j] = config.mask_token
        else:
            candidate[j] = "bad"
        if isinstance(victim, MockVictimModel):
            predicted = victim.probe_predict_label(sample.sample_id, group_score.group.member_indices, strategy, candidate)
        else:
            predicted = victim.predict_label(candidate)
        if predicted != target:
            success = True
            break
    return ProbeEvaluation(group=group_score.group, success=success, attempts=attempts, strategy=strategy)


def evaluate_probe_sets(
    sample: TextSample,
    joint_probes: list[GroupScore],
    syn_probes: list[GroupScore],
    victim: VictimModel,
    config: ExperimentConfig,
    state_before: CalibrationState,
) -> CalibrationResult:
    joint_results = [_probe_success(sample, item, victim, config, "joint") for item in joint_probes]
    syn_results = [_probe_success(sample, item, victim, config, "syn") for item in syn_probes]

    r_joint = sum(item.success for item in joint_results) / len(joint_results) if joint_results else 0.0
    r_syn = sum(item.success for item in syn_results) / len(syn_results) if syn_results else 0.0

    state_after = CalibrationState(alpha=state_before.alpha, lambda_value=state_before.lambda_value)
    update_reason = "no_update"
    if r_syn - r_joint > config.calibration.tau:
        state_after.lambda_value += config.calibration.eta
        update_reason = "increase_lambda"
    elif r_joint - r_syn > config.calibration.tau:
        state_after.alpha += config.calibration.eta
        update_reason = "increase_alpha"

    state_after.alpha = max(0.0, state_after.alpha)
    state_after.lambda_value = max(0.0, state_after.lambda_value)
    return CalibrationResult(
        state_before=state_before,
        state_after=state_after,
        r_joint=r_joint,
        r_syn=r_syn,
        probe_joint=joint_results,
        probe_syn=syn_results,
        update_reason=update_reason,
    )

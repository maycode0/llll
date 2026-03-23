from __future__ import annotations

from ai_coding.calibration.evaluator import evaluate_probe_sets
from ai_coding.calibration.probes import select_probe_sets
from ai_coding.calibration.state import initialize_calibration_state
from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import CalibrationState
from ai_coding.core.enums import ResetMode
from ai_coding.groups.builder import build_local_groups
from ai_coding.monte_carlo.estimator import estimate_group_score
from ai_coding.ranking.sorter import rank_group_scores


def _build_ranked_groups(sample_text_sample, sample_word_infos, sample_surrogate, experiment_config: ExperimentConfig):
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    scored = [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups]
    return rank_group_scores(scored, experiment_config.ranking)


def test_initialize_calibration_state_respects_sample_reset(experiment_config: ExperimentConfig) -> None:
    previous = CalibrationState(alpha=1.5, lambda_value=1.2)
    state = initialize_calibration_state(experiment_config, previous)
    assert state.alpha == experiment_config.ranking.alpha_init
    assert state.lambda_value == experiment_config.ranking.lambda_init


def test_initialize_calibration_state_respects_global_carry(experiment_config: ExperimentConfig) -> None:
    experiment_config.reset_mode = ResetMode.GLOBAL_CARRY
    previous = CalibrationState(alpha=1.5, lambda_value=1.2)
    state = initialize_calibration_state(experiment_config, previous)
    assert state.alpha == 1.5
    assert state.lambda_value == 1.2


def test_select_probe_sets_prefers_distinct_groups(sample_text_sample, sample_word_infos, sample_surrogate, experiment_config: ExperimentConfig) -> None:
    ranked = _build_ranked_groups(sample_text_sample, sample_word_infos, sample_surrogate, experiment_config)
    joint_probes, syn_probes = select_probe_sets(ranked, experiment_config.calibration)
    assert joint_probes
    assert syn_probes
    assert not ({item.group.member_indices for item in joint_probes} & {item.group.member_indices for item in syn_probes})


def test_evaluate_probe_sets_updates_lambda_when_syn_probes_win(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    sample_victim,
    experiment_config: ExperimentConfig,
    default_calibration_state,
) -> None:
    ranked = _build_ranked_groups(sample_text_sample, sample_word_infos, sample_surrogate, experiment_config)
    joint_probes, syn_probes = select_probe_sets(ranked, experiment_config.calibration)
    result = evaluate_probe_sets(sample_text_sample, joint_probes, syn_probes, sample_victim, experiment_config, default_calibration_state)

    assert result.r_syn >= result.r_joint
    assert result.r_syn - result.r_joint > experiment_config.calibration.tau
    assert result.state_after.lambda_value > result.state_before.lambda_value
    assert result.update_reason == "increase_lambda"

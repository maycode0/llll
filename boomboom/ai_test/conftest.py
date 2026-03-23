from __future__ import annotations

import pytest

from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import CalibrationState, GroupScore, LocalGroup, TextSample, TokenInfo, WordInfo
from ai_coding.models.mock_surrogate import MockSurrogateModel
from ai_coding.models.mock_victim import MockVictimModel
from ai_coding.ranking.score import compute_group_score
from ai_coding.ranking.synergy import compute_synergy


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.fixture
def sample_words() -> list[str]:
    return ["the", "movie", "was", "not", "very", "good", "."]


@pytest.fixture
def sample_tokens() -> list[TokenInfo]:
    return [
        TokenInfo(token="the", word_index=0, shap_value=0.0),
        TokenInfo(token="movie", word_index=1, shap_value=0.4),
        TokenInfo(token="was", word_index=2, shap_value=0.1),
        TokenInfo(token="not", word_index=3, shap_value=0.6),
        TokenInfo(token="ver", word_index=4, shap_value=0.2),
        TokenInfo(token="##y", word_index=4, shap_value=0.3),
        TokenInfo(token="good", word_index=5, shap_value=0.8),
        TokenInfo(token=".", word_index=6, shap_value=0.0),
    ]


@pytest.fixture
def sample_word_infos(sample_words: list[str]) -> list[WordInfo]:
    return [
        WordInfo(index=0, text="the", phi=0.0),
        WordInfo(index=1, text="movie", phi=0.4),
        WordInfo(index=2, text="was", phi=0.1),
        WordInfo(index=3, text="not", phi=0.6),
        WordInfo(index=4, text="very", phi=0.5),
        WordInfo(index=5, text="good", phi=0.8),
        WordInfo(index=6, text=".", phi=0.0),
    ]


@pytest.fixture
def sample_group_scores() -> list[GroupScore]:
    groups = [
        LocalGroup(anchor_index=3, member_indices=(3, 5)),
        LocalGroup(anchor_index=4, member_indices=(4, 5)),
        LocalGroup(anchor_index=1, member_indices=(1, 3)),
    ]
    specs = [
        (groups[0], 1.3, 0.1, 0.6, 1.78),
        (groups[1], 1.1, 0.0, 0.2, 1.28),
        (groups[2], 0.7, 0.1, 0.2, 0.82),
    ]
    result: list[GroupScore] = []
    for group, joint_mc, variance, synergy, expected in specs:
        score = compute_group_score(joint_mc, synergy, variance, alpha=1.0, lambda_value=0.9, beta=0.6)
        assert round(score, 2) == round(expected, 2)
        result.append(GroupScore(group=group, joint_mc=joint_mc, variance=variance, synergy=synergy, score=score))
    return result


@pytest.fixture
def sample_text_sample(sample_words: list[str]) -> TextSample:
    return TextSample(sample_id="demo-1", words=sample_words, original_label=1)


@pytest.fixture
def sample_surrogate(experiment_config: ExperimentConfig) -> MockSurrogateModel:
    return MockSurrogateModel(
        word_weights={"movie": 0.4, "was": 0.1, "not": 0.6, "very": 0.5, "good": 0.8, "bad": -0.7},
        pair_bonus={("good", "not"): 0.4, ("good", "very"): 0.2, ("bad", "not"): -0.2},
        mask_token=experiment_config.mask_token,
    )


@pytest.fixture
def sample_victim(experiment_config: ExperimentConfig) -> MockVictimModel:
    return MockVictimModel(
        target_label=1,
        word_weights={"movie": 0.4, "was": 0.1, "not": 0.6, "very": 0.5, "good": 0.8, "bad": -0.7},
        pair_bonus={("good", "not"): 0.4, ("good", "very"): 0.2, ("bad", "not"): -0.2},
        probe_forced_labels={
            "demo-1": {
                "joint": {(3, 5): 1, (4, 5): 1},
                "syn": {(1, 3): 0, (2, 3): 0},
            }
        },
        mask_token=experiment_config.mask_token,
        threshold=1.7,
    )


@pytest.fixture
def default_calibration_state() -> CalibrationState:
    return CalibrationState(alpha=1.0, lambda_value=1.0)

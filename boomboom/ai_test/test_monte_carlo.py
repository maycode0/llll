from __future__ import annotations

from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import LocalGroup, TextSample
from ai_coding.models.mock_surrogate import MockSurrogateModel
from ai_coding.monte_carlo.estimator import estimate_group_score
from ai_coding.monte_carlo.inputs import build_masked_pair
from ai_coding.monte_carlo.sampler import sample_background_indices


def test_build_masked_pair_matches_design_inputs(sample_words) -> None:
    group = LocalGroup(anchor_index=3, member_indices=(3, 5))
    x_s, x_s_union_g = build_masked_pair(sample_words, background_indices={1, 4}, group=group, mask_token="[MASK]")
    assert x_s == ["[MASK]", "movie", "[MASK]", "[MASK]", "very", "[MASK]", "[MASK]"]
    assert x_s_union_g == ["[MASK]", "movie", "[MASK]", "not", "very", "good", "[MASK]"]


def test_sample_background_indices_is_seeded() -> None:
    config = ExperimentConfig()
    rng_a = __import__("random").Random(13)
    rng_b = __import__("random").Random(13)
    left = sample_background_indices([0, 1, 2, 3], config.monte_carlo, rng_a)
    right = sample_background_indices([0, 1, 2, 3], config.monte_carlo, rng_b)
    assert left == right


def test_estimate_group_score_returns_consistent_joint_mc(sample_word_infos, sample_words) -> None:
    config = ExperimentConfig()
    sample = TextSample(sample_id="demo", words=sample_words, original_label=1)
    group = LocalGroup(anchor_index=3, member_indices=(3, 5))
    surrogate = MockSurrogateModel(
        word_weights={"movie": 0.4, "was": 0.1, "not": 0.6, "very": 0.5, "good": 0.8},
        pair_bonus={("good", "not"): 0.4, ("good", "very"): 0.2},
        mask_token=config.mask_token,
    )

    result = estimate_group_score(sample, sample_word_infos, group, surrogate, config)

    assert len(result.mc_samples) == config.monte_carlo.sample_count
    assert result.joint_mc > 1.5
    assert result.variance >= 0.0
    assert result.synergy > 0.0
    assert result.score > 0.0

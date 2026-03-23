from __future__ import annotations

import statistics

from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import GroupScore, LocalGroup, TextSample, WordInfo
from ai_coding.core.random_state import build_random
from ai_coding.models.base import SurrogateModel
from ai_coding.monte_carlo.inputs import build_masked_pair
from ai_coding.monte_carlo.sampler import sample_background_indices
from ai_coding.ranking.score import compute_group_score
from ai_coding.ranking.synergy import compute_synergy


def estimate_group_score(
    sample: TextSample,
    words: list[WordInfo],
    group: LocalGroup,
    surrogate: SurrogateModel,
    config: ExperimentConfig,
    *,
    alpha: float | None = None,
    lambda_value: float | None = None,
) -> GroupScore:
    rng = build_random(config.random_seed + group.member_indices[0] * 1000 + group.member_indices[1])
    universe_indices = [index for index in range(len(sample.words)) if index not in group.member_indices]

    mc_samples: list[float] = []
    for _ in range(config.monte_carlo.sample_count):
        background = sample_background_indices(universe_indices, config.monte_carlo, rng)
        x_s, x_s_union_g = build_masked_pair(sample.words, background, group, config.mask_token)
        with_group = surrogate.score_label_support(x_s_union_g, sample.original_label)
        without_group = surrogate.score_label_support(x_s, sample.original_label)
        mc_samples.append(with_group - without_group)

    joint_mc = statistics.fmean(mc_samples) if mc_samples else 0.0
    variance = statistics.pvariance(mc_samples) if len(mc_samples) > 1 else 0.0
    phi_i = words[group.member_indices[0]].phi
    phi_j = words[group.member_indices[1]].phi
    synergy = compute_synergy(joint_mc, phi_i, phi_j)
    alpha_value = config.ranking.alpha_init if alpha is None else alpha
    lambda_weight = config.ranking.lambda_init if lambda_value is None else lambda_value
    score = compute_group_score(joint_mc, synergy, variance, alpha_value, lambda_weight, config.ranking.beta)
    return GroupScore(
        group=group,
        joint_mc=joint_mc,
        variance=variance,
        synergy=synergy,
        score=score,
        mc_samples=mc_samples,
    )

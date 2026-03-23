from __future__ import annotations

from ai_coding.core.config import RankingConfig
from ai_coding.ranking.score import compute_group_score
from ai_coding.ranking.sorter import rank_group_scores
from ai_coding.ranking.synergy import compute_synergy


def test_compute_synergy_matches_design_formula() -> None:
    assert compute_synergy(joint_mc=1.2, phi_i=0.5, phi_j=0.4) == 0.29999999999999993


def test_compute_group_score_matches_design_formula() -> None:
    score = compute_group_score(joint_mc=1.2, synergy=0.3, variance=0.2, alpha=1.0, lambda_value=1.0, beta=0.5)
    assert round(score, 4) == 1.4


def test_rank_group_scores_orders_by_score_then_joint_mc(sample_group_scores) -> None:
    ranked = rank_group_scores(sample_group_scores, RankingConfig())
    assert [item.group.member_indices for item in ranked] == [(3, 5), (4, 5), (1, 3)]

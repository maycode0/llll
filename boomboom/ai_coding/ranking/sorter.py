from __future__ import annotations

from ai_coding.core.config import RankingConfig
from ai_coding.core.data_models import GroupScore


def rank_group_scores(group_scores: list[GroupScore], config: RankingConfig) -> list[GroupScore]:
    ranked = sorted(group_scores, key=lambda item: (-item.score, -item.joint_mc, item.group.member_indices))
    if config.top_k_groups is None:
        return ranked
    return ranked[: config.top_k_groups]

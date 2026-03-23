from __future__ import annotations

from ai_coding.core.config import GroupConfig
from ai_coding.groups.builder import build_local_groups
from ai_coding.groups.neighborhood import build_position_neighborhood


def test_build_position_neighborhood_respects_radius() -> None:
    assert build_position_neighborhood(anchor_index=3, total_words=7, radius=2) == [1, 2, 4, 5]


def test_build_local_groups_deduplicates_and_filters(sample_words) -> None:
    config = GroupConfig(window_radius=2)
    groups = build_local_groups(sample_words, seed_indices=[3, 5, 6], config=config)
    pairs = [item.member_indices for item in groups]
    assert pairs == [(1, 3), (2, 3), (3, 4), (3, 5), (4, 5)]

from __future__ import annotations

from ai_coding.core.config import GroupConfig
from ai_coding.core.data_models import LocalGroup
from ai_coding.groups.filters import should_keep_word
from ai_coding.groups.neighborhood import build_position_neighborhood


def build_local_groups(words: list[str], seed_indices: list[int], config: GroupConfig) -> list[LocalGroup]:
    groups: dict[tuple[int, int], LocalGroup] = {}
    total_words = len(words)
    for anchor_index in seed_indices:
        if anchor_index < 0 or anchor_index >= total_words:
            continue
        if not should_keep_word(words[anchor_index], config):
            continue
        for neighbor_index in build_position_neighborhood(anchor_index, total_words, config.window_radius):
            if not should_keep_word(words[neighbor_index], config):
                continue
            pair = tuple(sorted((anchor_index, neighbor_index)))
            groups.setdefault(pair, LocalGroup(anchor_index=anchor_index, member_indices=pair))
    return sorted(groups.values(), key=lambda item: (item.member_indices[0], item.member_indices[1], item.anchor_index))

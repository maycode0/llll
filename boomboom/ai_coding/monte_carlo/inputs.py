from __future__ import annotations

from ai_coding.core.data_models import LocalGroup


def build_masked_pair(words: list[str], background_indices: set[int], group: LocalGroup, mask_token: str) -> tuple[list[str], list[str]]:
    group_indices = set(group.member_indices)
    x_s: list[str] = []
    x_s_union_g: list[str] = []
    for index, word in enumerate(words):
        keep_background = index in background_indices
        keep_group = index in group_indices
        x_s.append(word if keep_background and not keep_group else mask_token)
        x_s_union_g.append(word if keep_background or keep_group else mask_token)
    return x_s, x_s_union_g

from __future__ import annotations

from ai_coding.core.config import CalibrationConfig
from ai_coding.core.data_models import GroupScore


def select_probe_sets(group_scores: list[GroupScore], config: CalibrationConfig) -> tuple[list[GroupScore], list[GroupScore]]:
    joint_ranked = sorted(group_scores, key=lambda item: (-item.joint_mc, -item.score, item.group.member_indices))
    joint_selected = joint_ranked[: config.probe_count]
    used_pairs = {item.group.member_indices for item in joint_selected}

    syn_ranked = sorted(group_scores, key=lambda item: (-item.synergy, -item.joint_mc, item.group.member_indices))
    syn_selected: list[GroupScore] = []
    for item in syn_ranked:
        if item.group.member_indices in used_pairs:
            continue
        syn_selected.append(item)
        if len(syn_selected) >= config.probe_count:
            break
    return joint_selected, syn_selected

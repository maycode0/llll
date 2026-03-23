from __future__ import annotations


def compute_synergy(joint_mc: float, phi_i: float, phi_j: float) -> float:
    return joint_mc - phi_i - phi_j

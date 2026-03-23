from __future__ import annotations


def compute_group_score(joint_mc: float, synergy: float, variance: float, alpha: float, lambda_value: float, beta: float) -> float:
    return alpha * joint_mc + lambda_value * synergy - beta * variance

from __future__ import annotations

import random

from ai_coding.core.config import MonteCarloConfig


def sample_background_indices(universe_indices: list[int], config: MonteCarloConfig, rng: random.Random) -> set[int]:
    selected: set[int] = set()
    for index in universe_indices:
        if rng.random() < config.keep_probability:
            selected.add(index)
    return selected

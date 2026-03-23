from __future__ import annotations

import math

from ai_coding.core.config import SeedConfig
from ai_coding.core.data_models import WordInfo


def select_seed_indices(words: list[WordInfo], config: SeedConfig) -> list[int]:
    if not words:
        return []
    candidate_count = max(config.min_seed_count, math.ceil(config.seed_ratio * len(words)))
    if config.max_seed_ratio is not None:
        max_allowed = max(config.min_seed_count, math.ceil(config.max_seed_ratio * len(words)))
        candidate_count = min(candidate_count, max_allowed)
    ranked = sorted(words, key=lambda item: (-item.phi, item.index))
    return sorted(word.index for word in ranked[:candidate_count])

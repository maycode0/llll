from __future__ import annotations

from ai_coding.core.config import SeedConfig
from ai_coding.seed.aggregator import aggregate_word_level_shap
from ai_coding.seed.selector import select_seed_indices


def test_aggregate_word_level_shap(sample_tokens, sample_words) -> None:
    word_infos = aggregate_word_level_shap(sample_tokens, sample_words)
    assert [round(item.phi, 2) for item in word_infos] == [0.0, 0.4, 0.1, 0.6, 0.5, 0.8, 0.0]


def test_select_seed_indices_uses_ratio_and_descending_phi(sample_word_infos) -> None:
    config = SeedConfig(seed_ratio=0.3, min_seed_count=1)
    assert select_seed_indices(sample_word_infos, config) == [3, 4, 5]


def test_select_seed_indices_honors_minimum_count(sample_word_infos) -> None:
    config = SeedConfig(seed_ratio=0.0, min_seed_count=2)
    assert select_seed_indices(sample_word_infos, config) == [3, 5]

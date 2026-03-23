from __future__ import annotations

from pathlib import Path

import pytest

from ai_coding.attack.mlm_replacer import RobertaMlmReplacementGenerator


def test_roberta_mlm_replacer_score_filter_rejects_low_probability_tail() -> None:
    replacer = RobertaMlmReplacementGenerator(
        model_path=r"E:\modelHub\roberta-base",
        device_name="cpu",
        top_k=5,
        min_score=0.05,
        relative_min_score=0.5,
    )
    assert replacer._passes_score_filter(0.04, 0.9) is False
    assert replacer._passes_score_filter(0.3, 0.9) is False
    assert replacer._passes_score_filter(0.5, 0.9) is True


def test_roberta_mlm_replacer_filters_stopword_like_candidates() -> None:
    replacer = RobertaMlmReplacementGenerator(
        model_path=r"E:\modelHub\roberta-base",
        device_name="cpu",
        top_k=5,
    )
    assert replacer._is_valid_candidate("them", "worst") is False
    assert replacer._is_valid_candidate("better", "worst") is True


@pytest.mark.skipif(not Path(r"E:\modelHub\roberta-base").exists(), reason="Local RoBERTa MLM is unavailable")
def test_roberta_mlm_replacer_generates_candidates_on_cpu() -> None:
    replacer = RobertaMlmReplacementGenerator(
        model_path=r"E:\modelHub\roberta-base",
        device_name="cpu",
        top_k=5,
    )
    candidates = replacer.get_candidates(["this", "movie", "was", "great"], 3)
    assert isinstance(candidates, list)
    assert len(candidates) <= 5
    assert all(candidate.strip() for candidate in candidates)
    assert all(candidate.lower() != "great" for candidate in candidates)


@pytest.mark.skipif(not Path(r"E:\modelHub\roberta-base").exists(), reason="Local RoBERTa MLM is unavailable")
def test_roberta_mlm_replacer_masks_selected_position() -> None:
    replacer = RobertaMlmReplacementGenerator(
        model_path=r"E:\modelHub\roberta-base",
        device_name="cpu",
        top_k=5,
    )
    masked = replacer.build_masked_text(["a", "beautiful", "film"], 1)
    assert masked == f"a {replacer.tokenizer.mask_token} film"


@pytest.mark.skipif(not Path(r"E:\modelHub\roberta-base").exists(), reason="Local RoBERTa MLM is unavailable")
def test_roberta_mlm_replacer_truncated_text_keeps_mask() -> None:
    replacer = RobertaMlmReplacementGenerator(
        model_path=r"E:\modelHub\roberta-base",
        device_name="cpu",
        top_k=5,
        max_length=8,
    )
    masked = replacer.build_truncated_masked_text(["a", "beautiful", "film"], 1)
    assert replacer.tokenizer.mask_token in masked


@pytest.mark.skipif(not Path(r"E:\modelHub\roberta-base").exists(), reason="Local RoBERTa MLM is unavailable")
def test_roberta_mlm_replacer_batch_generation_matches_request_count() -> None:
    replacer = RobertaMlmReplacementGenerator(
        model_path=r"E:\modelHub\roberta-base",
        device_name="cpu",
        top_k=3,
    )
    results = replacer.get_candidates_batch(
        [
            (["this", "movie", "was", "great"], 3),
            (["a", "beautiful", "film"], 1),
        ]
    )
    assert len(results) == 2
    assert all(isinstance(item, list) for item in results)

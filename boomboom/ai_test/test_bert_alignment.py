from __future__ import annotations

from pathlib import Path

from ai_coding.prep.bert_wordpiece import align_text_with_tokenizer, load_local_tokenizer, reconstruct_words_from_wordpieces
from ai_coding.prep.sst2_bert_alignment import build_aligned_record


def test_reconstruct_words_from_wordpieces_merges_continuations() -> None:
    assert reconstruct_words_from_wordpieces(["communicate", "##s", "beautiful"]) == ["communicates", "beautiful"]


def test_reconstruct_words_from_wordpieces_keeps_punctuation_units() -> None:
    assert reconstruct_words_from_wordpieces(["great", ",", "film", "!"]) == ["great", ",", "film", "!"]


def test_align_text_with_tokenizer_uses_wordpiece_boundaries() -> None:
    tokenizer = load_local_tokenizer(r"E:\modelHub\bert-base-uncased-SST-2")
    alignment = align_text_with_tokenizer(
        "that loves its characters and communicates something rather beautiful about human nature",
        tokenizer,
    )
    assert alignment.words == [
        "that",
        "loves",
        "its",
        "characters",
        "and",
        "communicates",
        "something",
        "rather",
        "beautiful",
        "about",
        "human",
        "nature",
    ]
    assert [item.token for item in alignment.tokens] == [
        "that",
        "loves",
        "its",
        "characters",
        "and",
        "communicate",
        "##s",
        "something",
        "rather",
        "beautiful",
        "about",
        "human",
        "nature",
    ]
    assert alignment.tokens[5].word_index == alignment.tokens[6].word_index == 5


def test_build_aligned_record_replaces_words_with_tokenizer_native_words() -> None:
    tokenizer = load_local_tokenizer(r"E:\modelHub\bert-base-uncased-SST-2")
    record = {
        "sample_id": "sample-1",
        "original_label": 1,
        "raw_text": "that loves its characters and communicates something",
        "metadata": {"dataset": "sst2"},
        "words": ["legacy", "words"],
    }
    aligned = build_aligned_record(record, "bert-local", tokenizer)
    assert aligned["words"] == ["that", "loves", "its", "characters", "and", "communicates", "something"]
    assert aligned["tokens"][5]["token"] == "communicate"
    assert aligned["tokens"][6]["token"] == "##s"
    assert aligned["tokens"][5]["word_index"] == aligned["tokens"][6]["word_index"]
    assert aligned["metadata"]["word_source"] == "tokenizer_reconstructed"

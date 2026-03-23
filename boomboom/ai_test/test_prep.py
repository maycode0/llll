from __future__ import annotations

from pathlib import Path

from ai_coding.prep.json_array_to_jsonl import tokenize_english_words
from ai_coding.prep.sst2_json_to_jsonl import convert_sst2_json


def test_tokenize_english_words_splits_punctuation() -> None:
    assert tokenize_english_words("contains no wit , only labored gags") == ["contains", "no", "wit", ",", "only", "labored", "gags"]


def test_convert_sst2_json_writes_expected_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.json"
    output_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        '[{"sentence":"that loves its characters ","label":1},{"sentence":"contains no wit , only labored gags ","label":0}]',
        encoding="utf-8",
    )

    summary = convert_sst2_json(input_path, output_path, dataset="sst2", split="train")
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()

    assert summary["sample_count"] == 2
    assert '"sample_id": "sst2-train-000001"' in lines[0]
    assert '"raw_text": "that loves its characters"' in lines[0]
    assert '"words": ["contains", "no", "wit", ",", "only", "labored", "gags"]' in lines[1]

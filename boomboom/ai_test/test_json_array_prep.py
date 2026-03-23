from __future__ import annotations

from pathlib import Path

from ai_coding.prep.json_array_to_jsonl import convert_json_array_dataset, normalize_text, tokenize_english_words


def test_normalize_text_removes_html_breaks_and_extra_spaces() -> None:
    text = 'Hello<br /><br />world &quot; test '
    assert normalize_text(text) == 'Hello world " test'


def test_convert_json_array_dataset_supports_custom_text_field(tmp_path: Path) -> None:
    input_path = tmp_path / "imdb.json"
    output_path = tmp_path / "imdb.jsonl"
    input_path.write_text(
        '[{"text":"A good movie.<br /><br />Really good.","label":1}]',
        encoding="utf-8",
    )

    summary = convert_json_array_dataset(
        input_path,
        output_path,
        dataset="imdb",
        split="train",
        text_field="text",
        label_field="label",
    )

    line = output_path.read_text(encoding="utf-8").strip()
    assert summary["sample_count"] == 1
    assert '"sample_id": "imdb-train-000001"' in line
    assert '"raw_text": "A good movie. Really good."' in line
    assert '"words": ["A", "good", "movie", ".", "Really", "good", "."]' in line

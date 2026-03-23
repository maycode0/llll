from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ai_coding.utils.io_utils import write_jsonl

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HTML_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    value = html.unescape(text)
    value = HTML_BREAK_PATTERN.sub(" ", value)
    value = WHITESPACE_PATTERN.sub(" ", value)
    return value.strip()


def tokenize_english_words(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def convert_record(
    record: dict[str, Any],
    index: int,
    dataset: str,
    split: str,
    source_file: str,
    text_field: str,
    label_field: str,
) -> dict[str, Any]:
    raw_text = normalize_text(str(record[text_field]))
    return {
        "sample_id": f"{dataset}-{split}-{index:06d}",
        "original_label": int(record[label_field]),
        "raw_text": raw_text,
        "metadata": {
            "language": "en",
            "dataset": dataset,
            "split": split,
            "source_file": source_file,
        },
        "words": tokenize_english_words(raw_text),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(item["original_label"] for item in records)
    word_lengths = [len(item["words"]) for item in records]
    return {
        "sample_count": len(records),
        "label_distribution": dict(labels),
        "min_words": min(word_lengths) if word_lengths else 0,
        "max_words": max(word_lengths) if word_lengths else 0,
        "avg_words": sum(word_lengths) / len(word_lengths) if word_lengths else 0.0,
    }


def convert_json_array_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    split: str,
    text_field: str,
    label_field: str,
) -> dict[str, Any]:
    source_path = Path(input_path)
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected the input JSON file to contain a top-level array")

    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if text_field not in row or label_field not in row:
            raise ValueError(f"Record {index} is missing '{text_field}' or '{label_field}'")
        raw_text = normalize_text(str(row[text_field]))
        if not raw_text:
            continue
        label = row[label_field]
        if label not in (0, 1):
            raise ValueError(f"Record {index} has unsupported label: {label}")
        converted.append(
            convert_record(
                row,
                index,
                dataset,
                split,
                source_path.name,
                text_field,
                label_field,
            )
        )

    write_jsonl(output_path, converted)
    return build_summary(converted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a JSON array dataset into normalized JSONL samples")
    parser.add_argument("--input", required=True, help="Path to the source JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file")
    parser.add_argument("--dataset", required=True, help="Dataset name recorded in metadata")
    parser.add_argument("--split", required=True, help="Split name recorded in metadata")
    parser.add_argument("--text-field", default="sentence", help="Field name containing the raw text")
    parser.add_argument("--label-field", default="label", help="Field name containing the label")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_json_array_dataset(
        args.input,
        args.output,
        dataset=args.dataset,
        split=args.split,
        text_field=args.text_field,
        label_field=args.label_field,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved normalized samples to {args.output}")


if __name__ == "__main__":
    main()

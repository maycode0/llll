from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_coding.data.io import read_jsonl
from ai_coding.prep.bert_wordpiece import align_text_with_tokenizer, load_local_tokenizer
from ai_coding.utils.io_utils import write_jsonl


def build_aligned_record(record: dict[str, Any], tokenizer_name: str, tokenizer) -> dict[str, Any]:
    raw_text = str(record["raw_text"])
    alignment = align_text_with_tokenizer(raw_text, tokenizer)
    metadata = dict(record.get("metadata", {}))
    metadata.update(
        {
            "word_source": "tokenizer_reconstructed",
            "tokenizer_name": tokenizer_name,
            "alignment_method": "bert_wordpiece",
        }
    )
    return {
        "sample_id": record["sample_id"],
        "original_label": int(record["original_label"]),
        "raw_text": raw_text,
        "metadata": metadata,
        "words": list(alignment.words),
        "tokens": [
            {
                "token": item.token,
                "word_index": item.word_index,
                "shap_value": 0.0,
            }
            for item in alignment.tokens
        ],
    }


def align_sst2_samples(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(input_path, max_rows=max_samples)
    tokenizer = load_local_tokenizer(model_path)
    tokenizer_name = str(Path(model_path))
    aligned = [build_aligned_record(row, tokenizer_name, tokenizer) for row in rows]
    write_jsonl(output_path, aligned)
    return {
        "sample_count": len(aligned),
        "tokenizer_name": tokenizer_name,
        "max_samples": max_samples,
        "output_path": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align SST-2 samples with local BERT WordPiece tokenization")
    parser.add_argument("--input", default="ai_inputs/sst2_train_samples.jsonl", help="Path to the normalized SST-2 JSONL file")
    parser.add_argument("--output", default="ai_inputs/sst2_train_first5_bert_aligned.jsonl", help="Path to the aligned JSONL file")
    parser.add_argument("--model", default=r"E:\modelHub\bert-base-uncased-SST-2", help="Local HuggingFace model directory")
    parser.add_argument("--max-samples", type=int, default=5, help="Number of samples to align")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = align_sst2_samples(args.input, args.output, model_path=args.model, max_samples=args.max_samples)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved aligned samples to {args.output}")


if __name__ == "__main__":
    main()

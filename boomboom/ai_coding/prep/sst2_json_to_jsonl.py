from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_coding.prep.json_array_to_jsonl import convert_json_array_dataset


def convert_sst2_json(input_path: str | Path, output_path: str | Path, dataset: str = "sst2", split: str = "train") -> dict[str, Any]:
    return convert_json_array_dataset(
        input_path,
        output_path,
        dataset=dataset,
        split=split,
        text_field="sentence",
        label_field="label",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SST-2 JSON array file into normalized JSONL samples")
    parser.add_argument("--input", default="E:/workspace/datasets/sst2/sst2-train.json", help="Path to the SST-2 JSON file")
    parser.add_argument("--output", default="ai_inputs/sst2_train_samples.jsonl", help="Path to the output JSONL file")
    parser.add_argument("--dataset", default="sst2", help="Dataset name recorded in metadata")
    parser.add_argument("--split", default="train", help="Split name recorded in metadata")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_sst2_json(args.input, args.output, dataset=args.dataset, split=args.split)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved normalized samples to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path


def sample_jsonl(input_path: str | Path, output_path: str | Path, count: int) -> int:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            written += 1
            if written >= count:
                break
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a small leading subset of a JSONL file")
    parser.add_argument("--input", required=True, help="Source JSONL path")
    parser.add_argument("--output", required=True, help="Target JSONL path")
    parser.add_argument("--count", type=int, default=8, help="Number of rows to keep")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = sample_jsonl(args.input, args.output, args.count)
    print({"written": written, "output": args.output})


if __name__ == "__main__":
    main()

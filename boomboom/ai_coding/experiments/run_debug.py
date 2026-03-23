from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_coding.experiments.cli import add_common_input_arguments, load_components_from_args
from ai_coding.experiments.pipeline import run_single_sample_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-sample debug experiment")
    add_common_input_arguments(parser)
    parser.add_argument("--sample-id", default=None, help="Optional sample id to run; defaults to the first sample")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, samples, token_map, surrogate, victim, replacer = load_components_from_args(args)
    sample = next((item for item in samples if item.sample_id == args.sample_id), samples[0]) if samples else None
    if sample is None:
        raise ValueError("No samples were loaded from the provided input file")
    payload, _ = run_single_sample_pipeline(sample, token_map[sample.sample_id], surrogate, victim, replacer, config)
    output_path = Path("ai_outputs/debug_runs") / f"{sample.sample_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved debug output to {output_path}")


if __name__ == "__main__":
    main()

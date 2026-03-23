from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_coding.experiments.cli import add_common_input_arguments, load_components_from_args
from ai_coding.experiments.pipeline import payload_to_table_row, run_single_sample_pipeline, summarize_payloads
from ai_coding.utils.io_utils import write_csv, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch experiment pipeline")
    add_common_input_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, samples, token_map, surrogate, victim, replacer = load_components_from_args(args)
    payloads = []
    previous_state = None
    for sample in samples:
        payload, previous_state = run_single_sample_pipeline(
            sample,
            token_map[sample.sample_id],
            surrogate,
            victim,
            replacer,
            config,
            previous_state=None,
        )
        payloads.append(payload)

    summary = summarize_payloads(config.reset_mode.value, payloads)
    ablation_dir = Path("ai_outputs/ablation")
    table_dir = Path("ai_outputs/tables")
    write_jsonl(ablation_dir / "batch_results.jsonl", payloads)
    write_csv(table_dir / "batch_results.csv", [payload_to_table_row(config.reset_mode.value, item) for item in payloads])
    summary_path = ablation_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved batch results to {ablation_dir / 'batch_results.jsonl'}")


if __name__ == "__main__":
    main()

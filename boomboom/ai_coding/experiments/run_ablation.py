from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_coding.core.enums import ResetMode
from ai_coding.experiments.cli import add_common_input_arguments, load_components_from_args
from ai_coding.experiments.pipeline import payload_to_table_row, run_single_sample_pipeline, summarize_payloads, write_markdown_summary
from ai_coding.utils.io_utils import write_csv, write_jsonl


def run_mode(mode: ResetMode, args: argparse.Namespace | None = None) -> tuple[list[dict], dict]:
    if args is None:
        parser = build_parser()
        args = parser.parse_args([])
    config, samples, token_map, surrogate, victim, replacer = load_components_from_args(args)
    config.reset_mode = mode
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
            previous_state=previous_state,
        )
        payloads.append(payload)
    summary = summarize_payloads(mode.value, payloads)
    return payloads, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reset-mode ablation")
    add_common_input_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = []
    combined_rows = []
    ablation_dir = Path("ai_outputs/ablation")
    table_dir = Path("ai_outputs/tables")

    for mode in (ResetMode.SAMPLE_RESET, ResetMode.GLOBAL_CARRY):
        payloads, summary = run_mode(mode, args)
        summaries.append(summary)
        combined_rows.extend(payload_to_table_row(mode.value, item) for item in payloads)
        write_jsonl(ablation_dir / f"{mode.value}_results.jsonl", payloads)
        (ablation_dir / f"{mode.value}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv(table_dir / "reset_mode_ablation.csv", combined_rows)
    write_markdown_summary("ai_outputs/conclusions/reset_mode_ablation.md", "Reset Mode Ablation", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Saved ablation table to {table_dir / 'reset_mode_ablation.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_coding.attack.mock_replacer import MockReplacementGenerator
from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import TextSample, TokenInfo
from ai_coding.models.mock_surrogate import MockSurrogateModel
from ai_coding.models.mock_victim import MockVictimModel


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_text_samples(path: str | Path, max_samples: int | None = None) -> tuple[list[TextSample], dict[str, list[TokenInfo]]]:
    rows = read_jsonl(path, max_rows=max_samples)
    samples: list[TextSample] = []
    token_map: dict[str, list[TokenInfo]] = {}
    for row in rows:
        sample = TextSample(
            sample_id=row["sample_id"],
            words=list(row["words"]),
            original_label=row["original_label"],
            raw_text=row.get("raw_text"),
            metadata=dict(row.get("metadata", {})),
            replacement_candidates={key: list(value) for key, value in row.get("replacement_candidates", {}).items()} or None,
        )
        if "tokens" in row:
            tokens = [
                TokenInfo(token=item["token"], word_index=item["word_index"], shap_value=item["shap_value"])
                for item in row["tokens"]
            ]
        else:
            tokens = [
                TokenInfo(token=word, word_index=index, shap_value=0.0)
                for index, word in enumerate(sample.words)
            ]
        samples.append(sample)
        token_map[sample.sample_id] = tokens
    return samples, token_map


def load_replacement_generator(path: str | Path) -> MockReplacementGenerator:
    payload = read_json(path)
    return MockReplacementGenerator(candidates={key: list(value) for key, value in payload.items()})


def build_replacement_generator(global_path: str | Path, sample: TextSample | None = None) -> MockReplacementGenerator:
    global_replacements = read_json(global_path)
    merged = {key: list(value) for key, value in global_replacements.items()}
    if sample is not None and sample.replacement_candidates:
        for key, value in sample.replacement_candidates.items():
            merged[key] = list(value)
    return MockReplacementGenerator(candidates=merged)


def load_mock_surrogate(path: str | Path, config: ExperimentConfig) -> MockSurrogateModel:
    payload = read_json(path)
    return MockSurrogateModel(
        word_weights=dict(payload.get("word_weights", {})),
        pair_bonus={tuple(item["pair"]): item["value"] for item in payload.get("pair_bonus", [])},
        mask_token=config.mask_token,
        bias=payload.get("bias", 0.0),
    )


def load_mock_victim(path: str | Path, config: ExperimentConfig) -> MockVictimModel:
    payload = read_json(path)
    probe_forced_labels = {
        sample_id: {
            strategy: {tuple(item["group"]): item["label"] for item in entries}
            for strategy, entries in strategy_map.items()
        }
        for sample_id, strategy_map in payload.get("probe_forced_labels", {}).items()
    }
    return MockVictimModel(
        target_label=payload["target_label"],
        word_weights=dict(payload.get("word_weights", {})),
        pair_bonus={tuple(item["pair"]): item["value"] for item in payload.get("pair_bonus", [])},
        probe_forced_labels=probe_forced_labels,
        mask_token=config.mask_token,
        threshold=payload.get("threshold", 1.6),
    )

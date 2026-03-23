from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification

from ai_coding.data.io import read_jsonl
from ai_coding.prep.bert_wordpiece import load_local_tokenizer, reconstruct_words_from_wordpieces
from ai_coding.utils.io_utils import write_jsonl


class BertWordPieceShapScorer:
    def __init__(self, model_path: str | Path, *, device: str = "cuda", max_length: int = 128) -> None:
        self.model_path = str(Path(model_path))
        self.tokenizer = load_local_tokenizer(model_path)
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.mask_token = self.tokenizer.mask_token or "[MASK]"

    @staticmethod
    def _resolve_device(requested_device: str) -> torch.device:
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for SHAP scoring but is unavailable")
            return torch.device("cuda")
        return torch.device("cpu")

    def _render_masked_text(self, pieces: list[str], mask_row: np.ndarray) -> str:
        masked_pieces = [piece if keep >= 0.5 else self.mask_token for piece, keep in zip(pieces, mask_row)]
        masked_words = reconstruct_words_from_wordpieces(masked_pieces)
        return " ".join(masked_words)

    def predict_proba(self, pieces: list[str], mask_matrix: np.ndarray) -> np.ndarray:
        texts = [self._render_masked_text(pieces, row) for row in mask_matrix]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().float().cpu().numpy()

    def explain(self, pieces: list[str], target_label: int, *, nsamples: int) -> list[float]:
        feature_count = len(pieces)
        if feature_count == 0:
            return []
        baseline = np.zeros((1, feature_count), dtype=np.float32)
        full_input = np.ones((1, feature_count), dtype=np.float32)

        def predict(mask_matrix: np.ndarray) -> np.ndarray:
            return self.predict_proba(pieces, mask_matrix)

        explainer = shap.KernelExplainer(predict, baseline)
        shap_values = explainer.shap_values(full_input, nsamples=nsamples, silent=True)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[target_label])[0]
        else:
            values = np.asarray(shap_values)[0, :, target_label]
        return [float(item) for item in values.tolist()]


def build_shap_record(record: dict[str, Any], scorer: BertWordPieceShapScorer, *, nsamples: int) -> dict[str, Any]:
    pieces = [str(item["token"]) for item in record.get("tokens", [])]
    target_label = int(record["original_label"])
    shap_values = scorer.explain(pieces, target_label, nsamples=nsamples)
    metadata = dict(record.get("metadata", {}))
    metadata.update(
        {
            "surrogate_model_path": scorer.model_path,
            "attribution_method": "kernel_shap",
            "attribution_target": f"label_{target_label}",
            "shap_nsamples": nsamples,
        }
    )
    updated_tokens = []
    for item, shap_value in zip(record.get("tokens", []), shap_values):
        updated = dict(item)
        updated["shap_value"] = shap_value
        updated_tokens.append(updated)

    return {
        "sample_id": record["sample_id"],
        "original_label": target_label,
        "raw_text": record.get("raw_text"),
        "metadata": metadata,
        "words": list(record.get("words", [])),
        "tokens": updated_tokens,
    }


def annotate_sst2_with_shap(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    max_samples: int | None = None,
    device: str = "cuda",
    max_length: int = 128,
    nsamples: int = 64,
) -> dict[str, Any]:
    rows = read_jsonl(input_path, max_rows=max_samples)
    scorer = BertWordPieceShapScorer(model_path, device=device, max_length=max_length)
    annotated = [build_shap_record(row, scorer, nsamples=nsamples) for row in rows]
    write_jsonl(output_path, annotated)
    return {
        "sample_count": len(annotated),
        "model_path": str(model_path),
        "device": device,
        "max_samples": max_samples,
        "nsamples": nsamples,
        "output_path": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Kernel SHAP scores for aligned SST-2 BERT samples")
    parser.add_argument("--input", default="ai_inputs/sst2_train_first5_bert_aligned.jsonl", help="Path to the aligned JSONL file")
    parser.add_argument("--output", default="ai_inputs/sst2_train_first5_bert_shap.jsonl", help="Path to the SHAP-enriched JSONL file")
    parser.add_argument("--model", default=r"E:\modelHub\bert-base-uncased-SST-2", help="Local HuggingFace model directory")
    parser.add_argument("--max-samples", type=int, default=5, help="Number of samples to annotate")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Device for SHAP model inference")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--nsamples", type=int, default=64, help="Kernel SHAP sample budget per example")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = annotate_sst2_with_shap(
        args.input,
        args.output,
        model_path=args.model,
        max_samples=args.max_samples,
        device=args.device,
        max_length=args.max_length,
        nsamples=args.nsamples,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved SHAP-enriched samples to {args.output}")


if __name__ == "__main__":
    main()

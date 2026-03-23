from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


WORD_PATTERN = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
STOPWORD_CANDIDATES = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "de",
    "for",
    "from",
    "in",
    "is",
    "it",
    "la",
    "of",
    "on",
    "or",
    "the",
    "them",
    "to",
    "un",
    "with",
}


def _resolve_device(requested_device: str) -> torch.device:
    normalized = requested_device.lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current PyTorch environment")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{requested_device}'. Expected 'cpu' or 'cuda'.")


def _normalize_candidate(candidate: str) -> str:
    return candidate.strip()


@dataclass(slots=True)
class RobertaMlmReplacementGenerator:
    model_path: str
    device_name: str = "cuda"
    top_k: int = 10
    max_length: int = 512
    min_score: float = 0.01
    relative_min_score: float = 0.2
    filter_stopwords: bool = True
    device: torch.device = field(init=False)
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        model_source = str(Path(self.model_path))
        self.device = _resolve_device(self.device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_source, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def build_masked_text(self, words: list[str], position: int) -> str:
        masked_words = list(words)
        masked_words[position] = self.tokenizer.mask_token
        return " ".join(masked_words)

    def build_truncated_masked_text(self, words: list[str], position: int) -> str:
        text = self.build_masked_text(words, position)
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        if self.tokenizer.mask_token not in tokens:
            raise ValueError("Mask token fell outside the truncated MLM window")
        return self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)

    def _encode_masked_texts(self, masked_texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            masked_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _extract_mask_positions(self, input_ids: torch.Tensor) -> list[int | None]:
        positions: list[int | None] = []
        for row in input_ids:
            matches = (row == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
            if matches.numel() == 0:
                positions.append(None)
            else:
                positions.append(int(matches[0].item()))
        return positions

    def _predict_candidates_batch(self, masked_texts: list[str]) -> list[list[tuple[str, float]]]:
        if not masked_texts:
            return []
        encoded = self._encode_masked_texts(masked_texts)
        input_ids = encoded["input_ids"]
        mask_positions = self._extract_mask_positions(input_ids)
        with torch.no_grad():
            logits = self.model(**encoded).logits.detach().float().cpu()

        candidate_lists: list[list[tuple[str, float]]] = []
        raw_top_k = max(self.top_k * 3, self.top_k)
        vocab_limit = logits.shape[-1]
        top_k = min(raw_top_k, vocab_limit)

        for row_index, mask_position in enumerate(mask_positions):
            if mask_position is None:
                candidate_lists.append([])
                continue
            mask_logits = logits[row_index, mask_position]
            probs = torch.softmax(mask_logits, dim=-1)
            values, indices = torch.topk(probs, k=top_k)
            row_candidates: list[tuple[str, float]] = []
            for value, token_id in zip(values.tolist(), indices.tolist()):
                token = self.tokenizer.decode([int(token_id)], skip_special_tokens=True)
                row_candidates.append((token, float(value)))
            candidate_lists.append(row_candidates)
        return candidate_lists

    def _passes_score_filter(self, candidate_score: float, best_score: float) -> bool:
        if candidate_score < self.min_score:
            return False
        if best_score <= 0:
            return True
        return candidate_score >= best_score * self.relative_min_score

    def _is_valid_candidate(self, candidate: str, original_word: str) -> bool:
        normalized = _normalize_candidate(candidate)
        if not normalized:
            return False
        normalized_lower = normalized.lower()
        original_lower = original_word.lower()
        if normalized_lower == original_lower:
            return False
        if normalized in self.tokenizer.all_special_tokens:
            return False
        if not WORD_PATTERN.fullmatch(normalized):
            return False
        if self.filter_stopwords and normalized_lower in STOPWORD_CANDIDATES:
            return False
        if abs(len(normalized) - len(original_word)) > 4:
            return False
        if len(original_word) >= 4 and len(normalized) <= 2:
            return False
        return True

    def get_candidates(self, words: list[str], position: int) -> list[str]:
        return self.get_candidates_batch([(words, position)])[0]

    def get_candidates_batch(self, requests: list[tuple[list[str], int]]) -> list[list[str]]:
        prepared_texts: list[str] = []
        originals: list[str] = []
        valid_flags: list[bool] = []
        for words, position in requests:
            originals.append(words[position])
            try:
                prepared_texts.append(self.build_truncated_masked_text(words, position))
                valid_flags.append(True)
            except ValueError:
                prepared_texts.append("")
                valid_flags.append(False)

        valid_texts = [text for text, valid in zip(prepared_texts, valid_flags) if valid]
        predicted = self._predict_candidates_batch(valid_texts)
        predicted_iter = iter(predicted)
        all_results: list[list[str]] = []

        for original_word, valid in zip(originals, valid_flags):
            if not valid:
                all_results.append([])
                continue
            results = next(predicted_iter)
            best_score = max((score for _, score in results), default=0.0)
            deduped: list[str] = []
            seen: set[str] = set()
            for candidate, candidate_score in results:
                candidate = _normalize_candidate(candidate)
                lowered = candidate.lower()
                if lowered in seen:
                    continue
                if not self._passes_score_filter(candidate_score, best_score):
                    continue
                if not self._is_valid_candidate(candidate, original_word):
                    continue
                seen.add(lowered)
                deduped.append(candidate)
                if len(deduped) >= self.top_k:
                    break
            all_results.append(deduped)
        return all_results

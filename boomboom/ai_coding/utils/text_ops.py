from __future__ import annotations


def mask_words(words: list[str], indices_to_mask: set[int], mask_token: str) -> list[str]:
    return [mask_token if idx in indices_to_mask else word for idx, word in enumerate(words)]


def join_words(words: list[str]) -> str:
    return " ".join(words)

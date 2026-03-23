from __future__ import annotations

from typing import Protocol


class ReplacementGenerator(Protocol):
    def get_candidates(self, words: list[str], position: int) -> list[str]:
        """Return replacement candidates for the word at a given position."""
        ...

    def get_candidates_batch(self, requests: list[tuple[list[str], int]]) -> list[list[str]]:
        """Return replacement candidates for a batch of (words, position) requests."""
        ...

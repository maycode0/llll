from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MockReplacementGenerator:
    candidates: dict[str, list[str]] = field(default_factory=dict)

    def get_candidates(self, words: list[str], position: int) -> list[str]:
        word = words[position]
        return list(self.candidates.get(word.lower(), []))

    def get_candidates_batch(self, requests: list[tuple[list[str], int]]) -> list[list[str]]:
        return [self.get_candidates(words, position) for words, position in requests]

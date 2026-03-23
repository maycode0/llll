from __future__ import annotations


def build_position_neighborhood(anchor_index: int, total_words: int, radius: int) -> list[int]:
    start = max(0, anchor_index - radius)
    end = min(total_words - 1, anchor_index + radius)
    return [index for index in range(start, end + 1) if index != anchor_index]

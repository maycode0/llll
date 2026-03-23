from __future__ import annotations

from collections import defaultdict

from ai_coding.core.data_models import TokenInfo, WordInfo
from ai_coding.core.enums import TokenKind


def aggregate_word_level_shap(tokens: list[TokenInfo], words: list[str]) -> list[WordInfo]:
    totals: dict[int, float] = defaultdict(float)
    for token_info in tokens:
        totals[token_info.word_index] += token_info.shap_value

    result: list[WordInfo] = []
    for index, word in enumerate(words):
        result.append(WordInfo(index=index, text=word, phi=totals.get(index, 0.0), token_kind=TokenKind.CONTENT))
    return result

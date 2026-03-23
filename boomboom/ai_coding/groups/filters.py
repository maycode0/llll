from __future__ import annotations

from ai_coding.core.config import GroupConfig
from ai_coding.core.constants import DEFAULT_PUNCTUATION


def is_punctuation(word: str) -> bool:
    return word in DEFAULT_PUNCTUATION


def is_special_token(word: str, config: GroupConfig) -> bool:
    return word in config.special_tokens


def should_keep_word(word: str, config: GroupConfig) -> bool:
    lowered = word.lower()
    if is_special_token(word, config):
        return False
    if is_punctuation(word):
        return False
    if config.keep_function_words and lowered in config.function_words:
        return True
    return True

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


@dataclass(slots=True)
class WordPieceToken:
    token: str
    word_index: int
    token_index: int
    char_start: int
    char_end: int


@dataclass(slots=True)
class WordPieceAlignment:
    words: list[str]
    tokens: list[WordPieceToken]


def load_local_tokenizer(model_path: str | Path):
    return AutoTokenizer.from_pretrained(str(Path(model_path)), local_files_only=True)


def reconstruct_words_from_wordpieces(pieces: list[str]) -> list[str]:
    words: list[str] = []
    current = ""
    for piece in pieces:
        if piece.startswith("##"):
            current += piece[2:]
            continue
        if current:
            words.append(current)
        current = piece
    if current:
        words.append(current)
    return words


def align_text_with_tokenizer(raw_text: str, tokenizer: Any) -> WordPieceAlignment:
    encoded = tokenizer(raw_text, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offset_mapping = encoded["offset_mapping"]
    pieces = tokenizer.convert_ids_to_tokens(input_ids)

    words: list[str] = []
    tokens: list[WordPieceToken] = []
    current_word_index = -1

    for token_index, (piece, (char_start, char_end)) in enumerate(zip(pieces, offset_mapping)):
        if char_start == char_end:
            continue
        if piece in tokenizer.all_special_tokens:
            continue

        if not piece.startswith("##"):
            words.append(piece)
            current_word_index += 1
        else:
            if current_word_index < 0:
                raise ValueError(f"Encountered continuation token '{piece}' before starting a word")
            words[current_word_index] += piece[2:]

        tokens.append(
            WordPieceToken(
                token=piece,
                word_index=current_word_index,
                token_index=token_index,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return WordPieceAlignment(words=words, tokens=tokens)

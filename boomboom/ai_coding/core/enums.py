from __future__ import annotations

from enum import Enum


class ResetMode(str, Enum):
    SAMPLE_RESET = "sample_reset"
    GLOBAL_CARRY = "global_carry"


class TokenKind(str, Enum):
    CONTENT = "content"
    FUNCTION = "function"
    PUNCT = "punct"
    SPECIAL = "special"


class AttackStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

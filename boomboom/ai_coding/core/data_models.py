from __future__ import annotations

from dataclasses import dataclass, field

from ai_coding.core.enums import AttackStatus, TokenKind
from ai_coding.core.types import Label


@dataclass(slots=True)
class TokenInfo:
    token: str
    word_index: int
    shap_value: float


@dataclass(slots=True)
class WordInfo:
    index: int
    text: str
    phi: float = 0.0
    token_kind: TokenKind = TokenKind.CONTENT


@dataclass(slots=True)
class TextSample:
    sample_id: str
    words: list[str]
    original_label: Label
    raw_text: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    replacement_candidates: dict[str, list[str]] | None = None


@dataclass(slots=True, frozen=True)
class LocalGroup:
    anchor_index: int
    member_indices: tuple[int, int]


@dataclass(slots=True)
class GroupScore:
    group: LocalGroup
    joint_mc: float = 0.0
    variance: float = 0.0
    synergy: float = 0.0
    score: float = 0.0
    mc_samples: list[float] = field(default_factory=list)


@dataclass(slots=True)
class CalibrationState:
    alpha: float = 1.0
    lambda_value: float = 1.0


@dataclass(slots=True)
class ProbeEvaluation:
    group: LocalGroup
    success: bool
    attempts: int
    strategy: str


@dataclass(slots=True)
class CalibrationResult:
    state_before: CalibrationState
    state_after: CalibrationState
    r_joint: float
    r_syn: float
    probe_joint: list[ProbeEvaluation] = field(default_factory=list)
    probe_syn: list[ProbeEvaluation] = field(default_factory=list)
    update_reason: str = ""


@dataclass(slots=True)
class AttackTraceStep:
    query_index: int
    text_snapshot: list[str]
    predicted_label: Label | None = None
    notes: str = ""


@dataclass(slots=True)
class AttackResult:
    status: AttackStatus
    final_words: list[str]
    total_queries: int
    successful_group: LocalGroup | None = None
    successful_replacement: tuple[tuple[int, str], ...] | None = None


@dataclass(slots=True)
class ExperimentRecord:
    sample_id: str
    status: AttackStatus = AttackStatus.NOT_STARTED
    selected_seed_indices: list[int] = field(default_factory=list)
    ranked_groups: list[GroupScore] = field(default_factory=list)
    calibration_state: CalibrationState = field(default_factory=CalibrationState)
    calibration_result: CalibrationResult | None = None
    trace_steps: list[AttackTraceStep] = field(default_factory=list)
    attack_result: AttackResult | None = None

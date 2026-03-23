from __future__ import annotations

from dataclasses import dataclass, field

from ai_coding.core.constants import DEFAULT_FUNCTION_WORDS, DEFAULT_MASK_TOKEN, DEFAULT_SPECIAL_TOKENS
from ai_coding.core.enums import ResetMode


@dataclass(slots=True)
class SeedConfig:
    seed_ratio: float = 0.3
    min_seed_count: int = 1
    max_seed_ratio: float | None = None


@dataclass(slots=True)
class GroupConfig:
    window_radius: int = 2
    keep_function_words: bool = True
    special_tokens: set[str] = field(default_factory=lambda: set(DEFAULT_SPECIAL_TOKENS))
    function_words: set[str] = field(default_factory=lambda: set(DEFAULT_FUNCTION_WORDS))


@dataclass(slots=True)
class RankingConfig:
    alpha_init: float = 1.0
    lambda_init: float = 1.0
    beta: float = 0.1
    top_k_groups: int | None = None


@dataclass(slots=True)
class MonteCarloConfig:
    sample_count: int = 20
    keep_probability: float = 0.5


@dataclass(slots=True)
class CalibrationConfig:
    probe_count: int = 2
    local_query_budget: int = 2
    tau: float = 0.05
    eta: float = 0.2


@dataclass(slots=True)
class AttackConfig:
    candidate_rerank: str = "none"
    candidate_eval_limit: int | None = None
    cascade_step2_candidate_eval_limit: int | None = None
    enable_joint_replacement: bool = False
    joint_candidate_limit_per_position: int = 2
    joint_eval_limit: int | None = 4
    cascade_step2_joint_candidate_limit_per_position: int | None = None
    cascade_step2_joint_eval_limit: int | None = None
    enable_cascade_replacement: bool = False
    cascade_group_limit: int = 2
    cascade_beam_width: int = 1
    cascade_min_word_count: int = 50


@dataclass(slots=True)
class ExperimentConfig:
    mask_token: str = DEFAULT_MASK_TOKEN
    random_seed: int = 13
    reset_mode: ResetMode = ResetMode.SAMPLE_RESET
    seed: SeedConfig = field(default_factory=SeedConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)

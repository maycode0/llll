from __future__ import annotations

from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import CalibrationState
from ai_coding.core.enums import ResetMode


def initialize_calibration_state(config: ExperimentConfig, previous_state: CalibrationState | None = None) -> CalibrationState:
    if config.reset_mode == ResetMode.GLOBAL_CARRY and previous_state is not None:
        return CalibrationState(alpha=previous_state.alpha, lambda_value=previous_state.lambda_value)
    return CalibrationState(alpha=config.ranking.alpha_init, lambda_value=config.ranking.lambda_init)

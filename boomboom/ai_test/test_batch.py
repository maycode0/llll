from __future__ import annotations

from ai_coding.core.enums import ResetMode
from ai_coding.experiments.demo_data import build_demo_components
from ai_coding.experiments.pipeline import run_single_sample_pipeline, summarize_payloads


def test_single_sample_pipeline_returns_attack_payload() -> None:
    config, samples, token_map, surrogate, victim, replacer = build_demo_components()
    payload, state_after = run_single_sample_pipeline(
        samples[0],
        token_map[samples[0].sample_id],
        surrogate,
        victim,
        replacer,
        config,
    )

    assert payload["sample_id"] == samples[0].sample_id
    assert payload["attack"]["status"] in {"success", "failed"}
    assert state_after.alpha >= 0.0
    assert state_after.lambda_value >= 0.0


def test_global_carry_uses_previous_state() -> None:
    config, samples, token_map, surrogate, victim, replacer = build_demo_components()
    config.reset_mode = ResetMode.GLOBAL_CARRY

    first_payload, first_state = run_single_sample_pipeline(
        samples[0],
        token_map[samples[0].sample_id],
        surrogate,
        victim,
        replacer,
        config,
    )
    second_payload, _ = run_single_sample_pipeline(
        samples[1],
        token_map[samples[1].sample_id],
        surrogate,
        victim,
        replacer,
        config,
        previous_state=first_state,
    )

    assert second_payload["calibration"]["alpha_before"] == first_state.alpha
    assert second_payload["calibration"]["lambda_before"] == first_state.lambda_value
    assert first_state.lambda_value > 1.0


def test_global_carry_changes_second_sample_attack_queries() -> None:
    config, samples, token_map, surrogate, victim, replacer = build_demo_components()

    sample_reset_payloads = []
    config.reset_mode = ResetMode.SAMPLE_RESET
    previous_state = None
    for sample in samples:
        payload, previous_state = run_single_sample_pipeline(
            sample,
            token_map[sample.sample_id],
            surrogate,
            victim,
            replacer,
            config,
            previous_state=previous_state,
        )
        sample_reset_payloads.append(payload)

    global_carry_payloads = []
    config.reset_mode = ResetMode.GLOBAL_CARRY
    previous_state = None
    for sample in samples:
        payload, previous_state = run_single_sample_pipeline(
            sample,
            token_map[sample.sample_id],
            surrogate,
            victim,
            replacer,
            config,
            previous_state=previous_state,
        )
        global_carry_payloads.append(payload)

    assert sample_reset_payloads[1]["attack"]["total_queries"] == 1
    assert global_carry_payloads[1]["attack"]["total_queries"] == 2


def test_summarize_payloads_counts_successes() -> None:
    payloads = [
        {"attack": {"status": "success", "total_queries": 2}, "calibration": {"alpha_after": 1.0, "lambda_after": 1.2}},
        {"attack": {"status": "failed", "total_queries": 3}, "calibration": {"alpha_after": 1.0, "lambda_after": 1.0}},
    ]
    summary = summarize_payloads("sample_reset", payloads)
    assert summary["sample_count"] == 2
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 0.5

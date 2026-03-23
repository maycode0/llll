from __future__ import annotations

from ai_coding.core.config import ExperimentConfig
from ai_coding.data.io import build_replacement_generator, load_mock_surrogate, load_mock_victim, load_replacement_generator, load_text_samples


def test_load_text_samples_reads_jsonl_records() -> None:
    samples, token_map = load_text_samples("ai_inputs/demo_samples.jsonl")
    assert len(samples) == 2
    assert samples[0].sample_id == "demo-1"
    assert samples[0].raw_text == "the movie was not very good ."
    assert samples[1].metadata["domain"] == "movie"
    assert samples[1].replacement_candidates is not None
    assert samples[1].replacement_candidates["good"] == ["awful", "bad"]
    assert token_map["demo-2"][5].token == "good"


def test_load_text_samples_honors_max_samples() -> None:
    samples, token_map = load_text_samples("ai_inputs/sst2_train_samples.jsonl", max_samples=10)
    assert len(samples) == 10
    assert len(token_map) == 10
    assert samples[-1].sample_id == "sst2-train-000010"


def test_load_replacement_generator_reads_candidate_map() -> None:
    replacer = load_replacement_generator("ai_inputs/demo_replacements.json")
    assert replacer.get_candidates(["good"], 0) == ["bad", "poor"]


def test_build_replacement_generator_merges_sample_specific_candidates() -> None:
    samples, _ = load_text_samples("ai_inputs/demo_samples.jsonl")
    replacer = build_replacement_generator("ai_inputs/demo_replacements.json", samples[1])
    assert replacer.get_candidates(samples[1].words, 4) == ["awful", "bad"]
    assert replacer.get_candidates(samples[1].words, 1) == ["storyline"]


def test_load_mock_models_read_weight_configs() -> None:
    config = ExperimentConfig()
    surrogate = load_mock_surrogate("ai_inputs/demo_surrogate.json", config)
    victim = load_mock_victim("ai_inputs/demo_victim.json", config)
    assert surrogate.word_weights["good"] == 0.8
    assert victim.threshold == 1.7
    assert victim.probe_forced_labels["demo-1"]["syn"][(1, 3)] == 0


def test_load_real_prep_samples_supports_optional_fields() -> None:
    samples, token_map = load_text_samples("ai_inputs/real_prep_samples.jsonl")
    assert samples[0].metadata["dataset"] == "demo-real-prep"
    assert samples[0].replacement_candidates is not None
    assert samples[0].replacement_candidates["warm"] == ["cold"]
    assert samples[1].replacement_candidates is None
    assert token_map["prep-001"][4].token == "##ly"

from __future__ import annotations

import argparse

from ai_coding.experiments.cli import load_components_from_args


def test_load_components_from_args_uses_provided_paths() -> None:
    args = argparse.Namespace(
        samples="ai_inputs/demo_samples.jsonl",
        replacements="ai_inputs/demo_replacements.json",
        surrogate="ai_inputs/demo_surrogate.json",
        victim="ai_inputs/demo_victim.json",
        surrogate_kind="mock",
        victim_kind="mock",
        device="cpu",
        surrogate_max_length=128,
        victim_max_length=128,
        seed_ratio=0.3,
        min_seed_count=1,
        seed_max_ratio=None,
        window_radius=2,
        keep_function_words="true",
        mc_sample_count=20,
        mc_keep_probability=0.5,
        alpha_init=1.0,
        lambda_init=1.0,
        ranking_beta=0.1,
        top_k_groups=None,
        calibration_probe_count=2,
        calibration_local_query_budget=2,
        calibration_tau=0.05,
        calibration_eta=0.2,
        replacer_kind="static",
        mlm_model=r"E:\modelHub\roberta-base",
        mlm_top_k=10,
        mlm_min_score=0.01,
        mlm_relative_min_score=0.2,
        mlm_filter_stopwords="true",
        candidate_rerank="none",
        candidate_eval_limit=None,
        cascade_step2_candidate_eval_limit=None,
        enable_joint_replacement="false",
        joint_candidate_limit_per_position=2,
        joint_eval_limit=4,
        cascade_step2_joint_candidate_limit_per_position=None,
        cascade_step2_joint_eval_limit=None,
        enable_cascade_replacement="false",
        cascade_group_limit=2,
        cascade_beam_width=1,
        cascade_min_word_count=50,
        max_samples=None,
    )
    config, samples, token_map, surrogate, victim, replacer = load_components_from_args(args)
    assert samples[0].sample_id == "demo-1"
    assert token_map["demo-2"][5].token == "good"
    assert surrogate.word_weights["good"] == 0.8
    assert victim.threshold == 1.7
    assert config.seed.seed_ratio == 0.3
    assert config.seed.min_seed_count == 1
    assert config.seed.max_seed_ratio is None
    assert config.groups.window_radius == 2
    assert config.groups.keep_function_words is True
    assert config.monte_carlo.sample_count == 20
    assert config.monte_carlo.keep_probability == 0.5
    assert config.ranking.alpha_init == 1.0
    assert config.ranking.lambda_init == 1.0
    assert config.ranking.beta == 0.1
    assert config.ranking.top_k_groups is None
    assert config.calibration.probe_count == 2
    assert config.calibration.local_query_budget == 2
    assert config.calibration.tau == 0.05
    assert config.calibration.eta == 0.2
    assert config.attack.candidate_rerank == "none"
    assert config.attack.cascade_step2_candidate_eval_limit is None
    assert config.attack.enable_joint_replacement is False
    assert config.attack.cascade_step2_joint_candidate_limit_per_position is None
    assert config.attack.cascade_step2_joint_eval_limit is None
    assert config.attack.enable_cascade_replacement is False
    assert replacer.get_candidates(samples[1].words, 4)

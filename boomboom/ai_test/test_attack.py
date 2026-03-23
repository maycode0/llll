from __future__ import annotations

from ai_coding.attack.mock_replacer import MockReplacementGenerator
from ai_coding.attack.search import run_local_group_attack
from ai_coding.core.config import ExperimentConfig
from ai_coding.core.data_models import GroupScore, LocalGroup, TextSample
from ai_coding.groups.builder import build_local_groups
from ai_coding.models.mock_victim import MockVictimModel
from ai_coding.monte_carlo.estimator import estimate_group_score
from ai_coding.ranking.sorter import rank_group_scores
from ai_coding.seed.selector import select_seed_indices


def test_run_local_group_attack_succeeds_on_mock_replacement(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    sample_victim,
    experiment_config: ExperimentConfig,
) -> None:
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    ranked = rank_group_scores(
        [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups],
        experiment_config.ranking,
    )
    replacer = MockReplacementGenerator(candidates={"good": ["bad"], "not": ["never"], "very": ["slightly"]})

    attack_result, trace_steps = run_local_group_attack(sample_text_sample, ranked, sample_surrogate, sample_victim, replacer)

    assert attack_result.status.value == "success"
    assert attack_result.total_queries >= 1
    assert attack_result.successful_group is not None
    assert attack_result.successful_replacement is not None
    assert len(attack_result.successful_replacement) == 1
    assert trace_steps


def test_run_local_group_attack_returns_failed_when_no_candidates(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    sample_victim,
    experiment_config: ExperimentConfig,
) -> None:
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    ranked = rank_group_scores(
        [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups],
        experiment_config.ranking,
    )
    replacer = MockReplacementGenerator(candidates={})

    attack_result, trace_steps = run_local_group_attack(sample_text_sample, ranked, sample_surrogate, sample_victim, replacer)

    assert attack_result.status.value == "failed"
    assert attack_result.total_queries == 0
    assert trace_steps == []


def test_mock_replacer_uses_position_context() -> None:
    replacer = MockReplacementGenerator(candidates={"good": ["bad"], "plot": ["storyline"]})
    words = ["the", "plot", "was", "good"]
    assert replacer.get_candidates(words, 1) == ["storyline"]
    assert replacer.get_candidates(words, 3) == ["bad"]


def test_run_local_group_attack_applies_candidate_eval_limit(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    sample_victim,
    experiment_config: ExperimentConfig,
) -> None:
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    ranked = rank_group_scores(
        [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups],
        experiment_config.ranking,
    )
    replacer = MockReplacementGenerator(candidates={"good": ["great", "bad"], "not": ["never"], "very": ["truly"]})

    attack_result, trace_steps = run_local_group_attack(
        sample_text_sample,
        ranked,
        sample_surrogate,
        sample_victim,
        replacer,
        candidate_rerank="surrogate",
        candidate_eval_limit=1,
    )

    assert attack_result.total_queries == len(trace_steps)
    assert attack_result.total_queries <= 3


def test_run_local_group_attack_reranks_candidates_with_surrogate(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    sample_victim,
    experiment_config: ExperimentConfig,
) -> None:
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    ranked = rank_group_scores(
        [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups],
        experiment_config.ranking,
    )
    replacer = MockReplacementGenerator(candidates={"good": ["great", "bad"], "not": ["never"], "very": ["truly"]})

    attack_result, trace_steps = run_local_group_attack(
        sample_text_sample,
        ranked,
        sample_surrogate,
        sample_victim,
        replacer,
        candidate_rerank="surrogate",
        candidate_eval_limit=2,
    )

    assert attack_result.status.value == "success"
    assert trace_steps
    good_replacements = [step for step in trace_steps if "replace=5->" in step.notes]
    assert good_replacements
    assert good_replacements[0].notes.endswith("bad")


def test_run_local_group_attack_can_fallback_to_joint_replacement(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    experiment_config: ExperimentConfig,
) -> None:
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    ranked = rank_group_scores(
        [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups],
        experiment_config.ranking,
    )
    replacer = MockReplacementGenerator(candidates={"not": ["never", "barely"], "good": ["poor", "nice"], "very": ["truly"]})
    joint_only_victim = MockVictimModel(
        target_label=1,
        word_weights={"movie": 0.4, "was": 0.1, "not": 0.6, "very": 0.5, "good": 0.8, "poor": 0.2},
        pair_bonus={("good", "not"): 0.4, ("good", "very"): 0.2},
        mask_token=experiment_config.mask_token,
        threshold=1.8,
    )

    attack_result, trace_steps = run_local_group_attack(
        sample_text_sample,
        ranked,
        sample_surrogate,
        joint_only_victim,
        replacer,
        candidate_rerank="surrogate",
        candidate_eval_limit=1,
        enable_joint_replacement=True,
        joint_candidate_limit_per_position=2,
        joint_eval_limit=4,
    )

    assert attack_result.status.value == "success"
    assert attack_result.successful_replacement is not None
    assert len(attack_result.successful_replacement) == 2
    assert any("joint_replace=" in step.notes for step in trace_steps)


def test_run_local_group_attack_respects_joint_eval_limit(
    sample_text_sample,
    sample_word_infos,
    sample_surrogate,
    sample_victim,
    experiment_config: ExperimentConfig,
) -> None:
    groups = build_local_groups(sample_text_sample.words, [3, 4, 5], experiment_config.groups)
    ranked = rank_group_scores(
        [estimate_group_score(sample_text_sample, sample_word_infos, group, sample_surrogate, experiment_config) for group in groups],
        experiment_config.ranking,
    )
    replacer = MockReplacementGenerator(candidates={"not": ["never", "hardly", "barely"], "good": ["poor", "bad", "awful"], "very": ["truly"]})

    attack_result, trace_steps = run_local_group_attack(
        sample_text_sample,
        ranked,
        sample_surrogate,
        sample_victim,
        replacer,
        candidate_rerank="surrogate",
        candidate_eval_limit=1,
        enable_joint_replacement=True,
        joint_candidate_limit_per_position=2,
        joint_eval_limit=1,
    )

    assert attack_result.total_queries == len(trace_steps)
    assert sum(1 for step in trace_steps if "joint_replace=" in step.notes) <= 1


def test_select_seed_indices_respects_max_seed_ratio(sample_word_infos) -> None:
    config = ExperimentConfig()
    config.seed.seed_ratio = 0.5
    config.seed.max_seed_ratio = 0.1
    selected = select_seed_indices(sample_word_infos, config.seed)
    assert len(selected) == 1


def test_run_local_group_attack_can_cascade_to_second_group(
    sample_surrogate,
    experiment_config: ExperimentConfig,
) -> None:
    words = [
        "this",
        "movie",
        "is",
        "very",
        "good",
        "but",
        "also",
        "quite",
        "good",
        "for",
        "many",
        "viewers",
        "and",
        "families",
        "watching",
        "together",
        "at",
        "night",
        "after",
        "dinner",
        "with",
        "friends",
        "near",
        "home",
        "today",
        "because",
        "everyone",
        "still",
        "likes",
        "it",
        "a",
        "lot",
        "and",
        "finds",
        "it",
        "warm",
        "and",
        "pleasant",
        "overall",
        "for",
        "casual",
        "weekend",
        "viewing",
        "with",
        "others",
        "nearby",
        "again",
        "tonight",
        ".",
        "extra",
        "words",
        "here",
    ]
    sample = TextSample(sample_id="cascade-demo", words=words, original_label=1)

    ranked = [
        GroupScore(group=LocalGroup(anchor_index=3, member_indices=(3, 4)), joint_mc=2.0, variance=0.1, synergy=1.0, score=2.0),
        GroupScore(group=LocalGroup(anchor_index=7, member_indices=(7, 8)), joint_mc=1.8, variance=0.1, synergy=0.9, score=1.8),
    ]
    replacer = MockReplacementGenerator(candidates={"very": ["slightly"], "good": ["decent", "bad"], "quite": ["barely"]})
    cascade_victim = MockVictimModel(
        target_label=1,
        word_weights={"movie": 0.4, "very": 0.5, "good": 0.8, "bad": -0.2, "quite": 0.2, "barely": -0.2},
        pair_bonus={("good", "very"): 0.2, ("good", "quite"): 0.2},
        mask_token=experiment_config.mask_token,
        threshold=1.35,
    )

    attack_result, trace_steps = run_local_group_attack(
        sample,
        ranked,
        sample_surrogate,
        cascade_victim,
        replacer,
        candidate_rerank="surrogate",
        candidate_eval_limit=2,
        enable_cascade_replacement=True,
        cascade_group_limit=2,
        cascade_min_word_count=50,
    )

    assert trace_steps
    assert any("cascade_step=1" in step.notes for step in trace_steps)
    assert any("cascade_step=2" in step.notes for step in trace_steps)

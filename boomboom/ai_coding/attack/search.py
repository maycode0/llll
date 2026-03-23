from __future__ import annotations

from dataclasses import dataclass

from ai_coding.attack.base import ReplacementGenerator
from ai_coding.core.data_models import AttackResult, AttackTraceStep, GroupScore, TextSample
from ai_coding.core.enums import AttackStatus
from ai_coding.core.types import Label
from ai_coding.models.base import SurrogateModel, VictimModel


def _score_support_batch(surrogate: SurrogateModel, words_batch: list[list[str]], target_label: Label) -> list[float]:
    if not words_batch:
        return []
    if hasattr(surrogate, "score_label_support_batch"):
        return list(surrogate.score_label_support_batch(words_batch, target_label))
    return [surrogate.score_label_support(words, target_label) for words in words_batch]


def _predict_labels_batch(victim: VictimModel, words_batch: list[list[str]]) -> list[Label]:
    if not words_batch:
        return []
    if hasattr(victim, "predict_label_batch"):
        return list(victim.predict_label_batch(words_batch))
    return [victim.predict_label(words) for words in words_batch]


def _surrogate_mask_token(surrogate: SurrogateModel) -> str:
    classifier = getattr(surrogate, "classifier", None)
    tokenizer = getattr(classifier, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "mask_token", None):
        return tokenizer.mask_token
    return getattr(surrogate, "mask_token", "[MASK]")


def _get_replacement_candidates_batch(
    replacer: ReplacementGenerator,
    requests: list[tuple[list[str], int]],
) -> list[list[str]]:
    if not requests:
        return []
    if hasattr(replacer, "get_candidates_batch"):
        return list(replacer.get_candidates_batch(requests))
    return [replacer.get_candidates(words, position) for words, position in requests]


def rerank_candidates(
    sample: TextSample,
    position: int,
    candidates: list[str],
    surrogate: SurrogateModel,
    *,
    target_label: Label,
) -> list[str]:
    base_support = surrogate.score_label_support(sample.words, target_label)
    candidate_batches: list[list[str]] = []
    for candidate_word in candidates:
        candidate_words = list(sample.words)
        candidate_words[position] = candidate_word
        candidate_batches.append(candidate_words)
    supports = _score_support_batch(surrogate, candidate_batches, target_label)
    scored = [(base_support - support, candidate_word) for support, candidate_word in zip(supports, candidates)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate_word for _, candidate_word in scored]


def rerank_joint_candidates(
    sample: TextSample,
    positions: tuple[int, int],
    candidates_i: list[str],
    candidates_j: list[str],
    surrogate: SurrogateModel,
    *,
    target_label: Label,
    joint_eval_limit: int | None,
) -> list[tuple[str, str]]:
    base_support = surrogate.score_label_support(sample.words, target_label)
    candidate_batches: list[list[str]] = []
    candidate_pairs: list[tuple[str, str]] = []
    for candidate_i in candidates_i:
        for candidate_j in candidates_j:
            candidate_words = list(sample.words)
            candidate_words[positions[0]] = candidate_i
            candidate_words[positions[1]] = candidate_j
            candidate_batches.append(candidate_words)
            candidate_pairs.append((candidate_i, candidate_j))
    supports = _score_support_batch(surrogate, candidate_batches, target_label)
    scored = [(base_support - support, pair) for support, pair in zip(supports, candidate_pairs)]
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [pair for _, pair in scored]
    if joint_eval_limit is not None:
        ranked = ranked[:joint_eval_limit]
    return ranked


@dataclass(slots=True)
class GroupAttackOutcome:
    attack_result: AttackResult | None
    trace_steps: list[AttackTraceStep]
    best_failed_words: list[str] | None
    best_failed_delta: float | None
    total_queries: int


def _score_candidate_delta(
    words: list[str],
    surrogate: SurrogateModel,
    *,
    target_label: Label,
    base_support: float,
) -> float:
    candidate_support = surrogate.score_label_support(words, target_label)
    return base_support - candidate_support


def _attack_single_group(
    sample: TextSample,
    group_score: GroupScore,
    surrogate: SurrogateModel,
    victim: VictimModel,
    replacer: ReplacementGenerator,
    *,
    candidate_rerank: str,
    candidate_eval_limit: int | None,
    enable_joint_replacement: bool,
    joint_candidate_limit_per_position: int,
    joint_eval_limit: int | None,
    query_start: int,
    cascade_step: int | None = None,
) -> GroupAttackOutcome:
    trace_steps: list[AttackTraceStep] = []
    query_index = query_start
    target_label = sample.original_label
    base_support = surrogate.score_label_support(sample.words, target_label)
    best_failed_words: list[str] | None = None
    best_failed_delta: float | None = None

    def note_suffix() -> str:
        return f", cascade_step={cascade_step}" if cascade_step is not None else ""

    positions = list(group_score.group.member_indices)
    candidate_lists = _get_replacement_candidates_batch(replacer, [(sample.words, position) for position in positions])

    group_candidates: dict[int, list[str]] = {}
    for position, candidates in zip(positions, candidate_lists):
        if candidate_rerank == "surrogate":
            candidates = rerank_candidates(sample, position, candidates, surrogate, target_label=target_label)
        group_candidates[position] = list(candidates)
        if candidate_eval_limit is not None:
            candidates = candidates[:candidate_eval_limit]

        candidate_batches: list[list[str]] = []
        candidate_words_list: list[tuple[str, list[str]]] = []
        for candidate_word in candidates:
            candidate_words = list(sample.words)
            candidate_words[position] = candidate_word
            candidate_batches.append(candidate_words)
            candidate_words_list.append((candidate_word, candidate_words))

        predicted_labels = _predict_labels_batch(victim, candidate_batches)
        for (candidate_word, candidate_words), predicted_label in zip(candidate_words_list, predicted_labels):
            query_index += 1
            trace_steps.append(
                AttackTraceStep(
                    query_index=query_index,
                    text_snapshot=candidate_words,
                    predicted_label=predicted_label,
                    notes=(
                        f"group={group_score.group.member_indices}, replace={position}->{candidate_word}"
                        f"{note_suffix()}"
                    ),
                )
            )
            if predicted_label != target_label:
                return GroupAttackOutcome(
                    attack_result=AttackResult(
                        status=AttackStatus.SUCCESS,
                        final_words=candidate_words,
                        total_queries=query_index,
                        successful_group=group_score.group,
                        successful_replacement=((position, candidate_word),),
                    ),
                    trace_steps=trace_steps,
                    best_failed_words=None,
                    best_failed_delta=None,
                    total_queries=query_index - query_start,
                )
            delta = _score_candidate_delta(candidate_words, surrogate, target_label=target_label, base_support=base_support)
            if best_failed_delta is None or delta > best_failed_delta:
                best_failed_delta = delta
                best_failed_words = candidate_words

    if enable_joint_replacement:
        position_i, position_j = group_score.group.member_indices
        candidates_i = group_candidates.get(position_i, [])[:joint_candidate_limit_per_position]
        candidates_j = group_candidates.get(position_j, [])[:joint_candidate_limit_per_position]
        if candidates_i and candidates_j:
            ranked_pairs = rerank_joint_candidates(
                sample,
                (position_i, position_j),
                candidates_i,
                candidates_j,
                surrogate,
                target_label=target_label,
                joint_eval_limit=joint_eval_limit,
            )
            candidate_batches = []
            for candidate_i, candidate_j in ranked_pairs:
                candidate_words = list(sample.words)
                candidate_words[position_i] = candidate_i
                candidate_words[position_j] = candidate_j
                candidate_batches.append(candidate_words)
            predicted_labels = _predict_labels_batch(victim, candidate_batches)
            for (candidate_i, candidate_j), candidate_words, predicted_label in zip(ranked_pairs, candidate_batches, predicted_labels):
                query_index += 1
                trace_steps.append(
                    AttackTraceStep(
                        query_index=query_index,
                        text_snapshot=candidate_words,
                        predicted_label=predicted_label,
                        notes=(
                            f"group={group_score.group.member_indices}, "
                            f"joint_replace={position_i}->{candidate_i},{position_j}->{candidate_j}"
                            f"{note_suffix()}"
                        ),
                    )
                )
                if predicted_label != target_label:
                    return GroupAttackOutcome(
                        attack_result=AttackResult(
                            status=AttackStatus.SUCCESS,
                            final_words=candidate_words,
                            total_queries=query_index,
                            successful_group=group_score.group,
                            successful_replacement=((position_i, candidate_i), (position_j, candidate_j)),
                        ),
                        trace_steps=trace_steps,
                        best_failed_words=None,
                        best_failed_delta=None,
                        total_queries=query_index - query_start,
                    )
                delta = _score_candidate_delta(candidate_words, surrogate, target_label=target_label, base_support=base_support)
                if best_failed_delta is None or delta > best_failed_delta:
                    best_failed_delta = delta
                    best_failed_words = candidate_words

    return GroupAttackOutcome(
        attack_result=None,
        trace_steps=trace_steps,
        best_failed_words=best_failed_words,
        best_failed_delta=best_failed_delta,
        total_queries=query_index - query_start,
    )


def _rank_remaining_groups_for_sample(
    sample: TextSample,
    ranked_groups: list[GroupScore],
    consumed_indices: set[int],
    surrogate: SurrogateModel,
    *,
    target_label: Label,
) -> list[int]:
    mask_token = _surrogate_mask_token(surrogate)
    base_support = surrogate.score_label_support(sample.words, target_label)
    candidate_batches: list[list[str]] = []
    candidate_indices: list[int] = []
    for index, item in enumerate(ranked_groups):
        if index in consumed_indices:
            continue
        i, j = item.group.member_indices
        candidate_words = list(sample.words)
        candidate_words[i] = mask_token
        candidate_words[j] = mask_token
        candidate_batches.append(candidate_words)
        candidate_indices.append(index)
    supports = _score_support_batch(surrogate, candidate_batches, target_label)
    rescored: list[tuple[float, int]] = []
    for support, index in zip(supports, candidate_indices):
        item = ranked_groups[index]
        local_drop = base_support - support
        combined = local_drop + item.score
        rescored.append((combined, index))
    rescored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in rescored]


def run_local_group_attack(
    sample: TextSample,
    ranked_groups: list[GroupScore],
    surrogate: SurrogateModel,
    victim: VictimModel,
    replacer: ReplacementGenerator,
    *,
    candidate_rerank: str = "none",
    candidate_eval_limit: int | None = None,
    cascade_step2_candidate_eval_limit: int | None = None,
    enable_joint_replacement: bool = False,
    joint_candidate_limit_per_position: int = 2,
    joint_eval_limit: int | None = 4,
    cascade_step2_joint_candidate_limit_per_position: int | None = None,
    cascade_step2_joint_eval_limit: int | None = None,
    enable_cascade_replacement: bool = False,
    cascade_group_limit: int = 2,
    cascade_beam_width: int = 1,
    cascade_min_word_count: int = 50,
) -> tuple[AttackResult, list[AttackTraceStep]]:
    trace_steps: list[AttackTraceStep] = []
    query_index = 0
    cascade_enabled = enable_cascade_replacement and len(sample.words) >= cascade_min_word_count

    if not cascade_enabled:
        for group_score in ranked_groups:
            outcome = _attack_single_group(
                sample,
                group_score,
                surrogate,
                victim,
                replacer,
                candidate_rerank=candidate_rerank,
                candidate_eval_limit=candidate_eval_limit,
                enable_joint_replacement=enable_joint_replacement,
                joint_candidate_limit_per_position=joint_candidate_limit_per_position,
                joint_eval_limit=joint_eval_limit,
                query_start=query_index,
            )
            trace_steps.extend(outcome.trace_steps)
            query_index += outcome.total_queries
            if outcome.attack_result is not None:
                outcome.attack_result.total_queries = query_index
                return outcome.attack_result, trace_steps
        return (
            AttackResult(
                status=AttackStatus.FAILED,
                final_words=list(sample.words),
                total_queries=query_index,
            ),
            trace_steps,
        )

    max_groups = min(len(ranked_groups), cascade_group_limit)
    active_states: list[tuple[TextSample, list[tuple[int, str]], set[int]]] = [(sample, [], set())]

    for cascade_step in range(1, max_groups + 1):
        next_states: list[tuple[float, TextSample, list[tuple[int, str]], set[int]]] = []
        for current_sample, accumulated_replacements, consumed_indices in active_states:
            remaining_order = _rank_remaining_groups_for_sample(
                current_sample,
                ranked_groups,
                consumed_indices,
                surrogate,
                target_label=current_sample.original_label,
            )
            if not remaining_order:
                continue
            group_index = remaining_order[0]
            group_score = ranked_groups[group_index]
            outcome = _attack_single_group(
                current_sample,
                group_score,
                surrogate,
                victim,
                replacer,
                candidate_rerank=candidate_rerank,
                candidate_eval_limit=(
                    cascade_step2_candidate_eval_limit
                    if cascade_step > 1 and cascade_step2_candidate_eval_limit is not None
                    else candidate_eval_limit
                ),
                enable_joint_replacement=enable_joint_replacement,
                joint_candidate_limit_per_position=(
                    cascade_step2_joint_candidate_limit_per_position
                    if cascade_step > 1 and cascade_step2_joint_candidate_limit_per_position is not None
                    else joint_candidate_limit_per_position
                ),
                joint_eval_limit=(
                    cascade_step2_joint_eval_limit
                    if cascade_step > 1 and cascade_step2_joint_eval_limit is not None
                    else joint_eval_limit
                ),
                query_start=query_index,
                cascade_step=cascade_step,
            )
            trace_steps.extend(outcome.trace_steps)
            query_index += outcome.total_queries

            if outcome.attack_result is not None:
                combined_replacements = accumulated_replacements + list(outcome.attack_result.successful_replacement or ())
                return (
                    AttackResult(
                        status=outcome.attack_result.status,
                        final_words=outcome.attack_result.final_words,
                        total_queries=query_index,
                        successful_group=outcome.attack_result.successful_group,
                        successful_replacement=tuple(combined_replacements) if combined_replacements else None,
                    ),
                    trace_steps,
                )

            if outcome.best_failed_words is None:
                continue
            updated_replacements = accumulated_replacements + [
                (index, word)
                for index, (original, word) in enumerate(zip(current_sample.words, outcome.best_failed_words))
                if original != word
            ]
            next_sample = TextSample(
                sample_id=sample.sample_id,
                words=list(outcome.best_failed_words),
                original_label=sample.original_label,
                raw_text=sample.raw_text,
                metadata=dict(sample.metadata),
                replacement_candidates=sample.replacement_candidates,
            )
            next_states.append(
                (
                    outcome.best_failed_delta or 0.0,
                    next_sample,
                    updated_replacements,
                    set(consumed_indices) | {group_index},
                )
            )

        if not next_states:
            break
        next_states.sort(key=lambda item: item[0], reverse=True)
        active_states = [(sample_state, replacements, consumed) for _, sample_state, replacements, consumed in next_states[:cascade_beam_width]]

    return (
        AttackResult(
            status=AttackStatus.FAILED,
            final_words=list(active_states[0][0].words) if active_states else list(sample.words),
            total_queries=query_index,
        ),
        trace_steps,
    )

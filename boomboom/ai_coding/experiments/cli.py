from __future__ import annotations

import argparse
from pathlib import Path


def add_common_input_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--samples", default="ai_inputs/demo_samples.jsonl", help="Path to the sample JSONL file")
    parser.add_argument("--replacements", default="ai_inputs/demo_replacements.json", help="Path to the replacement JSON file")
    parser.add_argument("--surrogate", default="ai_inputs/demo_surrogate.json", help="Path to the surrogate config JSON or local model directory")
    parser.add_argument("--victim", default="ai_inputs/demo_victim.json", help="Path to the victim config JSON or local model directory")
    parser.add_argument("--surrogate-kind", choices=("mock", "hf"), default="mock", help="Surrogate backend type")
    parser.add_argument("--victim-kind", choices=("mock", "hf"), default="mock", help="Victim backend type")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Inference device for real models")
    parser.add_argument("--surrogate-max-length", type=int, default=128, help="Tokenizer max length for the surrogate model")
    parser.add_argument("--victim-max-length", type=int, default=128, help="Tokenizer max length for the victim model")
    parser.add_argument("--seed-ratio", type=float, default=0.3, help="Base proportion of words used as seeds")
    parser.add_argument("--min-seed-count", type=int, default=1, help="Minimum number of seed words")
    parser.add_argument("--seed-max-ratio", type=float, default=None, help="Optional upper bound on the seed proportion for long texts")
    parser.add_argument("--window-radius", type=int, default=2, help="Neighborhood radius used to form local groups")
    parser.add_argument("--keep-function-words", choices=("true", "false"), default="true", help="Whether function words may appear in local groups")
    parser.add_argument("--mc-sample-count", type=int, default=20, help="Monte Carlo sample count for group estimation")
    parser.add_argument("--mc-keep-probability", type=float, default=0.5, help="Monte Carlo keep probability")
    parser.add_argument("--alpha-init", type=float, default=1.0, help="Initial alpha weight for ranking")
    parser.add_argument("--lambda-init", type=float, default=1.0, help="Initial lambda weight for ranking")
    parser.add_argument("--ranking-beta", type=float, default=0.1, help="Variance penalty coefficient in ranking")
    parser.add_argument("--top-k-groups", type=int, default=None, help="Optional cap on the number of groups kept after ranking")
    parser.add_argument("--calibration-probe-count", type=int, default=2, help="Number of probe groups per calibration strategy")
    parser.add_argument("--calibration-local-query-budget", type=int, default=2, help="Per-probe local query budget during calibration")
    parser.add_argument("--calibration-tau", type=float, default=0.05, help="Calibration update threshold")
    parser.add_argument("--calibration-eta", type=float, default=0.2, help="Calibration step size")
    parser.add_argument("--replacer-kind", choices=("static", "mlm"), default="static", help="Replacement candidate source")
    parser.add_argument("--mlm-model", default=r"E:\modelHub\roberta-base", help="Local masked language model directory")
    parser.add_argument("--mlm-top-k", type=int, default=10, help="Top-k dynamic replacement candidates per position")
    parser.add_argument("--mlm-min-score", type=float, default=0.01, help="Minimum MLM candidate probability to keep")
    parser.add_argument("--mlm-relative-min-score", type=float, default=0.2, help="Minimum candidate score ratio relative to the best MLM candidate")
    parser.add_argument("--mlm-filter-stopwords", choices=("true", "false"), default="true", help="Whether to filter common stopword-like MLM candidates")
    parser.add_argument("--candidate-rerank", choices=("none", "surrogate"), default="none", help="Optional reranking strategy for generated candidates")
    parser.add_argument("--candidate-eval-limit", type=int, default=None, help="Maximum number of candidates to test per position after reranking")
    parser.add_argument("--cascade-step2-candidate-eval-limit", type=int, default=None, help="Optional candidate limit override for cascade steps after the first")
    parser.add_argument("--enable-joint-replacement", choices=("true", "false"), default="false", help="Whether to try two-word joint replacement after single-word attempts fail")
    parser.add_argument("--joint-candidate-limit-per-position", type=int, default=2, help="Maximum reranked candidates per position used for joint replacement")
    parser.add_argument("--joint-eval-limit", type=int, default=4, help="Maximum number of joint replacement combinations to test per group")
    parser.add_argument("--cascade-step2-joint-candidate-limit-per-position", type=int, default=None, help="Optional joint per-position candidate override for later cascade steps")
    parser.add_argument("--cascade-step2-joint-eval-limit", type=int, default=None, help="Optional joint evaluation limit override for later cascade steps")
    parser.add_argument("--enable-cascade-replacement", choices=("true", "false"), default="false", help="Whether to cascade across multiple high-ranked groups for long texts")
    parser.add_argument("--cascade-group-limit", type=int, default=2, help="Maximum number of ranked groups used in cascade mode")
    parser.add_argument("--cascade-beam-width", type=int, default=1, help="Number of intermediate failed states retained in cascade mode")
    parser.add_argument("--cascade-min-word-count", type=int, default=50, help="Minimum word count required before cascade mode activates")
    parser.add_argument("--max-samples", type=int, default=None, help="Load only the first N samples from the JSONL file")
    return parser


def load_components_from_args(args: argparse.Namespace):
    from ai_coding.experiments.demo_data import build_demo_components

    return build_demo_components(
        samples_path=Path(args.samples),
        replacements_path=Path(args.replacements),
        surrogate_path=Path(args.surrogate),
        victim_path=Path(args.victim),
        surrogate_kind=args.surrogate_kind,
        victim_kind=args.victim_kind,
        device=args.device,
        surrogate_max_length=args.surrogate_max_length,
        victim_max_length=args.victim_max_length,
        seed_ratio=args.seed_ratio,
        min_seed_count=args.min_seed_count,
        seed_max_ratio=args.seed_max_ratio,
        window_radius=args.window_radius,
        keep_function_words=args.keep_function_words == "true",
        monte_carlo_sample_count=args.mc_sample_count,
        monte_carlo_keep_probability=args.mc_keep_probability,
        alpha_init=args.alpha_init,
        lambda_init=args.lambda_init,
        ranking_beta=args.ranking_beta,
        top_k_groups=args.top_k_groups,
        calibration_probe_count=args.calibration_probe_count,
        calibration_local_query_budget=args.calibration_local_query_budget,
        calibration_tau=args.calibration_tau,
        calibration_eta=args.calibration_eta,
        replacer_kind=args.replacer_kind,
        mlm_model_path=Path(args.mlm_model),
        mlm_top_k=args.mlm_top_k,
        mlm_min_score=args.mlm_min_score,
        mlm_relative_min_score=args.mlm_relative_min_score,
        mlm_filter_stopwords=args.mlm_filter_stopwords == "true",
        candidate_rerank=args.candidate_rerank,
        candidate_eval_limit=args.candidate_eval_limit,
        cascade_step2_candidate_eval_limit=args.cascade_step2_candidate_eval_limit,
        enable_joint_replacement=args.enable_joint_replacement == "true",
        joint_candidate_limit_per_position=args.joint_candidate_limit_per_position,
        joint_eval_limit=args.joint_eval_limit,
        cascade_step2_joint_candidate_limit_per_position=args.cascade_step2_joint_candidate_limit_per_position,
        cascade_step2_joint_eval_limit=args.cascade_step2_joint_eval_limit,
        enable_cascade_replacement=args.enable_cascade_replacement == "true",
        cascade_group_limit=args.cascade_group_limit,
        cascade_beam_width=args.cascade_beam_width,
        cascade_min_word_count=args.cascade_min_word_count,
        max_samples=args.max_samples,
    )

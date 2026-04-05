"""Evaluator for RewardBench 2."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from rcl.core.data_structures import EvaluationResult, ExecutionTrace
from rcl.core.interfaces import Evaluator


def _compute_ties_prompt_stats(
    scores: List[float],
    num_correct: int,
) -> Tuple[bool, float | None, float | None]:
    """Official RewardBench 2 helper for one ties prompt."""

    correct_scores = list(scores[:num_correct])
    incorrect_scores = list(scores[num_correct:])
    if not correct_scores or not incorrect_scores:
        return False, None, None

    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0
    return accurate, different_correct_margin, correct_incorrect_margin


def compute_official_ties_score(traces: Iterable[ExecutionTrace]) -> float:
    """Official RewardBench 2 ties scoring adapted from the reference code."""

    grouped: Dict[Tuple[str, int], Tuple[List[float], int]] = {}
    for trace in traces:
        task_id = trace.task_id
        if ":" not in task_id:
            continue
        sample_type, prompt_id_str = task_id.split(":", 1)
        try:
            prompt_id = int(prompt_id_str)
        except ValueError:
            continue

        scores = trace.metadata.get("candidate_scores") or []
        num_correct = len(trace.metadata.get("correct_ids") or [])
        grouped[(sample_type, prompt_id)] = (list(scores), num_correct)

    ref_stats = {}
    tied_stats = {}
    for (sample_type, prompt_id), (scores, num_correct) in grouped.items():
        stats = _compute_ties_prompt_stats(scores, num_correct)
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        elif sample_type == "tied":
            tied_stats[prompt_id] = stats

    ref_accuracy = (
        sum(1.0 for accurate, _, _ in ref_stats.values() if accurate) / len(ref_stats)
        if ref_stats
        else 0.0
    )
    tied_accuracy = (
        sum(1.0 for accurate, _, _ in tied_stats.values() if accurate) / len(tied_stats)
        if tied_stats
        else 0.0
    )

    paired_prompt_ids = sorted(set(ref_stats) & set(tied_stats))
    if not paired_prompt_ids:
        return tied_accuracy

    diff_corr_margin = [tied_stats[prompt_id][1] for prompt_id in paired_prompt_ids]
    corr_incorrect_ties = [tied_stats[prompt_id][2] for prompt_id in paired_prompt_ids]
    corr_incorrect_ref = [ref_stats[prompt_id][2] for prompt_id in paired_prompt_ids]

    preferred_values = []
    preferred_hard_values = []
    margin_values = []
    for diff_margin, tie_margin, ref_margin in zip(
        diff_corr_margin,
        corr_incorrect_ties,
        corr_incorrect_ref,
    ):
        if diff_margin is None or tie_margin is None or ref_margin is None:
            continue
        preferred_values.append(1.0 if tie_margin > diff_margin else 0.0)
        preferred_hard_values.append(1.0 if min(ref_margin, tie_margin) > diff_margin else 0.0)
        if math.isclose(diff_margin, 0.0):
            margin_values.append(0.0)
        else:
            margin_values.append(math.tanh(min(ref_margin, tie_margin) / diff_margin - 1.0))

    correctness_preferred = (
        sum(preferred_values) / len(preferred_values) if preferred_values else 0.0
    )
    correctness_preferred_hard = (
        sum(preferred_hard_values) / len(preferred_hard_values) if preferred_hard_values else 0.0
    )
    correctness_margin_score = (
        sum(margin_values) / len(margin_values) if margin_values else 0.0
    )

    return float(
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )


class RewardBench2Evaluator(Evaluator):
    """Aggregate RewardBench 2 traces into prompt-level and official scores."""

    def evaluate(self, traces: List[ExecutionTrace]) -> EvaluationResult:
        if not traces:
            return EvaluationResult(score=0.0, tgc=0.0, per_instance_scores=[], traces=[])

        per_instance_scores = [trace.metadata.get("pass_pct", 0.0) for trace in traces]
        avg_pairwise_score = sum(per_instance_scores) / len(per_instance_scores)
        prompt_accuracy = sum(
            1.0 for trace in traces if trace.metadata.get("task_completed", False)
        ) / len(traces)

        subset_groups: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        for trace in traces:
            subset_groups[trace.metadata.get("subset", "unknown")].append(trace)

        subset_scores: Dict[str, float] = {}
        subset_counts: Dict[str, int] = {}
        for subset, subset_traces in sorted(subset_groups.items()):
            subset_counts[subset] = len(subset_traces)
            if subset.lower() == "ties":
                subset_scores[subset] = compute_official_ties_score(subset_traces)
            else:
                winner_fractions = [
                    trace.metadata.get("winner_fraction", trace.metadata.get("pass_pct", 0.0))
                    for trace in subset_traces
                ]
                subset_scores[subset] = sum(winner_fractions) / len(winner_fractions)

        leaderboard_score = (
            sum(subset_scores.values()) / len(subset_scores) if subset_scores else 0.0
        )

        return EvaluationResult(
            score=avg_pairwise_score,
            tgc=prompt_accuracy,
            per_instance_scores=per_instance_scores,
            traces=traces,
            metadata={
                "leaderboard_score": leaderboard_score,
                "subset_scores": subset_scores,
                "subset_counts": subset_counts,
                "avg_pairwise_score": avg_pairwise_score,
                "prompt_accuracy": prompt_accuracy,
                "ties_official_score": subset_scores.get("Ties"),
            },
        )

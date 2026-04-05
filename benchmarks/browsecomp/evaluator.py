"""Evaluator for BrowseComp+ tasks.

Uses LLM-judge correctness as the primary metric.
BrowseComp+ queries are independent (no scenario grouping), so we compute:
- accuracy (fraction correct) as the score
- TGC maps to accuracy (binary correct/incorrect per query)
"""

from typing import List

from rcl.core.data_structures import EvaluationResult, ExecutionTrace
from rcl.core.interfaces import Evaluator


class BrowseCompEvaluator(Evaluator):
    """Evaluator that computes accuracy-based metrics from judge results.

    Metrics:
    - score: Accuracy (fraction of queries judged correct), maps to pass_pct
    - tgc: Same as score (binary correct/incorrect = task completion)
    - per_instance_scores: 1.0 if correct, 0.0 if not
    """

    def evaluate(self, traces: List[ExecutionTrace]) -> EvaluationResult:
        if not traces:
            return EvaluationResult(score=0.0, tgc=0.0, per_instance_scores=[], traces=[])

        per_instance_scores = []
        correct_count = 0
        error_count = 0

        for trace in traces:
            score = trace.metadata.get("pass_pct", 0.0)
            per_instance_scores.append(score)

            if score >= 0.999:
                correct_count += 1
            if trace.metadata.get("error"):
                error_count += 1

        accuracy = correct_count / len(traces)

        return EvaluationResult(
            score=accuracy,
            tgc=accuracy,  # For binary tasks, TGC = accuracy
            per_instance_scores=per_instance_scores,
            traces=traces,
            metadata={
                "num_queries": len(traces),
                "correct_count": correct_count,
                "error_count": error_count,
                "accuracy_pct": accuracy * 100,
            },
        )

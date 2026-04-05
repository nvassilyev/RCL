"""Evaluator for AppWorld tasks."""

from collections import defaultdict
from typing import List

from rcl.core.data_structures import EvaluationResult, ExecutionTrace
from rcl.core.interfaces import Evaluator


def _task_id_to_scenario_id(task_id: str) -> str:
    """Extract scenario ID from task ID (e.g., '325d6ec_1' -> '325d6ec')."""
    return task_id.rsplit("_", 1)[0]


class AppWorldEvaluator(Evaluator):
    """Evaluator that computes pass percentage, TGC, and SGC from traces.

    Metrics:
    - score: Average pass percentage across all tasks
    - tgc: Task Goal Completion rate (fraction of tasks with 100% pass)
    - sgc: Scenario Goal Completion rate (fraction of scenarios where all tasks pass)
    - per_instance_scores: Pass percentage for each task
    """

    def evaluate(self, traces: List[ExecutionTrace]) -> EvaluationResult:
        """Evaluate execution traces.

        Args:
            traces: Execution traces from SystemAdapter

        Returns:
            EvaluationResult with score, TGC, SGC, and per-instance scores
        """
        if not traces:
            return EvaluationResult(score=0.0, tgc=0.0, per_instance_scores=[], traces=[])

        # Filter out infrastructure errors (server crashes, disconnects) that
        # don't reflect agent quality.  These are marked by the system adapter
        # with infra_error=True, or detected by error + num_turns==0.
        scoreable_traces = []
        infra_error_count = 0
        for trace in traces:
            is_infra = trace.metadata.get("infra_error", False)
            # Also detect unmarked infra errors: error present + zero turns
            if not is_infra and trace.metadata.get("error") and trace.metadata.get("num_turns", 0) == 0:
                is_infra = True
            if is_infra:
                infra_error_count += 1
            else:
                scoreable_traces.append(trace)

        if infra_error_count > 0:
            print(f"    [Evaluator] Excluded {infra_error_count}/{len(traces)} infra errors from scoring")

        # Fall back to all traces if everything was infra errors
        eval_traces = scoreable_traces if scoreable_traces else traces

        per_instance_scores = []
        tgc_count = 0
        scenario_passes = defaultdict(list)

        for trace in eval_traces:
            pass_pct = trace.metadata.get("pass_pct", 0.0)
            task_passes_fully = pass_pct >= 0.999

            per_instance_scores.append(pass_pct)
            if task_passes_fully:
                tgc_count += 1

            scenario_id = _task_id_to_scenario_id(trace.task_id)
            scenario_passes[scenario_id].append(task_passes_fully)

        avg_score = sum(per_instance_scores) / len(per_instance_scores)
        tgc = tgc_count / len(eval_traces)

        # SGC: fraction of complete scenarios (all 3 variants present) where ALL tasks pass
        complete_scenarios = {sid: passes for sid, passes in scenario_passes.items() if len(passes) == 3}
        if complete_scenarios:
            sgc = sum(1 for passes in complete_scenarios.values() if all(passes)) / len(complete_scenarios)
        else:
            sgc = 0.0
        num_scenarios = len(complete_scenarios)

        return EvaluationResult(
            score=avg_score,
            tgc=tgc,
            per_instance_scores=per_instance_scores,
            traces=traces,
            metadata={"sgc": sgc, "num_complete_scenarios": num_scenarios},
        )

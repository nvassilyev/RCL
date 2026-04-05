"""SystemAdapter for RewardBench 2."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from rcl.core.data_structures import ExecutionTrace, Playbook
from rcl.core.interfaces import SystemAdapter
from rcl.core.trace_writer import (
    TraceWriter,
    build_rollout_descriptors,
    rollout_metadata,
)

from .rewardbench2_client import (
    BASE_SYSTEM_INSTRUCTIONS,
    RewardBench2Client,
    RewardBench2Config,
    RewardBench2Result,
    RewardBench2Task,
)

logger = logging.getLogger(__name__)

PLAYBOOK_HEADER = """

# Playbook

You have been given a curated playbook of judging strategies, failure patterns,
and calibration rules. Read it carefully and apply it while rating candidates.
"""


def _build_system_prompt(playbook: Playbook) -> str:
    parts = [BASE_SYSTEM_INSTRUCTIONS]
    if len(playbook) > 0:
        parts.append(PLAYBOOK_HEADER + playbook.to_prompt())
    return "\n".join(parts)


class RewardBench2SystemAdapter(SystemAdapter):
    """Run RewardBench 2 judging tasks with benchmark-local concurrency."""

    def __init__(
        self,
        config: RewardBench2Config,
        trace_writer: Optional[TraceWriter] = None,
    ):
        self.config = config
        self.trace_writer = trace_writer
        self.client = RewardBench2Client(config)
        self._task_map: Optional[Dict[str, RewardBench2Task]] = None

    def execute(
        self,
        task_ids: List[str],
        playbook: Playbook,
        experiment_prefix: str = "eval",
        max_workers: int = 1,
        verbose: bool = True,
        trace_subdir: str = "traces",
        playbook_overrides: Optional[Dict[str, Playbook]] = None,
    ) -> List[ExecutionTrace]:
        if not task_ids:
            return []

        _overrides = playbook_overrides or {}
        task_map = self._get_task_map()
        rollout_descriptors = build_rollout_descriptors(task_ids)

        # Strip __pp__ prefix for task lookup (optimizer adds it for playbook_overrides routing)
        def _resolve_task_id(tid: str) -> str:
            return tid.removeprefix("__pp__")

        execution_specs = []
        for task_id, rollout in zip(task_ids, rollout_descriptors):
            resolved_id = _resolve_task_id(task_id)
            task = task_map.get(resolved_id)
            task_playbook = _overrides.get(task_id, playbook)
            execution_specs.append(
                {
                    "task_id": task_id,
                    "resolved_id": resolved_id,
                    "task": task,
                    "system_prompt": _build_system_prompt(task_playbook),
                    "rollout": rollout,
                    "result": None,
                }
            )

        runnable_specs = [spec for spec in execution_specs if spec["task"] is not None]
        system_prompt = _build_system_prompt(playbook)

        if verbose:
            print(
                f"    Running {len(runnable_specs)} RewardBench2 tasks "
                f"with {self.config.n_concurrent} workers",
                flush=True,
            )

        # Run tasks with per-task system prompts if overrides exist
        if _overrides:
            # Group tasks by system prompt to batch efficiently
            prompt_groups = defaultdict(list)
            for spec in runnable_specs:
                prompt_groups[spec["system_prompt"]].append(spec)

            for prompt, group_specs in prompt_groups.items():
                group_results = self.client.run_tasks(
                    [spec["task"] for spec in group_specs],
                    prompt,
                )
                for spec, result in zip(group_specs, group_results):
                    spec["result"] = result
        else:
            results = self.client.run_tasks(
                [spec["task"] for spec in runnable_specs],
                system_prompt,
            )
            for spec, result in zip(runnable_specs, results):
                spec["result"] = result

        traces: List[ExecutionTrace] = []
        for spec in execution_specs:
            task_id = spec["task_id"]
            task = spec["task"]
            result = spec["result"]
            rollout = spec["rollout"]
            if task is None or result is None:
                trace = ExecutionTrace(
                    task_id=task_id,
                    input_query=task_id,
                    system_output=None,
                    trace="Error: missing result",
                    metadata={
                        "pass_pct": 0.0,
                        "task_completed": False,
                        "error": "missing_result",
                        "evaluation_details": "Error: missing result",
                    },
                )
            else:
                trace = self._build_trace(task, result)
                # Preserve original task_id (may include __pp__ prefix)
                trace.task_id = task_id

            trace.metadata.update(rollout_metadata(rollout))
            traces.append(trace)

            if verbose:
                status = "PASS" if trace.metadata.get("task_completed") else "FAIL"
                pct = trace.metadata.get("pass_pct", 0.0) * 100
                print(
                    f"    [{task_id}] subset={trace.metadata.get('subset')} "
                    f"pairwise={pct:.1f}% {status}",
                    flush=True,
                )

            if self.trace_writer:
                self.trace_writer.write_trace(
                    task_id,
                    {
                        "task_id": task_id,
                        "subset": trace.metadata.get("subset"),
                        "prompt": trace.input_query,
                        "result": {
                            "pass_pct": trace.metadata.get("pass_pct", 0.0),
                            "task_completed": trace.metadata.get("task_completed", False),
                            "winner_fraction": trace.metadata.get("winner_fraction", 0.0),
                            "prompt_accurate": trace.metadata.get("prompt_accurate", False),
                            "duration_s": trace.metadata.get("duration_s", 0.0),
                        },
                        "candidate_scores": trace.metadata.get("candidate_scores", []),
                        "best_response_ids": trace.metadata.get("best_response_ids", []),
                        "correct_ids": trace.metadata.get("correct_ids", []),
                        "judge_reasoning": trace.metadata.get("judge_reasoning", ""),
                        "raw_response": trace.metadata.get("raw_response", ""),
                        "rollout": rollout_metadata(rollout),
                    },
                    subdir=trace_subdir,
                    artifact_id=str(rollout["artifact_id"]),
                )

        return traces

    def _build_trace(self, task: RewardBench2Task, result: RewardBench2Result) -> ExecutionTrace:
        if result.error:
            return ExecutionTrace(
                task_id=task.task_id,
                input_query=task.prompt,
                system_output=None,
                trace=self._format_failed_trace(task, result),
                metadata={
                    "subset": task.subset,
                    "pass_pct": 0.0,
                    "task_completed": False,
                    "error": result.error,
                    "infra_error": result.infra_error,
                    "evaluation_details": f"Error: {result.error}",
                    "duration_s": round(result.duration_sec, 2),
                },
            )

        ordered_scores = [result.ratings[candidate_id] for candidate_id in task.candidate_ids]
        eval_details = self._format_evaluation_details(task, result, ordered_scores)

        return ExecutionTrace(
            task_id=task.task_id,
            input_query=task.prompt,
            system_output=result.best_response_ids,
            trace=self._format_trace(task, result),
            metadata={
                "subset": task.subset,
                "pass_pct": result.pairwise_accuracy,
                "task_completed": result.prompt_accurate,
                "winner_fraction": result.winner_fraction,
                "prompt_accurate": result.prompt_accurate,
                "candidate_scores": ordered_scores,
                "candidate_ids": list(task.candidate_ids),
                "correct_ids": list(task.correct_ids),
                "best_response_ids": list(result.best_response_ids),
                "judge_reasoning": result.reasoning,
                "raw_response": result.raw_response,
                "evaluation_details": eval_details,
                "duration_s": round(result.duration_sec, 2),
            },
        )

    def _format_trace(self, task: RewardBench2Task, result: RewardBench2Result) -> str:
        lines = [f"## Prompt\n{task.prompt}", "", "## Candidate Responses"]
        for candidate_id, candidate in zip(task.candidate_ids, task.candidates):
            lines.append(f"\n[{candidate_id}]\n{candidate}")
        lines.extend(
            [
                "",
                "## Judge Reasoning",
                result.reasoning or "(no reasoning provided)",
                "",
                "## Judge Output JSON",
                json.dumps(result.parsed_response or {}, indent=2, ensure_ascii=False),
            ]
        )
        return "\n".join(lines)

    def _format_failed_trace(self, task: RewardBench2Task, result: RewardBench2Result) -> str:
        lines = [f"## Prompt\n{task.prompt}", "", "## Candidate Responses"]
        for candidate_id, candidate in zip(task.candidate_ids, task.candidates):
            lines.append(f"\n[{candidate_id}]\n{candidate}")
        lines.extend(["", "## Raw Judge Output", result.raw_response or f"Error: {result.error}"])
        return "\n".join(lines)

    def _format_evaluation_details(
        self,
        task: RewardBench2Task,
        result: RewardBench2Result,
        ordered_scores: List[float],
    ) -> str:
        lines = [
            f"Subset: {task.subset}",
            f"Pairwise preference accuracy: {result.pairwise_accuracy * 100:.1f}%",
            f"Prompt accurate (all correct > all incorrect): {result.prompt_accurate}",
            f"Winner fraction: {result.winner_fraction:.3f}",
            f"Correct candidate ids: {', '.join(task.correct_ids)}",
            f"Top-rated candidate ids: {', '.join(result.best_response_ids) or '(none)'}",
            "",
            "Candidate scores:",
        ]
        for candidate_id, score in zip(task.candidate_ids, ordered_scores):
            tag = "CORRECT" if candidate_id in task.correct_ids else "INCORRECT"
            lines.append(f"  [{candidate_id}] {tag} score={score:.3f}")
        return "\n".join(lines)

    def get_ground_truth(self, task_id: str) -> Optional[str]:
        task = self._get_task_map().get(task_id)
        if task is None:
            return None
        return "\n\n".join(task.chosen)

    def load_tasks(self, split: str = "train", limit: Optional[int] = None) -> List[str]:
        task_ids = self.client.load_split_ids(split)
        if limit:
            task_ids = task_ids[:limit]
        return task_ids

    def _get_task_map(self) -> Dict[str, RewardBench2Task]:
        if self._task_map is None:
            self._task_map = self.client.load_task_map()
        return self._task_map

    def clone_for_parallel(self) -> "RewardBench2SystemAdapter":
        """Create an isolated adapter for frontier slot parallelism."""
        cloned = RewardBench2SystemAdapter(
            config=self.config,
            trace_writer=self.trace_writer,
        )
        cloned._task_map = self._task_map
        return cloned

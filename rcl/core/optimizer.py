"""RCL Optimizer — composable optimization loop.

Features (PP, Group, Replay, Dual-trace, Optimizer State) are controlled
via RCLConfig flags and compose freely.
"""

import json
import math
import random
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import RCLConfig
from .data_structures import EvaluationResult, ExecutionTrace, Playbook
from .interfaces import Evaluator, Mutator, Reflector, SystemAdapter
from .replay_buffer import ReplayBuffer
from .trace_writer import TraceWriter
from ..components.llm_client import is_content_filtered_llm_error


class RCLOptimizer:
    """RCL optimizer with composable primitives.

    Primitives controlled by config flags:
    - Dual-trace (dual_trace): Credit assignment via annotated + baseline traces
    - Group (group_size > 1):  Grouped rollouts per task with contrastive signal
    - Replay (failure_replay_ratio > 0): Failure replay buffer
    - PP (perturbation_set):   Injects perturbation instructions during execution
    - Optimization state (use_optimization_state): Rolling optimizer state document
    """

    def __init__(
        self,
        system_adapter: SystemAdapter,
        evaluator: Evaluator,
        reflector: Reflector,
        mutator: Mutator,
        config: Optional[RCLConfig] = None,
        trace_writer: Optional[TraceWriter] = None,
        trace_summarizer=None,
    ):
        self.system_adapter = system_adapter
        self.evaluator = evaluator
        self.reflector = reflector
        self.mutator = mutator
        self.config = config or RCLConfig()
        self.trace_writer = trace_writer
        self.trace_summarizer = trace_summarizer

        # Training sampler state
        self._single_pass_order: List[str] = []
        self._single_pass_idx: int = 0
        self._replay = ReplayBuffer(
            replay_ratio=self.config.failure_replay_ratio,
            max_size=self.config.failure_replay_max_size,
            passes_to_graduate=self.config.failure_replay_passes_to_graduate,
            failures_to_evict=self.config.failure_replay_failures_to_evict,
            unseen_first=self.config.failure_replay_unseen_first,
        )

        # Optimization state manager (async background update)
        self._optimization_state_mgr = None
        self._opt_state_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opt_state")
        self._opt_state_future: Optional[Future] = None
        self._opt_state_iter_dir: Optional[str] = None
        self._opt_state_launch_time: float = 0.0
        if self.config.use_optimization_state:
            from .optimization_state import OptimizationStateManager
            self._optimization_state_mgr = OptimizationStateManager(
                model=self.config.optimization_state_model or self.config.reflector_model,
                target_model_name=self.config.model,
                max_history=self.config.optimization_state_max_history,
            )

        # PP: configure perturbations
        self._perturbation_names: List[str] = []
        if self.config.perturbation_set:
            from ..components.perturber import PERTURBATION_SETS, build_reflector_tag_note
            ps = self.config.perturbation_set
            if ps in PERTURBATION_SETS:
                self._perturbation_names = PERTURBATION_SETS[ps]
            else:
                self._perturbation_names = [p.strip() for p in ps.split(",")]
            tag_note = build_reflector_tag_note(self._perturbation_names)
            if tag_note:
                if hasattr(self.reflector, 'prompt_template'):
                    self.reflector.prompt_template += tag_note
                elif hasattr(self.reflector, 'tag_note'):
                    self.reflector.tag_note = tag_note
        elif self.config.dual_trace:
            from ..components.perturber import PERTURBATION_SETS, build_reflector_tag_note
            tag_note = build_reflector_tag_note(PERTURBATION_SETS["full"])
            if tag_note:
                if hasattr(self.reflector, 'prompt_template'):
                    self.reflector.prompt_template += tag_note
                elif hasattr(self.reflector, 'tag_note'):
                    self.reflector.tag_note = tag_note

    def close(self) -> None:
        """Shut down background thread pools."""
        if self._opt_state_executor is not None:
            self._opt_state_executor.shutdown(wait=False)
            self._opt_state_executor = None

    def __del__(self):
        self.close()

    # ── Main optimization loop ───────────────────────────────────

    def optimize(
        self,
        seed_playbook: Playbook,
        train_task_ids: List[str],
        val_task_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        start_iteration: int = 0,
        existing_history: Optional[List[Dict]] = None,
        best_val_score_init: float = 0.0,
    ) -> Playbook:
        """Run RCL optimization loop. Returns the best playbook found."""
        config = self.config
        output_dir = output_dir or config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        playbook = seed_playbook.copy()
        best_playbook = playbook.copy()
        best_val_score = best_val_score_init
        history = existing_history or []

        # Single-pass: shuffle tasks once, auto-calculate iterations
        if config.single_pass:
            self._single_pass_order = list(train_task_ids)
            random.shuffle(self._single_pass_order)
            self._single_pass_idx = 0
            total_iterations = math.ceil(len(train_task_ids) / max(config.batch_size, 1))
            if config.iterations != total_iterations:
                print(f"Single-pass mode: overriding iterations {config.iterations} -> {total_iterations} "
                      f"({len(train_task_ids)} tasks / bsz {config.batch_size})")
                config.iterations = total_iterations

        if start_iteration > 0:
            self._load_resume_artifacts(output_dir, start_iteration)

        for iteration in range(start_iteration, config.iterations):
            iter_start = time.time()
            timings: Dict[str, float] = {}
            iter_num = iteration + 1
            iter_dir = self._iter_subdir(iter_num)

            self._print_iter_header(iter_num, config.iterations, playbook)

            # Phase 1: Execute batch
            t0 = time.time()
            all_traces = self._execute_batch(playbook, train_task_ids, iteration, iter_dir)
            train_eval = self.evaluator.evaluate(all_traces)
            timings["training_s"] = round(time.time() - t0, 1)
            print(f"\nTraining: Pass%={train_eval.score*100:.1f}%, TGC={train_eval.tgc*100:.1f}% ({timings['training_s']}s)")

            # Phase 2: Validation
            val_eval = None
            run_val = (iter_num % config.val_interval == 0)
            if val_task_ids and not config.skip_validation and run_val:
                t0 = time.time()
                print(f"\nEvaluating on {len(val_task_ids)} validation tasks...")
                val_traces = self.system_adapter.execute(
                    val_task_ids, playbook, f"val_iter{iteration}",
                    max_workers=config.max_workers,
                )
                val_eval = self.evaluator.evaluate(val_traces)
                timings["validation_s"] = round(time.time() - t0, 1)
                print(f"Validation: Pass%={val_eval.score*100:.1f}%, TGC={val_eval.tgc*100:.1f}% ({timings['validation_s']}s)")

                val_score = val_eval.score * 0.7 + val_eval.tgc * 0.3
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_playbook = playbook.copy()
                    print(f"  *** New best! (val_score={val_score:.3f}) ***")
                    best_playbook.save(f"{output_dir}/rcl_best.json")
            else:
                timings["validation_s"] = 0.0

            # Phase 3: Select signal traces
            signal_traces = self._select_signal(all_traces, train_eval)
            signal_traces = self._ensure_signal_budget(all_traces, signal_traces, train_eval)
            self._replay.mark_reflected([t.task_id for t in signal_traces])
            self._print_signal_debug_summary(signal_traces)

            if not signal_traces:
                print("\nAll training tasks above threshold! Skipping reflect/mutate.")
                timings.update({"reflection_s": 0.0, "mutation_s": 0.0})
            else:
                # Phase 4: Reflect
                t0 = time.time()
                print(f"\nReflector analyzing {len(signal_traces)} tasks...")
                reflection = self._reflect(signal_traces, playbook, train_eval, iter_dir)
                timings["reflection_s"] = round(time.time() - t0, 1)
                print(f"  Reflection done ({timings['reflection_s']}s)")

                # Phase 4b: Optional trace summarization
                if self.trace_summarizer is not None:
                    self.trace_summarizer.summarize(signal_traces, reflection)

                # Phase 5b: Update entry counts from reflector assessments
                assessments = reflection.get("all_entry_assessments", [])
                recently_assessed = set()
                if assessments:
                    recently_assessed = playbook.update_counts(assessments, iteration=iter_num)
                    n_helpful = sum(1 for a in assessments if a.get("tag") == "helpful")
                    n_harmful = sum(1 for a in assessments if a.get("tag") == "harmful")
                    print(f"  Entry assessments: {n_helpful} helpful, {n_harmful} harmful ({len(recently_assessed)} entries)")

                # Phase 5c: Auto-prune entries with high harmful counts
                pruned = []
                if config.prune_threshold > 0:
                    pruned = playbook.prune(config.prune_threshold)
                    if pruned:
                        print(f"  Auto-pruned {len(pruned)} entries (threshold={config.prune_threshold}):")
                        for p in pruned:
                            print(f"    [{p.entry_id}] (+{p.helpful_count} -{p.harmful_count}) {p.content[:60]}...")

                # Phase 6: Mutate
                self._await_opt_state_update()
                t0 = time.time()
                print(f"\nCurator proposing mutations...")
                mutations, raw_response, mutation_summary = self._mutate(reflection, playbook, history, iter_dir, recently_assessed, signal_traces)
                timings["mutation_s"] = round(time.time() - t0, 1)
                print(f"  Proposed {len(mutations)} mutations ({timings['mutation_s']}s)")

                # Phase 7: Apply
                applied, rejected = [], []
                if mutations:
                    playbook, applied, rejected = self.apply_mutations(playbook, mutations, iteration=iter_num)

                self._write_iteration_artifacts(
                    iter_num, iter_dir, reflection, mutations, applied, rejected,
                    raw_response, playbook, signal_traces, assessments,
                    pruned if config.prune_threshold > 0 else [],
                )

                # Phase 8: Launch optimization state update in background
                if self._optimization_state_mgr:
                    _opt_state_kwargs = dict(
                        iteration=iter_num,
                        playbook=playbook.copy(),
                        train_eval=train_eval,
                        reflection=reflection,
                        applied_mutations=list(applied),
                        mutation_summary=mutation_summary,
                        recent_history=list(history),
                        sampling_stats=self._replay.get_stats(),
                    )
                    self._opt_state_launch_time = time.time()
                    self._opt_state_iter_dir = self._iter_subdir(iter_num)
                    print(f"\nLaunching optimization state update in background...")
                    self._opt_state_future = self._opt_state_executor.submit(
                        self._optimization_state_mgr.update, **_opt_state_kwargs,
                    )

            # Write playbook snapshot
            self._write_playbook_snapshot(iter_num, iter_dir, playbook)

            timings["iteration_total_s"] = round(time.time() - iter_start, 1)
            self._print_iter_timing(iter_num, timings, config.skip_validation)

            # Phase 9: Record
            self._record_iteration(
                iter_num, train_eval, val_eval, all_traces, signal_traces,
                history, timings, output_dir, len(playbook), len(train_task_ids),
            )

            # Checkpoint
            if iter_num % config.checkpoint_interval == 0:
                playbook.save(f"{output_dir}/rcl_iter{iter_num}.json")

        # Finalize
        self._await_opt_state_update()

        if config.skip_validation or not val_task_ids:
            best_playbook = playbook.copy()

        best_playbook.save(f"{output_dir}/rcl_best.json")
        playbook.save(f"{output_dir}/rcl_final.json")

        print(f"\n{'='*60}")
        print(f"Optimization complete!")
        print(f"Best playbook: {output_dir}/rcl_best.json")
        print(f"Final playbook: {output_dir}/rcl_final.json ({len(playbook)} entries)")
        if val_task_ids and not config.skip_validation:
            print(f"Best val_score: {best_val_score:.3f}")
        print(f"{'='*60}")

        return best_playbook

    def _execute_batch(
        self, playbook: Playbook, train_task_ids: List[str],
        iteration: int, iter_dir: str,
    ) -> List[ExecutionTrace]:
        """Execute a batch of training tasks with composable features."""
        self._current_playbook = playbook

        # Phase 1: Sample batch
        batch = self._sample_training_batch(train_task_ids)

        # Phase 2: Pre-execute transform (PP)
        exec_playbook = playbook
        if self._perturbation_names:
            from ..components.perturber import make_perturbed_playbook
            exec_playbook = make_perturbed_playbook(playbook, self._perturbation_names)

        # Phase 3: Expand for group rollouts
        group_size = self.config.group_size
        group_dual_trace = group_size > 1 and self.config.dual_trace
        if group_dual_trace:
            normal_rollouts = group_size - 1
            exec_batch = batch * normal_rollouts
        else:
            exec_batch = batch * group_size if group_size > 1 else batch

        # Print batch info
        stats = self._replay.get_stats()
        parts = [f"\nExecuting {len(batch)} tasks"]
        if group_size > 1:
            parts.append(f" x {group_size} rollouts = {len(batch) * group_size} total")
            if group_dual_trace:
                parts.append(f" ({normal_rollouts} normal + 1 PP, concurrent)")
        parts.append("...")
        info_parts = []
        if stats.get("replay_count", 0) > 0 or stats.get("replay_buffer_size_before_update", 0) > 0:
            info_parts.append(f"replay={stats.get('replay_count', 0)}, fresh={stats.get('fresh_count', 0)}")
        if stats.get("train_pool_size"):
            info_parts.append(
                f"coverage={stats.get('seen_task_count_after', 0)}/{stats.get('train_pool_size')}"
            )
        if self._perturbation_names:
            info_parts.append(f"perturbations={self.config.perturbation_set}")
        if info_parts:
            parts.append(f" ({', '.join(info_parts)})")
        print("".join(parts))

        # Phase 4: Execute
        if group_dual_trace:
            traces = self._execute_group_dual_trace(batch, exec_batch, playbook, exec_playbook, iteration, iter_dir)
        elif self.config.dual_trace:
            traces = self._execute_dual_trace(batch, exec_batch, playbook, exec_playbook, iteration, iter_dir)
        else:
            traces = self.system_adapter.execute(
                exec_batch, exec_playbook, f"iter{iteration}",
                max_workers=self.config.max_workers,
                trace_subdir=f"{iter_dir}/traces",
            )

        # Phase 5: Update replay
        self._replay.update_from_traces(traces)

        return traces

    def _execute_group_dual_trace(
        self, batch, exec_batch, playbook, exec_playbook, iteration, iter_dir,
    ) -> List[ExecutionTrace]:
        from ..components.perturber import PERTURBATION_SETS, make_perturbed_playbook
        pp_playbook = make_perturbed_playbook(playbook, PERTURBATION_SETS["full"])
        pp_prefix = "__pp__"
        pp_task_ids = [f"{pp_prefix}{tid}" for tid in batch]
        pp_overrides = {tid: pp_playbook for tid in pp_task_ids}
        combined_batch = exec_batch + pp_task_ids
        traces = self.system_adapter.execute(
            combined_batch, exec_playbook, f"iter{iteration}",
            max_workers=self.config.max_workers,
            trace_subdir=f"{iter_dir}/traces",
            playbook_overrides=pp_overrides,
        )
        for t in traces:
            if t.task_id.startswith(pp_prefix):
                t.task_id = t.task_id[len(pp_prefix):]
                t.metadata["is_pp_rollout"] = True
        return traces

    def _execute_dual_trace(
        self, batch, exec_batch, playbook, exec_playbook, iteration, iter_dir,
    ) -> List[ExecutionTrace]:
        from ..components.perturber import PERTURBATION_SETS, make_perturbed_playbook
        pp_playbook = make_perturbed_playbook(playbook, PERTURBATION_SETS["full"])
        pp_prefix = "__pp__"
        pp_task_ids = [f"{pp_prefix}{tid}" for tid in batch]
        pp_overrides = {tid: pp_playbook for tid in pp_task_ids}
        combined_batch = exec_batch + pp_task_ids
        print(f"  Dual-trace: executing {len(batch)} normal + {len(batch)} PP concurrently...")
        all_traces = self.system_adapter.execute(
            combined_batch, exec_playbook, f"iter{iteration}",
            max_workers=self.config.max_workers,
            trace_subdir=f"{iter_dir}/traces",
            playbook_overrides=pp_overrides,
        )
        traces = []
        pp_by_task: Dict[str, ExecutionTrace] = {}
        for t in all_traces:
            if t.task_id.startswith(pp_prefix):
                t.task_id = t.task_id[len(pp_prefix):]
                pp_by_task[t.task_id] = t
            else:
                traces.append(t)
        for t in traces:
            pp_t = pp_by_task.get(t.task_id)
            if pp_t is not None:
                t.metadata["pp_trace"] = pp_t.get_afc_trace_str()
                t.metadata["pp_pass_pct"] = pp_t.metadata.get("pass_pct", 0.0)
        return traces

    def _default_signal_pool(
        self,
        traces: List[ExecutionTrace],
        eval_result: Optional[EvaluationResult] = None,
    ) -> List[ExecutionTrace]:
        """Return eligible reflection pool (failures, no infra errors)."""
        threshold = self.config.reflection_threshold
        scores: Optional[List[float]] = None
        if (
            eval_result is not None
            and isinstance(getattr(eval_result, "per_instance_scores", None), list)
            and len(eval_result.per_instance_scores) == len(traces)
        ):
            scores = list(eval_result.per_instance_scores)

        pool: List[ExecutionTrace] = []
        for idx, trace in enumerate(traces):
            metadata_score = trace.metadata.get("pass_pct", 0.0)
            eval_score = scores[idx] if scores is not None else None
            if eval_score is None:
                score_below = metadata_score < threshold
            else:
                score_below = (eval_score < threshold) or (metadata_score < threshold)
            if (
                score_below
                and not trace.metadata.get("infra_error", False)
                and not trace.metadata.get("timed_out", False)
                and not (trace.metadata.get("error") and trace.metadata.get("n_tool_calls", 0) == 0)
            ):
                pool.append(trace)
        return pool

    def _signal_priority_key(self, trace: ExecutionTrace) -> Tuple[int, int, float, float]:
        """Prefer fresh-task failures over replay, then under-reflected/under-seen tasks."""
        task_id = trace.task_id
        return (
            int(task_id in self._replay.current_replay_ids),
            int(self._replay.task_reflection_count.get(task_id, 0)),
            int(self._replay.task_seen_count.get(task_id, 0)),
            float(trace.metadata.get("pass_pct", 0.0)),
            random.random(),
        )

    def _select_prioritized_signal_traces(
        self, eligible: List[ExecutionTrace], n: int,
    ) -> List[ExecutionTrace]:
        if n <= 0 or not eligible:
            return []
        ranked = sorted(eligible, key=self._signal_priority_key)
        return ranked[:min(n, len(ranked))]

    def _ensure_signal_budget(
        self,
        traces: List[ExecutionTrace],
        signal_traces: List[ExecutionTrace],
        eval_result: Optional[EvaluationResult] = None,
    ) -> List[ExecutionTrace]:
        if self.config.reflect_all_traces or self.config.group_size > 1:
            return signal_traces

        eligible = self._default_signal_pool(traces, eval_result)
        if not eligible:
            return signal_traces

        target = min(self.config.mini_batch, len(eligible))
        if len(signal_traces) >= target:
            return signal_traces

        repaired = self._select_prioritized_signal_traces(eligible, target)
        print(
            f"  Warning: signal selection returned fewer traces than expected; "
            f"expanding {len(signal_traces)} -> {len(repaired)} "
            f"(eligible={len(eligible)}, reflect_n={self.config.mini_batch})"
        )
        return repaired

    def _select_signal(
        self, traces: List[ExecutionTrace], eval_result: EvaluationResult,
    ) -> List[ExecutionTrace]:
        """Select signal traces for reflection."""
        if self.config.reflect_all_traces:
            valid = [
                t for t in traces
                if not t.metadata.get("infra_error", False)
                and not t.metadata.get("timed_out", False)
                and not (t.metadata.get("error") and t.metadata.get("n_tool_calls", 0) == 0)
            ]
            if not valid:
                return []
            return self._select_prioritized_signal_traces(valid, self.config.mini_batch)

        if self.config.group_size > 1:
            traces = self._merge_group_rollouts(traces)
            return self._select_signal_group(traces)

        failed = self._default_signal_pool(traces, eval_result)
        if not failed:
            return []
        n = self.config.mini_batch
        if n > 1:
            print(f"  Signal pool: eligible={len(failed)} reflect_n={n} selected={min(n, len(failed))}")
        return self._select_prioritized_signal_traces(failed, n)

    def _reflect(
        self, signal_traces: List[ExecutionTrace], playbook: Playbook,
        train_eval: EvaluationResult, iter_dir: str,
    ) -> Dict[str, Any]:
        return self.reflector.reflect(signal_traces, playbook, train_eval)

    def _print_signal_debug_summary(self, signal_traces: List[ExecutionTrace]) -> None:
        if not signal_traces:
            return
        replay = self._replay
        preview = []
        for trace in signal_traces[:5]:
            tid = trace.task_id
            preview.append(
                f"{tid}(pass={trace.metadata.get('pass_pct', 0.0)*100:.1f}%"
                f",seen={replay.task_seen_count.get(tid, 0)}"
                f",refl={replay.task_reflection_count.get(tid, 0)})"
            )
        suffix = " ..." if len(signal_traces) > 5 else ""
        print(f"  Signal traces: {', '.join(preview)}{suffix}")

    def _await_opt_state_update(self) -> None:
        """Block until the background optimization state update completes."""
        if self._opt_state_future is None:
            return
        already_done = self._opt_state_future.done()
        try:
            self._opt_state_future.result()
            elapsed = round(time.time() - self._opt_state_launch_time, 1)
            health = self._optimization_state_mgr.state.get('playbook_assessment', {}).get('health', '?')
            if already_done:
                print(f"  Background opt-state was already done ({elapsed}s total, health: {health})")
            else:
                print(f"  Waited for opt-state update ({elapsed}s total, health: {health})")
            if self.trace_writer and self._opt_state_iter_dir:
                self.trace_writer.write_json(
                    "optimization_state.json",
                    self._optimization_state_mgr.to_dict(),
                    subdir=self._opt_state_iter_dir,
                )
        except Exception as exc:
            print(f"  Warning: optimization state update failed: {exc}")
        finally:
            self._opt_state_future = None
            self._opt_state_iter_dir = None

    def _mutate(
        self, reflection: Dict[str, Any], playbook: Playbook,
        history: List[Dict], iter_dir: str,
        recently_assessed: Optional[set] = None,
        signal_traces: Optional[List[ExecutionTrace]] = None,
    ) -> Tuple[List[Dict], str, str]:
        kwargs: Dict[str, Any] = dict(
            recently_assessed=recently_assessed,
            signal_traces=signal_traces,
        )
        if self._optimization_state_mgr:
            kwargs["optimization_state_context"] = self._optimization_state_mgr.get_shared_context()
        try:
            return self.mutator.mutate(playbook, reflection, **kwargs)
        except Exception as exc:
            if is_content_filtered_llm_error(exc):
                message = f"Mutation skipped due to content filter block: {exc}"
                print(f"    Warning: {message}")
                return [], message, "blocked_by_content_filter_noop"
            raise

    def _record_iteration(
        self, iter_num: int, train_eval: EvaluationResult,
        val_eval: Optional[EvaluationResult],
        all_traces: List[ExecutionTrace], signal_traces: List[ExecutionTrace],
        history: List[Dict], timings: Dict, output_dir: str,
        playbook_size: int, train_pool_size: int,
    ) -> None:
        seen_count = len(self._replay.seen_task_ids)
        history_entry = {
            "iteration": iter_num,
            "train_pass": train_eval.score,
            "train_tgc": train_eval.tgc,
            "val_pass": val_eval.score if val_eval else 0.0,
            "val_tgc": val_eval.tgc if val_eval else 0.0,
            "playbook_size": playbook_size,
            "playbook_entries": playbook_size,
            "timings": timings,
            "time_total": timings.get("iteration_total_s", 0.0),
            "time_training": timings.get("training_s", 0.0),
            "time_reflection": timings.get("reflection_s", 0.0),
            "time_mutation": timings.get("mutation_s", 0.0),
            "time_validation": timings.get("validation_s", 0.0),
            "rollouts_total": len(all_traces),
            "rollouts_signal": len(signal_traces),
            "seen_task_count": seen_count,
            "coverage_frac": round(seen_count / max(1, train_pool_size), 4),
            "total_reflection_count": sum(self._replay.task_reflection_count.values()),
            "sampling": self._replay.get_stats(),
        }
        history.append(history_entry)
        with open(f"{output_dir}/history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Shared utilities ─────────────────────────────────────────

    def apply_mutations(
        self, playbook: Playbook, mutations: List[Dict],
        iteration: Optional[int] = None,
    ) -> Tuple[Playbook, List[Dict], List[Dict]]:
        new_playbook = playbook.copy()
        applied = []
        rejected = []
        entry_char_cap = max(0, int(getattr(self.config, "entry_char_cap", 0) or 0))

        for mutation in mutations:
            op = mutation.get("op", "").upper()

            if op == "ADD":
                content = mutation.get("content", "")
                section = mutation.get("section", "others")
                if content and len(content.strip()) > 10:
                    if entry_char_cap and len(content) > entry_char_cap:
                        rejected.append({**mutation, "reason": f"content exceeds entry_char_cap={entry_char_cap}"})
                        print(f"    Rejected ADD in {section}: content length {len(content)} > cap {entry_char_cap}")
                        continue
                    result = new_playbook.add_entry(content, section, check_duplicate=True)
                    if result:
                        print(f"    Added entry to {section}")
                        applied.append({**mutation, "entry_id": result.entry_id})
                    else:
                        print(f"    Skipped duplicate entry")
                        rejected.append({**mutation, "reason": "duplicate"})
                else:
                    rejected.append({**mutation, "reason": "content too short"})

            elif op == "UPDATE":
                entry_id = mutation.get("entry_id", "")
                content = mutation.get("content", "")
                entry = new_playbook.get_entry(entry_id)
                if entry and content:
                    if entry_char_cap and len(content) > entry_char_cap:
                        rejected.append({**mutation, "reason": f"content exceeds entry_char_cap={entry_char_cap}"})
                        print(f"    Rejected UPDATE {entry_id}: content length {len(content)} > cap {entry_char_cap}")
                        continue
                    entry.content = content
                    print(f"    Updated entry {entry_id}")
                    applied.append(mutation)
                else:
                    rejected.append({**mutation, "reason": "entry not found"})

            elif op == "DELETE":
                entry_id = mutation.get("entry_id", "")
                entry = new_playbook.get_entry(entry_id)
                if entry:
                    new_playbook.remove_entry(entry_id)
                    print(f"    Deleted entry {entry_id}")
                    applied.append(mutation)
                else:
                    rejected.append({**mutation, "reason": "entry not found"})

        return new_playbook, applied, rejected

    def _iter_subdir(self, iteration: int) -> str:
        return f"iterations/iter_{iteration}"

    def _sample_training_batch(self, train_task_ids: List[str]) -> List[str]:
        # Single-pass: take next sequential chunk
        if self.config.single_pass:
            start = self._single_pass_idx
            end = min(start + self.config.batch_size, len(self._single_pass_order))
            batch = self._single_pass_order[start:end]
            self._single_pass_idx = end
            self._replay.seen_task_ids.update(batch)
            for tid in batch:
                self._replay.task_seen_count[tid] = self._replay.task_seen_count.get(tid, 0) + 1
            return batch

        return self._replay.sample_batch(train_task_ids, self.config.batch_size)

    @staticmethod
    def discover_latest_iteration(output_dir: str) -> Optional[int]:
        iters_dir = Path(output_dir) / "iterations"
        if not iters_dir.exists():
            return None
        max_iter = None
        for p in iters_dir.iterdir():
            if not p.is_dir() or not p.name.startswith("iter_"):
                continue
            if not (p / "playbook.json").exists():
                continue
            try:
                n = int(p.name.split("_")[1])
                if max_iter is None or n > max_iter:
                    max_iter = n
            except (ValueError, IndexError):
                continue
        return max_iter

    def _load_resume_artifacts(self, output_dir: str, start_iteration: int) -> None:
        iter_dir = Path(output_dir) / self._iter_subdir(start_iteration)

        # Load optimization state
        opt_state_path = iter_dir / "optimization_state.json"
        if self._optimization_state_mgr:
            if opt_state_path.exists():
                with open(opt_state_path) as f:
                    self._optimization_state_mgr.load_dict(json.load(f))
                print(f"Loaded optimization state: health={self._optimization_state_mgr.state.get('playbook_assessment', {}).get('health', '?')}")
            else:
                self._recompute_opt_state_on_resume(iter_dir, start_iteration)

        # Load sampler state
        sampler_state_path = iter_dir / "sampler_state.json"
        if sampler_state_path.exists():
            with open(sampler_state_path) as f:
                sampler_state = json.load(f)
            self._replay.restore(sampler_state)
            print(
                f"Loaded sampler state: "
                f"{self._replay.entry_count} replay tasks, "
                f"{len(self._replay.seen_task_ids)} seen tasks"
            )
        elif self._replay.enabled or self.config.single_pass:
            if self._rebuild_sampler_from_artifacts(output_dir, start_iteration):
                print(
                    f"Rebuilt sampler state from iteration artifacts: "
                    f"{self._replay.entry_count} replay tasks, "
                    f"{len(self._replay.seen_task_ids)} seen tasks"
                )
            else:
                print("Warning: sampler_state.json not found and artifact reconstruction failed; sampler restarts fresh.")

    def _recompute_opt_state_on_resume(self, iter_dir: Path, iteration: int) -> None:
        playbook_path = iter_dir / "playbook.json"
        mutations_path = iter_dir / "mutations.json"
        if not playbook_path.exists():
            print(f"  Warning: optimization_state.json missing and no playbook.json to recompute from")
            return

        from .data_structures import Playbook as _Playbook
        with open(playbook_path) as f:
            playbook = _Playbook.from_dict(json.load(f))

        applied = []
        reflection = {}
        mutation_summary = ""
        if mutations_path.exists():
            with open(mutations_path) as f:
                mutations_data = json.load(f)
            applied = mutations_data.get("applied", [])
            mutation_summary = mutations_data.get("raw_mutator_response", "")[:500]
            reflection = {"combined_analysis": mutations_data.get("combined_analysis", "")}

        _opt_state_kwargs = dict(
            iteration=iteration,
            playbook=playbook,
            train_eval=None,
            reflection=reflection,
            applied_mutations=applied,
            mutation_summary=mutation_summary,
            recent_history=[],
            sampling_stats={},
        )
        self._opt_state_launch_time = time.time()
        self._opt_state_iter_dir = self._iter_subdir(iteration)
        print(f"  optimization_state.json missing — recomputing in background...")
        self._opt_state_future = self._opt_state_executor.submit(
            self._optimization_state_mgr.update, **_opt_state_kwargs,
        )

    def _rebuild_sampler_from_artifacts(self, output_dir: str, start_iteration: int) -> bool:
        if start_iteration <= 0:
            return False

        self._replay.reset()
        history_by_iter = self._load_history_sampling_map(output_dir)

        for iter_num in range(1, start_iteration + 1):
            iter_dir = Path(output_dir) / self._iter_subdir(iter_num)
            traces_root = iter_dir / "traces"

            # Handle frontier-style (slot*) or flat traces
            if any(iter_dir.glob("slot*")):
                frontier_scores: Dict[str, List[float]] = defaultdict(list)
                for slot_dir in sorted(traces_root.glob("slot*")):
                    for tid, score in self._load_trace_scores(slot_dir).items():
                        frontier_scores[tid].append(score)
                task_scores = {tid: min(scores) for tid, scores in frontier_scores.items() if scores}
            else:
                task_scores = self._load_trace_scores(traces_root)

            if not task_scores:
                return False

            self._replay.seen_task_ids.update(task_scores.keys())
            for tid in task_scores:
                self._replay.task_seen_count[tid] = self._replay.task_seen_count.get(tid, 0) + 1
            self._replay.update_from_scores(task_scores)

            reflected_ids = self._load_reflected_task_ids(iter_dir / "reflections")
            self._replay.mark_reflected(reflected_ids)

        return True

    @staticmethod
    def _load_history_sampling_map(output_dir: str) -> Dict[int, Dict[str, Any]]:
        history_path = Path(output_dir) / "history.json"
        if not history_path.exists():
            return {}
        with open(history_path) as f:
            history = json.load(f)
        sampling_by_iter: Dict[int, Dict[str, Any]] = {}
        if not isinstance(history, list):
            return sampling_by_iter
        for entry in history:
            iter_num = int(entry.get("iteration", 0))
            sampling = entry.get("sampling")
            if iter_num > 0 and isinstance(sampling, dict):
                sampling_by_iter[iter_num] = sampling
        return sampling_by_iter

    @staticmethod
    def _load_trace_scores(traces_dir: Path) -> Dict[str, float]:
        if not traces_dir.exists():
            return {}
        task_scores: Dict[str, float] = {}
        for trace_path in traces_dir.glob("*.json"):
            with open(trace_path) as f:
                payload = json.load(f)
            task_id = payload.get("task_id") or trace_path.stem
            result = payload.get("result", {})
            pass_pct = result.get("pass_pct", payload.get("pass_pct", 0.0))
            task_scores[task_id] = max(task_scores.get(task_id, 0.0), float(pass_pct))
        return task_scores

    @staticmethod
    def _load_reflected_task_ids(reflections_dir: Path) -> List[str]:
        if not reflections_dir.exists():
            return []
        return [p.stem for p in reflections_dir.glob("*.json")]

    # ── Group helpers ────────────────────────────────────────────

    def _merge_group_rollouts(self, traces: List[ExecutionTrace]) -> List[ExecutionTrace]:
        groups: Dict[str, List[ExecutionTrace]] = defaultdict(list)
        for t in traces:
            groups[t.task_id].append(t)
        merged = []
        for task_id, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_single_group(task_id, group))
        return merged

    @staticmethod
    def _merge_single_group(task_id: str, group: List[ExecutionTrace]) -> ExecutionTrace:
        pp_rollout = None
        normal_rollouts = []
        for t in group:
            if t.metadata.get("is_pp_rollout"):
                pp_rollout = t
            else:
                normal_rollouts.append(t)

        group_sorted = sorted(normal_rollouts, key=lambda t: t.metadata.get("pass_pct", 0.0))
        k = len(group_sorted)
        pcts = [t.metadata.get("pass_pct", 0.0) for t in group_sorted]
        rollout_ids = [f"rollout_{i}" for i in range(1, k + 1)]
        pct_summary = ", ".join(f"{rid}: {pct*100:.1f}%" for rid, pct in zip(rollout_ids, pcts))

        blurb = (
            f"The following are independent rollouts of the same task "
            f"using the same playbook. Since all rollouts used identical instructions, "
            f"differences in outcomes reflect variance in the agent's execution choices. "
            f"Each rollout has a corresponding evaluation in the Evaluation Feedback "
            f"section (matched by rollout ID). Ground your analysis in the differences "
            f"between evaluation scores and details: {pct_summary}."
        )

        trace_parts = [blurb]
        eval_parts = []
        for rid, t in zip(rollout_ids, group_sorted):
            pct = t.metadata.get("pass_pct", 0.0)
            trace_parts.append(f"\n## {rid} — pass_pct: {pct*100:.1f}%\n\n{t.get_afc_trace_str()}")
            eval_detail = t.metadata.get("evaluation_details", f"Pass percentage: {pct*100:.1f}%")
            eval_parts.append(f"## {rid} Evaluation\n\n{eval_detail}")

        metadata = {
            "pass_pct": min(pcts),
            "task_completed": any(t.metadata.get("task_completed", False) for t in group_sorted),
            "evaluation_details": "\n\n".join(eval_parts),
            "group_size": k,
            "group_pass_pcts": pcts,
        }
        if pp_rollout is not None:
            metadata["pp_trace"] = pp_rollout.get_afc_trace_str()
            metadata["pp_pass_pct"] = pp_rollout.metadata.get("pass_pct", 0.0)

        return ExecutionTrace(
            task_id=task_id,
            input_query=group_sorted[0].input_query,
            system_output=None,
            trace="\n".join(trace_parts),
            metadata=metadata,
        )

    def _select_signal_group(self, merged_traces: List[ExecutionTrace]) -> List[ExecutionTrace]:
        contrastive = {}
        all_fail = {}
        for t in merged_traces:
            pcts = t.metadata.get("group_pass_pcts")
            if pcts is None:
                if t.metadata.get("pass_pct", 0.0) < self.config.reflection_threshold:
                    all_fail[t.task_id] = (t, 0.0)
                continue
            has_pass = any(p >= 1.0 for p in pcts)
            has_fail = any(p < 1.0 for p in pcts)
            variance = max(pcts) - min(pcts)
            if has_pass and has_fail:
                contrastive[t.task_id] = (t, variance)
            elif not has_pass:
                all_fail[t.task_id] = (t, variance)

        print(f"  Groups: {len(contrastive)} contrastive, {len(all_fail)} all-fail")
        pool = contrastive or all_fail
        if not pool:
            return []
        n = self.config.mini_batch
        sorted_pool = sorted(pool.items(), key=lambda x: x[1][1], reverse=True)
        return [t for _, (t, _) in sorted_pool[:n]]

    # ── Iteration artifacts / printing ───────────────────────────

    def _print_iter_header(self, iter_num: int, total: int, playbook: Playbook) -> None:
        print(f"\n{'='*60}")
        print(f"Iteration {iter_num}/{total}")
        print(f"Playbook size: {len(playbook)} entries")
        if self.config.group_size > 1:
            print(f"Group size: K={self.config.group_size}")
        print(f"{'='*60}")

    def _write_iteration_artifacts(
        self, iter_num: int, iter_dir: str,
        reflection: Dict, mutations: List, applied: List, rejected: List,
        raw_response: str, playbook: Playbook,
        signal_traces: Optional[List[ExecutionTrace]] = None,
        assessments: Optional[List[Dict]] = None,
        pruned: Optional[List] = None,
    ) -> None:
        if not self.trace_writer:
            return
        subdir = self._iter_subdir(iter_num)
        trace_map = {t.task_id: t for t in signal_traces} if signal_traces else {}

        for analysis in reflection.get("analyses", []):
            task_id = analysis["task_id"]
            trace = trace_map.get(task_id)
            payload = {
                "task_id": task_id,
                "pass_pct": analysis.get("pass_pct", 0.0),
                "task_completed": analysis.get("task_completed", False),
                "entry_assessments": analysis.get("entry_assessments", []),
                "analysis": analysis.get("analysis", ""),
                "afc_trace_truncated": trace.get_afc_trace_str() if trace else "",
                "evaluation_details": trace.metadata.get("evaluation_details", "") if trace else "",
            }
            if "best_pass_pct" in analysis:
                payload["best_pass_pct"] = analysis["best_pass_pct"]
            self.trace_writer.write_trace(task_id, payload, subdir=f"{subdir}/reflections")

        self.trace_writer.write_json("mutations.json", {
            "combined_analysis": reflection.get("combined_analysis", ""),
            "head_names": reflection.get("head_names", []),
            "merge_strategy": reflection.get("merge_strategy", ""),
            "proposed": mutations,
            "applied": applied,
            "rejected": rejected,
            "raw_mutator_response": raw_response,
        }, subdir=subdir)

        if assessments or pruned:
            self.trace_writer.write_json("assessments.json", {
                "entry_assessments": assessments or [],
                "pruned_entries": [
                    {
                        "entry_id": p.entry_id, "section": p.section,
                        "content": p.content,
                        "helpful_count": p.helpful_count,
                        "harmful_count": p.harmful_count,
                    }
                    for p in (pruned or [])
                ],
            }, subdir=subdir)

    def _write_playbook_snapshot(self, iter_num: int, iter_dir: str, playbook: Playbook) -> None:
        if not self.trace_writer:
            return
        self.trace_writer.write_json(
            "playbook.json", playbook.to_dict(), subdir=self._iter_subdir(iter_num),
        )
        self._write_resume_state_snapshot(iter_num)

    def _write_resume_state_snapshot(self, iter_num: int) -> None:
        if not self.trace_writer:
            return
        subdir = self._iter_subdir(iter_num)
        self.trace_writer.write_json(
            "sampler_state.json", self._replay.serialize(), subdir=subdir,
        )
        if self._optimization_state_mgr and self._opt_state_future is None:
            self.trace_writer.write_json(
                "optimization_state.json",
                self._optimization_state_mgr.to_dict(),
                subdir=subdir,
            )

    def _print_iter_timing(self, iter_num: int, timings: Dict, skip_val: bool) -> None:
        print(f"\n--- Iteration {iter_num} Timing ---")
        print(f"  Training:   {timings.get('training_s', 0):.1f}s")
        if not skip_val:
            print(f"  Validation: {timings.get('validation_s', 0):.1f}s")
        else:
            print(f"  Validation: skipped")
        print(f"  Reflection: {timings.get('reflection_s', 0):.1f}s")
        print(f"  Mutation:   {timings.get('mutation_s', 0):.1f}s")
        print(f"  TOTAL:      {timings.get('iteration_total_s', 0):.1f}s")

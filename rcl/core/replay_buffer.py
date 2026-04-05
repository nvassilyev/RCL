"""Failure replay buffer for RCL training.

Tracks tasks that have failed during training and replays them in future
batches. Tasks graduate (are removed) after consecutive passes, and are
evicted after consecutive post-reflection failures to avoid wasting budget
on stuck tasks.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple


@dataclass
class _ReplayEntry:
    consecutive_passes: int = 0
    consecutive_failures: int = 0
    reflected: bool = False


class ReplayBuffer:
    """Failure replay buffer with pass-graduation and stuck-eviction."""

    def __init__(
        self,
        replay_ratio: float = 0.0,
        max_size: int = 0,
        passes_to_graduate: int = 5,
        failures_to_evict: int = 3,
        unseen_first: bool = True,
    ):
        self.replay_ratio = min(max(replay_ratio, 0.0), 1.0)
        self.max_size = max_size
        self.passes_to_graduate = passes_to_graduate
        self.failures_to_evict = failures_to_evict
        self.unseen_first = unseen_first

        self._entries: Dict[str, _ReplayEntry] = {}
        self._seen_task_ids: Set[str] = set()
        self._task_seen_count: Dict[str, int] = {}
        self._task_reflection_count: Dict[str, int] = {}
        self._current_replay_ids: Set[str] = set()
        self._last_stats: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self.replay_ratio > 0

    @property
    def seen_task_ids(self) -> Set[str]:
        return self._seen_task_ids

    @property
    def task_seen_count(self) -> Dict[str, int]:
        return self._task_seen_count

    @property
    def task_reflection_count(self) -> Dict[str, int]:
        return self._task_reflection_count

    @property
    def current_replay_ids(self) -> Set[str]:
        return self._current_replay_ids

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._last_stats)

    def reset(self) -> None:
        self._entries.clear()
        self._seen_task_ids.clear()
        self._task_seen_count.clear()
        self._task_reflection_count.clear()
        self._current_replay_ids = set()
        self._last_stats = {}

    # ── Sampling ──────────────────────────────────────────────

    def sample_batch(
        self,
        train_task_ids: List[str],
        batch_size: int,
    ) -> List[str]:
        """Sample a training batch mixing replay and fresh tasks."""
        self._current_replay_ids = set()
        unique_train = len(set(train_task_ids))
        seen_before = len(self._seen_task_ids)
        target = min(batch_size, len(train_task_ids))

        if target <= 0:
            self._last_stats = self._build_stats(0, 0, 0, 0, unique_train, seen_before)
            return []

        replay_ids = list(self._entries.keys())
        desired_replay = min(len(replay_ids), int(round(target * self.replay_ratio))) if self.replay_ratio > 0 else 0

        selected: List[str] = []

        # Step 1: Sample from replay buffer
        if desired_replay > 0:
            replay_selected = self._select_underseen(replay_ids, desired_replay)
            selected.extend(replay_selected)
            self._current_replay_ids = set(replay_selected)

        # Step 2: Fill with fresh tasks
        selected_set = set(selected)
        fresh_pool = [tid for tid in train_task_ids if tid not in selected_set and tid not in self._seen_task_ids]
        unseen_before = len(fresh_pool)

        if not fresh_pool or not self.unseen_first:
            fresh_pool = [tid for tid in train_task_ids if tid not in selected_set]

        fresh_needed = target - len(selected)
        if fresh_needed > 0 and fresh_pool:
            selected.extend(self._select_underseen(fresh_pool, fresh_needed))

        # Step 3: Backfill if still short
        if len(selected) < target:
            selected_set = set(selected)
            remaining = [tid for tid in train_task_ids if tid not in selected_set]
            if remaining:
                selected.extend(self._select_underseen(remaining, target - len(selected)))

        random.shuffle(selected)
        self._seen_task_ids.update(selected)
        for tid in selected:
            self._task_seen_count[tid] = self._task_seen_count.get(tid, 0) + 1

        self._last_stats = self._build_stats(
            len(selected), len(self._current_replay_ids),
            max(0, len(selected) - len(self._current_replay_ids)),
            unseen_before, unique_train, seen_before,
        )
        return selected

    # ── Update from execution results ─────────────────────────

    def update_from_scores(self, task_scores: Dict[str, float]) -> None:
        """Update replay buffer after evaluating a batch."""
        failed = {tid for tid, score in task_scores.items() if score < 1.0}
        passed = set(task_scores.keys()) - failed

        # Add new failures
        for tid in failed:
            entry = self._entries.get(tid)
            if entry is not None:
                entry.consecutive_passes = 0
                entry.consecutive_failures += 1
            else:
                self._add(tid)

        # Graduate passed tasks
        graduated = []
        if self.passes_to_graduate > 0:
            for tid in passed:
                entry = self._entries.get(tid)
                if entry is None:
                    continue
                entry.consecutive_passes += 1
                entry.consecutive_failures = 0
                if entry.consecutive_passes >= self.passes_to_graduate:
                    graduated.append(tid)
            for tid in graduated:
                del self._entries[tid]

        # Evict stuck tasks (consecutive failures after reflection)
        evicted = []
        if self.failures_to_evict > 0:
            for tid in failed:
                entry = self._entries.get(tid)
                if entry is None or not entry.reflected:
                    continue
                if entry.consecutive_failures >= self.failures_to_evict:
                    evicted.append(tid)
            for tid in evicted:
                del self._entries[tid]

        self._last_stats.update({
            "failed_task_count": len(failed),
            "passed_task_count": len(passed),
            "graduated_task_count": len(graduated),
            "evicted_stuck_count": len(evicted),
            "replay_buffer_size_after_update": len(self._entries),
            "seen_task_count": len(self._seen_task_ids),
        })

    def update_from_traces(self, traces) -> None:
        """Update from ExecutionTrace objects (convenience wrapper)."""
        task_scores: Dict[str, float] = {}
        for trace in traces:
            if trace.metadata.get("is_pp_rollout"):
                continue
            task_scores[trace.task_id] = max(
                task_scores.get(trace.task_id, 0.0),
                trace.metadata.get("pass_pct", 0.0),
            )
        self.update_from_scores(task_scores)

    def mark_reflected(self, task_ids: List[str]) -> None:
        """Mark tasks as having been reflected on."""
        for tid in task_ids:
            self._task_reflection_count[tid] = self._task_reflection_count.get(tid, 0) + 1
            entry = self._entries.get(tid)
            if entry is not None:
                entry.reflected = True

    # ── Serialization ─────────────────────────────────────────

    def serialize(self) -> Dict[str, Any]:
        return {
            "seen_task_ids": sorted(self._seen_task_ids),
            "task_seen_count": dict(self._task_seen_count),
            "task_reflection_count": dict(self._task_reflection_count),
            "entries": {
                tid: {
                    "consecutive_passes": e.consecutive_passes,
                    "consecutive_failures": e.consecutive_failures,
                    "reflected": e.reflected,
                }
                for tid, e in self._entries.items()
            },
            "last_stats": dict(self._last_stats),
        }

    def restore(self, state: Dict[str, Any]) -> None:
        self._seen_task_ids = set(state.get("seen_task_ids", []))
        self._task_seen_count = {k: int(v) for k, v in state.get("task_seen_count", {}).items()}
        self._task_reflection_count = {k: int(v) for k, v in state.get("task_reflection_count", {}).items()}
        for tid, entry_state in state.get("entries", {}).items():
            self._entries[tid] = _ReplayEntry(
                consecutive_passes=int(entry_state.get("consecutive_passes", 0)),
                consecutive_failures=int(entry_state.get("consecutive_failures", 0)),
                reflected=bool(entry_state.get("reflected", False)),
            )
        self._last_stats = state.get("last_stats", {})

    # ── Internals ─────────────────────────────────────────────

    def _add(self, task_id: str) -> None:
        if task_id in self._entries:
            return
        self._entries[task_id] = _ReplayEntry()
        if self.max_size > 0 and len(self._entries) > self.max_size:
            overflow = len(self._entries) - self.max_size
            keys = list(self._entries.keys())
            for k in keys[:overflow]:
                del self._entries[k]

    def _select_underseen(self, pool: List[str], n: int) -> List[str]:
        if n <= 0 or not pool:
            return []
        ranked = sorted(pool, key=lambda tid: (
            self._task_seen_count.get(tid, 0),
            self._task_reflection_count.get(tid, 0),
            random.random(),
        ))
        return ranked[:min(n, len(ranked))]

    def _build_stats(
        self, batch_size, replay_count, fresh_count,
        unseen_remaining, train_pool_size, seen_before,
    ) -> Dict[str, Any]:
        return {
            "batch_size": batch_size,
            "replay_count": replay_count,
            "fresh_count": fresh_count,
            "unseen_remaining_before_update": unseen_remaining,
            "replay_buffer_size_before_update": len(self._entries),
            "replay_buffer_size_after_update": len(self._entries),
            "train_pool_size": train_pool_size,
            "seen_task_count_before": seen_before,
            "seen_task_count_after": len(self._seen_task_ids),
        }

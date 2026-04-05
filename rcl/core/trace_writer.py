"""Simple trace and artifact writer for debugging and SFT data collection."""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def build_rollout_descriptors(task_ids: Sequence[str]) -> List[Dict[str, object]]:
    """Assign stable rollout metadata for an ordered task-id list."""

    totals = Counter(task_ids)
    seen = defaultdict(int)
    descriptors: List[Dict[str, object]] = []

    for task_id in task_ids:
        seen[task_id] += 1
        rollout_number = seen[task_id]
        rollout_count = totals[task_id]
        rollout_id = f"{task_id}__r{rollout_number}"
        artifact_id = rollout_id if rollout_count > 1 else task_id
        descriptors.append(
            {
                "task_id": task_id,
                "rollout_index": rollout_number - 1,
                "rollout_number": rollout_number,
                "rollout_count": rollout_count,
                "rollout_id": rollout_id,
                "artifact_id": artifact_id,
                "is_duplicate_rollout": rollout_count > 1,
            }
        )

    return descriptors


def rollout_metadata(rollout: Dict[str, object]) -> Dict[str, object]:
    """Return the rollout fields persisted on traces/artifacts."""

    return {
        "rollout_index": rollout["rollout_index"],
        "rollout_number": rollout["rollout_number"],
        "rollout_count": rollout["rollout_count"],
        "rollout_id": rollout["rollout_id"],
        "artifact_id": rollout["artifact_id"],
        "is_duplicate_rollout": rollout["is_duplicate_rollout"],
    }


class TraceWriter:
    """Writes execution traces and optimization artifacts to disk.

    Thread-safe by design: writes are serialized and duplicate task ids get
    unique artifact paths instead of overwriting prior rollouts.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._write_lock = threading.Lock()

    def _unique_trace_path(self, stem: str, subdir: str) -> Path:
        trace_dir = self.base_dir / subdir
        path = trace_dir / f"{stem}.json"
        if not path.exists():
            return path

        suffix = 2
        base_stem = stem
        if "__r" in stem:
            prefix, _, tail = stem.rpartition("__r")
            if tail.isdigit():
                base_stem = prefix
                suffix = int(tail) + 1

        while True:
            candidate = trace_dir / f"{base_stem}__r{suffix}.json"
            if not candidate.exists():
                return candidate
            suffix += 1

    def write_trace(
        self,
        task_id: str,
        data: dict,
        subdir: str = "traces",
        artifact_id: str | None = None,
    ):
        with self._write_lock:
            path = self._unique_trace_path(artifact_id or task_id, subdir)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def write_json(self, filename: str, data, subdir: str = ""):
        if subdir:
            path = self.base_dir / subdir / filename
        else:
            path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

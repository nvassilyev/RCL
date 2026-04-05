#!/usr/bin/env python3
"""Create deterministic RewardBench 2 train/val/test splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..adapters.rewardbench2_client import build_default_splits, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RewardBench 2 splits")
    parser.add_argument("--dataset", default="benchmarks/rewardbench2/data/rewardbench2.jsonl")
    parser.add_argument("--output-dir", default="benchmarks/rewardbench2/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    tasks = load_dataset(args.dataset)
    splits = build_default_splits(
        tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_map = {task.task_id: task for task in tasks}
    manifest = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "counts": {},
    }

    for split_name, task_ids in splits.items():
        with (output_dir / f"{split_name}.json").open("w", encoding="utf-8") as f:
            json.dump(task_ids, f, indent=2)

        subset_counts = defaultdict(int)
        for task_id in task_ids:
            subset_counts[task_map[task_id].subset] += 1
        manifest["counts"][split_name] = {
            "total": len(task_ids),
            "subsets": dict(sorted(subset_counts.items())),
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

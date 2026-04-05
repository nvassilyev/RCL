#!/usr/bin/env python3
"""Download RewardBench 2 into benchmarks/rewardbench2/data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RewardBench 2 locally")
    parser.add_argument("--dataset", default="allenai/reward-bench-2")
    parser.add_argument(
        "--output",
        default="benchmarks/rewardbench2/data/rewardbench2.jsonl",
        help="Local JSONL output path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="test")
    dataset.to_json(str(output_path), orient="records", lines=True, force_ascii=False)

    subset_counts = {}
    for row in dataset:
        subset_counts[row["subset"]] = subset_counts.get(row["subset"], 0) + 1

    print(json.dumps({"rows": len(dataset), "subset_counts": subset_counts}, indent=2, sort_keys=True))
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()

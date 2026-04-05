#!/usr/bin/env python3
"""Quick RewardBench 2 baseline run via RCL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rcl.core.data_structures import Playbook
from rcl.core.trace_writer import TraceWriter

from ..adapters.rewardbench2_client import RewardBench2Config
from ..adapters.system_adapter import RewardBench2SystemAdapter
from ..benchmark import REWARDBENCH2_SECTIONS
from ..evaluator import RewardBench2Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="RewardBench 2 baseline eval")
    parser.add_argument("--playbook", default=None, help="Path to playbook JSON")
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-concurrent", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        model_dir = args.model.replace("/", "_")
        args.output = f"results/evals/{model_dir}/rb2_baseline"

    random.seed(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.playbook:
        playbook = Playbook.load(args.playbook, allowed_sections=set(REWARDBENCH2_SECTIONS.keys()))
    else:
        playbook = Playbook(allowed_sections=set(REWARDBENCH2_SECTIONS.keys()))

    config = RewardBench2Config(model=args.model, n_concurrent=args.n_concurrent)
    trace_writer = TraceWriter(args.output)
    adapter = RewardBench2SystemAdapter(config=config, trace_writer=trace_writer)
    evaluator = RewardBench2Evaluator()

    task_ids = adapter.load_tasks(args.split)
    random.shuffle(task_ids)
    task_ids = task_ids[: args.limit]

    print(f"Running {len(task_ids)} RewardBench 2 tasks from {args.split}")
    traces = adapter.execute(task_ids, playbook, "baseline", trace_subdir="traces")
    result = evaluator.evaluate(traces)

    leaderboard = result.metadata.get("leaderboard_score", 0.0) * 100
    pairwise = result.score * 100
    prompt_acc = result.tgc * 100

    print(f"\n{'=' * 60}")
    print(f"Leaderboard score: {leaderboard:.2f}%")
    print(f"Average pairwise score: {pairwise:.2f}%")
    print(f"Prompt accuracy: {prompt_acc:.2f}%")
    print(f"Subset scores: {result.metadata.get('subset_scores', {})}")
    print(f"{'=' * 60}")

    with open(f"{args.output}/results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "leaderboard_score": result.metadata.get("leaderboard_score", 0.0),
                "avg_pairwise_score": result.score,
                "prompt_accuracy": result.tgc,
                "subset_scores": result.metadata.get("subset_scores", {}),
                "task_ids": task_ids,
            },
            f,
            indent=2,
        )

    if traces:
        trace = traces[0]
        print("\n--- TRACE ---")
        print(trace.trace[:2000])
        print("\n--- EVALUATION DETAILS ---")
        print(trace.metadata.get("evaluation_details", "(not set)"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a RewardBench 2 playbook."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from rcl.core.data_structures import Playbook
from rcl.core.trace_writer import TraceWriter

from ..adapters.rewardbench2_client import RewardBench2Config
from ..adapters.system_adapter import RewardBench2SystemAdapter
from ..benchmark import REWARDBENCH2_CONFIG
from ..evaluator import RewardBench2Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="RewardBench 2 evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--playbook", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking-level", default="none")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args()

    if args.output is None:
        model_dir = args.model.replace("/", "_")
        playbook_stem = Path(args.playbook).parent.name
        args.output = f"results/evals/{model_dir}/{playbook_stem}_{args.split}"

    random.seed(args.seed)
    playbook = Playbook.load(args.playbook, allowed_sections=REWARDBENCH2_CONFIG.section_names)
    evaluator = RewardBench2Evaluator()

    print(f"{'=' * 60}")
    print("RCL Eval: rewardbench2")
    print(f"  Model:      {args.model}")
    print(f"  Playbook:   {args.playbook}")
    print(f"  Split:      {args.split}")
    print(f"  Runs:       {args.runs}")
    print(f"  Concurrent: {args.n_concurrent}")
    print(f"  Output:     {args.output}")
    print(f"{'=' * 60}\n")

    leaderboard_scores = []
    for run in range(1, args.runs + 1):
        run_output = f"{args.output}_r{run}" if args.runs > 1 else args.output
        Path(run_output).mkdir(parents=True, exist_ok=True)
        trace_writer = TraceWriter(run_output)
        config = RewardBench2Config(
            model=args.model,
            n_concurrent=args.n_concurrent,
            max_output_tokens=args.max_output_tokens,
            thinking_level=None if args.thinking_level == "none" else args.thinking_level,
        )
        adapter = RewardBench2SystemAdapter(config=config, trace_writer=trace_writer)
        task_ids = adapter.load_tasks(args.split, limit=args.limit)

        print(f"--- Run {run}/{args.runs} ({len(task_ids)} tasks) ---")
        t0 = time.time()
        traces = adapter.execute(task_ids, playbook, f"eval_r{run}", trace_subdir="traces")
        elapsed = time.time() - t0
        result = evaluator.evaluate(traces)

        leaderboard = result.metadata.get("leaderboard_score", 0.0) * 100
        leaderboard_scores.append(leaderboard)

        payload = {
            "leaderboard_score": result.metadata.get("leaderboard_score", 0.0),
            "avg_pairwise_score": result.score,
            "prompt_accuracy": result.tgc,
            "subset_scores": result.metadata.get("subset_scores", {}),
            "subset_counts": result.metadata.get("subset_counts", {}),
            "elapsed_s": round(elapsed, 1),
            "model": args.model,
            "playbook": args.playbook,
            "playbook_entries": len(playbook),
            "split": args.split,
        }
        with open(f"{run_output}/results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"  Leaderboard: {leaderboard:.2f}%")
        print(f"  Pairwise:    {result.score * 100:.2f}%")
        print(f"  Prompt acc:  {result.tgc * 100:.2f}%")
        print(f"  Subsets:     {result.metadata.get('subset_scores', {})}")
        print(f"  Elapsed:     {elapsed:.1f}s\n")

    if args.runs > 1:
        mean = sum(leaderboard_scores) / len(leaderboard_scores)
        std = math.sqrt(sum((score - mean) ** 2 for score in leaderboard_scores) / len(leaderboard_scores))
        summary = {
            "leaderboard_runs": leaderboard_scores,
            "leaderboard_mean": round(mean, 3),
            "leaderboard_std": round(std, 3),
            "model": args.model,
            "playbook": args.playbook,
            "split": args.split,
        }
        with open(f"{args.output}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Mean +/- Std leaderboard: {mean:.2f} +/- {std:.2f}%")


if __name__ == "__main__":
    main()

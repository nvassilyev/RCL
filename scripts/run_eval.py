#!/usr/bin/env python3
"""Run evaluation for AppWorld, BrowseComp, or RewardBench2 with a trained playbook.

Usage:
    cd /path/to/RCL

    # BrowseComp eval (3 runs for mean+-std)
    python -m scripts.run_eval --benchmark browsecomp \
        --model gemini-3-flash-preview \
        --playbook results/training/bc_flash_add/rcl_iter30.json \
        --runs 3 --output results/evals/bc_flash_add_iter30

    # AppWorld eval on test_normal
    python -m scripts.run_eval --benchmark appworld \
        --model gemini-3-flash-preview \
        --playbook results/training/aw_flash_add/rcl_iter30.json \
        --split test_normal --output results/evals/aw_flash_add_iter30_tn
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

from benchmarks.appworld.appworld_root import validate_appworld_root
from rcl.core.data_structures import Playbook
from rcl.core.trace_writer import TraceWriter


def run_browsecomp(args):
    from benchmarks.browsecomp.adapters.browsecomp_client import BrowseCompConfig
    from benchmarks.browsecomp.adapters.system_adapter import BrowseCompSystemAdapter
    from benchmarks.browsecomp.evaluator import BrowseCompEvaluator

    thinking = args.thinking_level if args.thinking_level != "none" else None
    bc_config = BrowseCompConfig(
        model=args.model,
        thinking_level=thinking,
        n_concurrent=args.n_concurrent,
        query_timeout=args.task_timeout,
    )

    playbook = Playbook.load(args.playbook)
    evaluator = BrowseCompEvaluator()

    # Load test task IDs
    test_split = Path(args.task_ids or "benchmarks/browsecomp/splits/test_150.json")
    with open(test_split) as f:
        task_ids = json.load(f)
    if args.limit:
        task_ids = task_ids[:args.limit]

    print(f"Evaluating {len(task_ids)} BC queries, {args.runs} run(s)")
    print(f"Playbook: {args.playbook} ({len(playbook)} entries)")
    print(f"Concurrency: {args.n_concurrent} | Timeout: {args.task_timeout}s\n")

    all_scores = []
    for run in range(1, args.runs + 1):
        run_output = f"{args.output}_r{run}" if args.runs > 1 else args.output
        trace_writer = TraceWriter(run_output)
        adapter = BrowseCompSystemAdapter(config=bc_config, trace_writer=trace_writer)

        print(f"--- Run {run}/{args.runs} ---")
        t0 = time.time()
        traces = adapter.execute(task_ids, playbook, f"eval_r{run}", trace_subdir="traces")

        # Retry timed-out queries once
        timed_out_ids = [t.task_id for t in traces if t.metadata.get("timed_out")]
        if timed_out_ids:
            print(f"  Retrying {len(timed_out_ids)} timed-out queries...")
            retry_traces = adapter.execute(timed_out_ids, playbook, f"eval_r{run}_retry", trace_subdir="traces_retry")
            retry_map = {t.task_id: t for t in retry_traces}
            traces = [retry_map.get(t.task_id, t) if t.metadata.get("timed_out") else t for t in traces]
            still_timed_out = sum(1 for t in traces if t.metadata.get("timed_out"))
            if still_timed_out:
                print(f"  {still_timed_out} queries still timed out after retry")

        elapsed = time.time() - t0
        result = evaluator.evaluate(traces)

        correct = sum(1 for t in traces if t.metadata.get("task_completed"))
        timed_out_final = sum(1 for t in traces if t.metadata.get("timed_out"))
        accuracy = correct / len(traces) * 100
        all_scores.append(accuracy)

        Path(run_output).mkdir(parents=True, exist_ok=True)
        with open(f"{run_output}/results.json", "w") as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": len(traces),
                "timed_out": timed_out_final,
                "elapsed_s": round(elapsed, 1),
                "model": args.model,
                "playbook": args.playbook,
                "playbook_entries": len(playbook),
            }, f, indent=2)

        print(f"  Accuracy: {correct}/{len(traces)} = {accuracy:.1f}% (timed_out={timed_out_final}, {elapsed:.0f}s)\n")

    if args.runs > 1:
        mean = sum(all_scores) / len(all_scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in all_scores) / len(all_scores))
        print(f"{'='*50}")
        print(f"Runs: {[f'{s:.1f}%' for s in all_scores]}")
        print(f"Mean +/- Std: {mean:.1f} +/- {std:.1f}%")
        print(f"{'='*50}")

        with open(f"{args.output}_summary.json", "w") as f:
            json.dump({
                "runs": all_scores,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "model": args.model,
                "playbook": args.playbook,
            }, f, indent=2)


def run_appworld(args):
    from benchmarks.appworld.adapters.system_adapter import AppWorldSystemAdapter
    from benchmarks.appworld.evaluator import AppWorldEvaluator

    thinking = args.thinking_level if args.thinking_level != "none" else None
    playbook = Playbook.load(args.playbook)
    evaluator = AppWorldEvaluator()

    print(f"Evaluating AW split={args.split}, {args.runs} run(s)")
    print(f"Playbook: {args.playbook} ({len(playbook)} entries)")
    print(f"Concurrency: {args.n_concurrent} | Timeout: {args.task_timeout}s\n")

    all_tgc = []
    all_sgc = []
    for run in range(1, args.runs + 1):
        run_output = f"{args.output}_r{run}" if args.runs > 1 else args.output
        trace_writer = TraceWriter(run_output)
        adapter = AppWorldSystemAdapter(
            model=args.model,
            max_remote_calls=args.max_remote_calls,
            trace_writer=trace_writer,
            thinking_level=thinking,
            n_concurrent=args.n_concurrent,
            task_timeout=args.task_timeout,
        )

        task_ids = adapter.load_tasks(args.split, limit=args.limit)
        print(f"--- Run {run}/{args.runs} ({len(task_ids)} tasks) ---")

        t0 = time.time()
        traces = adapter.execute(task_ids, playbook, f"eval_r{run}", trace_subdir="traces")

        # Retry timed-out tasks once
        timed_out_ids = [t.task_id for t in traces if t.metadata.get("timed_out")]
        if timed_out_ids:
            print(f"  Retrying {len(timed_out_ids)} timed-out tasks...")
            retry_traces = adapter.execute(timed_out_ids, playbook, f"eval_r{run}_retry", trace_subdir="traces_retry")
            retry_map = {t.task_id: t for t in retry_traces}
            traces = [retry_map.get(t.task_id, t) if t.metadata.get("timed_out") else t for t in traces]
            still_timed_out = sum(1 for t in traces if t.metadata.get("timed_out"))
            if still_timed_out:
                print(f"  {still_timed_out} tasks still timed out after retry")

        elapsed = time.time() - t0
        result = evaluator.evaluate(traces)

        tgc = result.tgc * 100
        sgc = result.metadata.get("sgc", 0) * 100
        timed_out_final = sum(1 for t in traces if t.metadata.get("timed_out"))
        all_tgc.append(tgc)
        all_sgc.append(sgc)

        Path(run_output).mkdir(parents=True, exist_ok=True)
        with open(f"{run_output}/results.json", "w") as f:
            json.dump({
                "tgc": round(tgc, 2),
                "sgc": round(sgc, 2),
                "score": round(result.score * 100, 2),
                "total_tasks": len(traces),
                "timed_out": timed_out_final,
                "num_scenarios": result.metadata.get("num_complete_scenarios", 0),
                "elapsed_s": round(elapsed, 1),
                "model": args.model,
                "split": args.split,
                "playbook": args.playbook,
                "playbook_entries": len(playbook),
            }, f, indent=2)

        print(f"  TGC: {tgc:.1f}% | SGC: {sgc:.1f}% (timed_out={timed_out_final}, {elapsed:.0f}s)\n")

    if args.runs > 1:
        mean_tgc = sum(all_tgc) / len(all_tgc)
        std_tgc = math.sqrt(sum((s - mean_tgc) ** 2 for s in all_tgc) / len(all_tgc))
        mean_sgc = sum(all_sgc) / len(all_sgc)
        std_sgc = math.sqrt(sum((s - mean_sgc) ** 2 for s in all_sgc) / len(all_sgc))
        print(f"{'='*50}")
        print(f"TGC runs: {[f'{s:.1f}%' for s in all_tgc]}")
        print(f"TGC Mean +/- Std: {mean_tgc:.1f} +/- {std_tgc:.1f}%")
        print(f"SGC runs: {[f'{s:.1f}%' for s in all_sgc]}")
        print(f"SGC Mean +/- Std: {mean_sgc:.1f} +/- {std_sgc:.1f}%")
        print(f"{'='*50}")

        with open(f"{args.output}_summary.json", "w") as f:
            json.dump({
                "tgc_runs": [round(s, 2) for s in all_tgc],
                "tgc_mean": round(mean_tgc, 2),
                "tgc_std": round(std_tgc, 2),
                "sgc_runs": [round(s, 2) for s in all_sgc],
                "sgc_mean": round(mean_sgc, 2),
                "sgc_std": round(std_sgc, 2),
                "model": args.model,
                "split": args.split,
                "playbook": args.playbook,
            }, f, indent=2)


def run_rewardbench2(args):
    from benchmarks.rewardbench2.adapters.rewardbench2_client import RewardBench2Config
    from benchmarks.rewardbench2.adapters.system_adapter import RewardBench2SystemAdapter
    from benchmarks.rewardbench2.benchmark import REWARDBENCH2_CONFIG
    from benchmarks.rewardbench2.evaluator import RewardBench2Evaluator

    thinking = args.thinking_level if args.thinking_level != "none" else None
    playbook = Playbook.load(args.playbook, allowed_sections=REWARDBENCH2_CONFIG.section_names)
    evaluator = RewardBench2Evaluator()

    print(f"Evaluating RB2 split={args.split}, {args.runs} run(s)")
    print(f"Playbook: {args.playbook} ({len(playbook)} entries)")
    print(f"Concurrency: {args.n_concurrent}\n")

    leaderboard_scores = []
    for run in range(1, args.runs + 1):
        run_output = f"{args.output}_r{run}" if args.runs > 1 else args.output
        trace_writer = TraceWriter(run_output)
        config = RewardBench2Config(
            model=args.model,
            n_concurrent=args.n_concurrent,
            max_output_tokens=8192,
            thinking_level=thinking,
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

        Path(run_output).mkdir(parents=True, exist_ok=True)
        with open(f"{run_output}/results.json", "w") as f:
            json.dump({
                "leaderboard_score": result.metadata.get("leaderboard_score", 0.0),
                "avg_pairwise_score": result.score,
                "prompt_accuracy": result.tgc,
                "subset_scores": result.metadata.get("subset_scores", {}),
                "subset_counts": result.metadata.get("subset_counts", {}),
                "elapsed_s": round(elapsed, 1),
                "model": args.model,
                "split": args.split,
                "playbook": args.playbook,
                "playbook_entries": len(playbook),
            }, f, indent=2)

        print(
            f"  Leaderboard: {leaderboard:.2f}% | "
            f"Pairwise: {result.score * 100:.2f}% | "
            f"Prompt acc: {result.tgc * 100:.2f}% ({elapsed:.0f}s)\n"
        )

    if args.runs > 1:
        mean = sum(leaderboard_scores) / len(leaderboard_scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in leaderboard_scores) / len(leaderboard_scores))
        with open(f"{args.output}_summary.json", "w") as f:
            json.dump({
                "leaderboard_runs": [round(s, 2) for s in leaderboard_scores],
                "leaderboard_mean": round(mean, 2),
                "leaderboard_std": round(std, 2),
                "model": args.model,
                "split": args.split,
                "playbook": args.playbook,
            }, f, indent=2)
        print(f"Leaderboard Mean +/- Std: {mean:.2f} +/- {std:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="RCL evaluation")
    parser.add_argument("--benchmark", required=True, choices=["appworld", "browsecomp", "rewardbench2"])
    parser.add_argument("--model", required=True, help="Agent model")
    parser.add_argument("--playbook", required=True, help="Path to playbook JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--runs", type=int, default=1, help="Number of eval runs (for mean+-std)")
    parser.add_argument("--n-concurrent", type=int, default=25, help="Concurrency level")
    parser.add_argument("--task-timeout", type=int, default=900, help="Per-task timeout in seconds (default: 900 = 15min)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--max-remote-calls", type=int, default=100, help="Max tool calls per task (AW)")
    parser.add_argument("--split", default="test_normal", help="Benchmark split")
    parser.add_argument("--task-ids", default=None, help="Path to task IDs JSON (BC override)")
    parser.add_argument("--thinking-level", default="HIGH")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"{'='*60}")
    print(f"RCL Eval: {args.benchmark}")
    print(f"  Model:      {args.model}")
    print(f"  Playbook:   {args.playbook}")
    print(f"  Runs:       {args.runs}")
    print(f"  Concurrent: {args.n_concurrent}")
    print(f"  Timeout:    {args.task_timeout}s")
    print(f"  Output:     {args.output}")
    if args.benchmark == "appworld":
        appworld_root = validate_appworld_root()
        print(f"  AW root:    {appworld_root}")
    print(f"{'='*60}\n")

    if args.benchmark == "browsecomp":
        run_browsecomp(args)
    elif args.benchmark == "rewardbench2":
        run_rewardbench2(args)
    else:
        run_appworld(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run validation on multiple checkpoints from a training run.

Discovers rcl_iter{N}.json playbooks in a training directory and evaluates
each on a *validation* split, producing a per-checkpoint results table.

Validation sets:
  - AppWorld: "dev" split (56 tasks) — held out from training ("train", 89 tasks)
  - BrowseComp: "train_100" split (100 queries) — no separate val split exists,
    so we reuse the training queries with a fixed seed for reproducibility.
    Use --n-val to subsample (e.g., --n-val 50).

Usage:
    cd /path/to/RCL

    # AppWorld: validate every 5th checkpoint on dev split (all 56 tasks)
    python -m scripts.run_validation_sweep \
        --run-dir results-v3/training/gemini/base_ap_gemini \
        --benchmark appworld --model gemini-3.1-flash-lite-preview \
        --n-concurrent 50 \
        --output results-v3/val_sweeps/base_ap_gemini_dev

    # AppWorld: validate with --n-val to subsample dev
    python -m scripts.run_validation_sweep \
        --run-dir results-v3/training/gemini/base_ap_gemini \
        --benchmark appworld --model gemini-3.1-flash-lite-preview \
        --n-val 30 --n-concurrent 50 \
        --output results-v3/val_sweeps/base_ap_gemini_dev_30

    # Specific checkpoints only
    python -m scripts.run_validation_sweep \
        --run-dir results-v3/training/gemini/base_ap_gemini \
        --benchmark appworld --model gemini-3.1-flash-lite-preview \
        --checkpoints 5,10,20,30 \
        --output results-v3/val_sweeps/base_ap_gemini_dev

    # BrowseComp: validate on train_100 (subsample 50)
    python -m scripts.run_validation_sweep \
        --run-dir results-v3/training/nano/base_br_nano \
        --benchmark browsecomp --model openai/gpt-5.4-nano \
        --n-val 50 --every 10 \
        --output results-v3/val_sweeps/base_br_nano
"""

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rcl.core.data_structures import Playbook
from rcl.core.trace_writer import TraceWriter

VAL_SEED = 8675309


def discover_checkpoints(run_dir: str, every: int, checkpoints: str | None) -> list[int]:
    """Find checkpoint iterations in a training run directory."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    iters = []
    for f in run_path.glob("rcl_iter*.json"):
        m = re.match(r"rcl_iter(\d+)\.json", f.name)
        if m:
            iters.append(int(m.group(1)))
    iters.sort()

    if not iters:
        raise FileNotFoundError(f"No rcl_iter*.json checkpoints found in {run_dir}")

    if checkpoints:
        selected = [int(x.strip()) for x in checkpoints.split(",")]
        missing = [i for i in selected if i not in iters]
        if missing:
            print(f"Warning: checkpoints {missing} not found, skipping")
        selected = [i for i in selected if i in iters]
    else:
        selected = [i for i in iters if i % every == 0]

    if not selected:
        raise ValueError(f"No checkpoints selected. Available: {iters}")

    return selected


def _load_val_task_ids_appworld(args) -> list[str]:
    """Load AppWorld dev split task IDs, optionally subsampled."""
    from benchmarks.appworld.adapters.system_adapter import AppWorldSystemAdapter
    from rcl.core.trace_writer import TraceWriter

    # Create a throwaway adapter just to load task IDs
    trace_writer = TraceWriter("/tmp/_val_sweep_probe")
    thinking = args.thinking_level if args.thinking_level != "none" else None
    adapter = AppWorldSystemAdapter(
        model=args.model,
        max_remote_calls=args.max_remote_calls,
        trace_writer=trace_writer,
        thinking_level=thinking,
        n_concurrent=args.n_concurrent,
        task_timeout=args.task_timeout,
    )
    task_ids = adapter.load_tasks("dev")

    if args.n_val and args.n_val < len(task_ids):
        rng = random.Random(VAL_SEED)
        task_ids = rng.sample(task_ids, args.n_val)

    return task_ids


def _load_val_task_ids_browsecomp(args) -> list[str]:
    """Load BrowseComp val split task IDs (30 queries held out from the 580 unused pool)."""
    split_path = Path(args.task_ids or "benchmarks/browsecomp/splits/val_30.json")
    with open(split_path) as f:
        task_ids = json.load(f)

    if args.n_val and args.n_val < len(task_ids):
        rng = random.Random(VAL_SEED)
        task_ids = rng.sample(task_ids, args.n_val)

    return task_ids


def eval_appworld(args, playbook_path: str, output_dir: str, task_ids: list[str]):
    from benchmarks.appworld.appworld_root import validate_appworld_root
    from benchmarks.appworld.adapters.system_adapter import AppWorldSystemAdapter
    from benchmarks.appworld.evaluator import AppWorldEvaluator

    validate_appworld_root()
    thinking = args.thinking_level if args.thinking_level != "none" else None
    playbook = Playbook.load(playbook_path)
    evaluator = AppWorldEvaluator()

    trace_writer = TraceWriter(output_dir)
    adapter = AppWorldSystemAdapter(
        model=args.model,
        max_remote_calls=args.max_remote_calls,
        trace_writer=trace_writer,
        thinking_level=thinking,
        n_concurrent=args.n_concurrent,
        task_timeout=args.task_timeout,
    )

    t0 = time.time()
    traces = adapter.execute(task_ids, playbook, "val", trace_subdir="traces")

    # Retry timed-out tasks once
    timed_out_ids = [t.task_id for t in traces if t.metadata.get("timed_out")]
    if timed_out_ids:
        print(f"  Retrying {len(timed_out_ids)} timed-out tasks...")
        retry_traces = adapter.execute(timed_out_ids, playbook, "val_retry", trace_subdir="traces_retry")
        retry_map = {t.task_id: t for t in retry_traces}
        traces = [retry_map.get(t.task_id, t) if t.metadata.get("timed_out") else t for t in traces]

    elapsed = time.time() - t0
    result = evaluator.evaluate(traces)

    tgc = result.tgc * 100
    sgc = result.metadata.get("sgc", 0) * 100
    timed_out_final = sum(1 for t in traces if t.metadata.get("timed_out"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "tgc": round(tgc, 2),
            "sgc": round(sgc, 2),
            "score": round(result.score * 100, 2),
            "total_tasks": len(traces),
            "timed_out": timed_out_final,
            "elapsed_s": round(elapsed, 1),
            "model": args.model,
            "split": "dev",
            "n_val": len(task_ids),
            "playbook": playbook_path,
            "playbook_entries": len(playbook),
            "val_seed": VAL_SEED,
        }, f, indent=2)

    return {"tgc": tgc, "sgc": sgc, "timed_out": timed_out_final, "elapsed": elapsed, "entries": len(playbook)}


def eval_browsecomp(args, playbook_path: str, output_dir: str, task_ids: list[str]):
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

    playbook = Playbook.load(playbook_path)
    evaluator = BrowseCompEvaluator()

    trace_writer = TraceWriter(output_dir)
    adapter = BrowseCompSystemAdapter(config=bc_config, trace_writer=trace_writer)

    t0 = time.time()
    traces = adapter.execute(task_ids, playbook, "val", trace_subdir="traces")

    # Retry timed-out queries once
    timed_out_ids = [t.task_id for t in traces if t.metadata.get("timed_out")]
    if timed_out_ids:
        print(f"  Retrying {len(timed_out_ids)} timed-out queries...")
        retry_traces = adapter.execute(timed_out_ids, playbook, "val_retry", trace_subdir="traces_retry")
        retry_map = {t.task_id: t for t in retry_traces}
        traces = [retry_map.get(t.task_id, t) if t.metadata.get("timed_out") else t for t in traces]

    elapsed = time.time() - t0
    evaluator.evaluate(traces)

    correct = sum(1 for t in traces if t.metadata.get("task_completed"))
    timed_out_final = sum(1 for t in traces if t.metadata.get("timed_out"))
    accuracy = correct / len(traces) * 100

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": len(traces),
            "timed_out": timed_out_final,
            "elapsed_s": round(elapsed, 1),
            "model": args.model,
            "n_val": len(task_ids),
            "playbook": playbook_path,
            "playbook_entries": len(playbook),
            "val_seed": VAL_SEED,
        }, f, indent=2)

    return {"accuracy": accuracy, "correct": correct, "total": len(traces), "timed_out": timed_out_final, "elapsed": elapsed, "entries": len(playbook)}


def main():
    parser = argparse.ArgumentParser(description="RCL validation sweep across training checkpoints")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing rcl_iter*.json")
    parser.add_argument("--benchmark", required=True, choices=["appworld", "browsecomp"])
    parser.add_argument("--model", required=True, help="Agent model")
    parser.add_argument("--output", required=True, help="Output directory for sweep results")
    parser.add_argument("--every", type=int, default=5, help="Evaluate every Nth checkpoint (default: 5)")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated iterations to evaluate (overrides --every)")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Number of validation examples to use (subsample with fixed seed). "
                             "Default: all (AW dev=56, BC train=100)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of checkpoints to evaluate in parallel (default: 1 = sequential)")
    parser.add_argument("--n-concurrent", type=int, default=25)
    parser.add_argument("--task-timeout", type=int, default=900)
    parser.add_argument("--max-remote-calls", type=int, default=100, help="Max tool calls per task (AW)")
    parser.add_argument("--task-ids", default=None, help="Path to task IDs JSON (BC override)")
    parser.add_argument("--thinking-level", default="HIGH")
    args = parser.parse_args()

    # Discover checkpoints
    selected = discover_checkpoints(args.run_dir, args.every, args.checkpoints)
    run_path = Path(args.run_dir)

    # Load validation task IDs once (same set for all checkpoints)
    if args.benchmark == "appworld":
        val_task_ids = _load_val_task_ids_appworld(args)
        val_split_name = "dev"
    else:
        val_task_ids = _load_val_task_ids_browsecomp(args)
        val_split_name = "val_30"

    print(f"{'='*60}")
    print(f"RCL Validation Sweep: {args.benchmark}")
    print(f"  Run dir:      {args.run_dir}")
    print(f"  Model:        {args.model}")
    print(f"  Val split:    {val_split_name} ({len(val_task_ids)} tasks)")
    print(f"  Val seed:     {VAL_SEED}")
    print(f"  Checkpoints:  {selected}")
    print(f"  Parallel:     {args.parallel} checkpoint(s) at a time")
    print(f"  Concurrent:   {args.n_concurrent} tasks per checkpoint")
    print(f"  Output:       {args.output}")
    print(f"{'='*60}\n")

    results = {}
    eval_fn = eval_appworld if args.benchmark == "appworld" else eval_browsecomp

    def _load_cached(iter_num: int) -> dict | None:
        """Try to load a cached result for this checkpoint."""
        results_file = Path(f"{args.output}/iter{iter_num}/results.json")
        if not results_file.exists():
            return None
        with open(results_file) as f:
            cached = json.load(f)
        if args.benchmark == "appworld":
            return {
                "tgc": cached["tgc"], "sgc": cached["sgc"],
                "timed_out": cached.get("timed_out", 0),
                "elapsed": cached.get("elapsed_s", 0),
                "entries": cached.get("playbook_entries", 0),
            }
        else:
            return {
                "accuracy": cached["accuracy"], "correct": cached["correct"],
                "total": cached["total"], "timed_out": cached.get("timed_out", 0),
                "elapsed": cached.get("elapsed_s", 0),
                "entries": cached.get("playbook_entries", 0),
            }

    def _eval_checkpoint(iter_num: int) -> tuple[int, dict]:
        """Evaluate a single checkpoint (may run in a thread)."""
        cached = _load_cached(iter_num)
        if cached is not None:
            print(f"\n--- iter{iter_num}: already evaluated, loading cached result ---")
            return iter_num, cached

        playbook_path = str(run_path / f"rcl_iter{iter_num}.json")
        iter_output = f"{args.output}/iter{iter_num}"
        print(f"\n--- iter{iter_num} ({playbook_path}) ---")
        result = eval_fn(args, playbook_path, iter_output, val_task_ids)

        if args.benchmark == "appworld":
            print(f"  => iter{iter_num}: TGC: {result['tgc']:.1f}% | SGC: {result['sgc']:.1f}% | entries: {result['entries']} ({result['elapsed']:.0f}s)")
        else:
            print(f"  => iter{iter_num}: Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.1f}% ({result['elapsed']:.0f}s)")

        return iter_num, result

    if args.parallel <= 1:
        # Sequential
        for iter_num in selected:
            itn, result = _eval_checkpoint(iter_num)
            results[f"iter{itn}"] = result
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_eval_checkpoint, itn): itn for itn in selected}
            for future in as_completed(futures):
                itn, result = future.result()
                results[f"iter{itn}"] = result

        # Re-sort by iteration number for display
        results = dict(sorted(results.items(), key=lambda kv: int(kv[0].replace("iter", ""))))

    # Print summary table
    print(f"\n{'='*60}")
    print(f"VALIDATION SWEEP SUMMARY ({val_split_name}, n={len(val_task_ids)})")
    print(f"{'='*60}")

    if args.benchmark == "appworld":
        print(f"{'Checkpoint':<15} {'Entries':>8} {'TGC':>8} {'SGC':>8} {'Timeout':>8}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for label, r in results.items():
            print(f"{label:<15} {r['entries']:>8} {r['tgc']:>7.1f}% {r['sgc']:>7.1f}% {r['timed_out']:>8}")
    else:
        print(f"{'Checkpoint':<15} {'Entries':>8} {'Accuracy':>10} {'Correct':>8} {'Timeout':>8}")
        print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for label, r in results.items():
            print(f"{label:<15} {r['entries']:>8} {r['accuracy']:>9.1f}% {r['correct']:>8} {r['timed_out']:>8}")

    print(f"{'='*60}")

    # Save summary
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output}/sweep_summary.json", "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "model": args.model,
            "run_dir": args.run_dir,
            "val_split": val_split_name,
            "n_val": len(val_task_ids),
            "val_seed": VAL_SEED,
            "checkpoints": {
                k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in results.items()
            },
        }, f, indent=2)
    print(f"\nSummary saved to {args.output}/sweep_summary.json")


if __name__ == "__main__":
    main()

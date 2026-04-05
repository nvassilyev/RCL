#!/usr/bin/env python3
"""Quick baseline test for AppWorld via RCL.

Usage:
    cd /path/to/RCL
    python -m benchmarks.appworld.scripts.run_baseline --limit 1
"""

import argparse
import json
import random
from pathlib import Path

from rcl.core.data_structures import Playbook
from rcl.core.trace_writer import TraceWriter

from ..adapters.system_adapter import AppWorldSystemAdapter
from ..evaluator import AppWorldEvaluator


def main():
    parser = argparse.ArgumentParser(description="AppWorld baseline eval")
    parser.add_argument("--model", default="google/gemini-3-flash-preview")
    parser.add_argument("--max-remote-calls", type=int, default=100)
    parser.add_argument("--split", default="test_normal")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/eval/aw_test")
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    playbook = Playbook()
    trace_writer = TraceWriter(args.output)
    adapter = AppWorldSystemAdapter(
        model=args.model,
        max_remote_calls=args.max_remote_calls,
        trace_writer=trace_writer,
    )
    evaluator = AppWorldEvaluator()

    task_ids = adapter.load_tasks(args.split, limit=args.limit)
    print(f"Running {len(task_ids)} tasks from {args.split}...\n")

    traces = adapter.execute(task_ids, playbook, "baseline")
    result = evaluator.evaluate(traces)

    print(f"\n{'='*60}")
    print(f"Pass%: {result.score*100:.1f}% | TGC: {result.tgc*100:.1f}%")
    print(f"{'='*60}")

    # Show decoupled trace structure
    if traces:
        t = traces[0]
        print(f"\n--- TRACE (first 1500 chars) ---")
        print(t.trace[:1500])
        if len(t.trace) > 1500:
            print(f"... ({len(t.trace)} total chars)")
        print(f"\n--- EVALUATION DETAILS ---")
        print(t.metadata.get("evaluation_details", "(not set)"))

    with open(f"{args.output}/results.json", "w") as f:
        json.dump({"score": result.score, "tgc": result.tgc}, f, indent=2)
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()

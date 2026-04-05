#!/usr/bin/env python3
"""Quick baseline test for BrowseComp+ via RCL.

Usage:
    cd /path/to/RCL
    python -m benchmarks.browsecomp.scripts.run_baseline --limit 1
"""

import argparse
import json
import os
import random
from pathlib import Path

from rcl.core.data_structures import Playbook
from rcl.core.trace_writer import TraceWriter

from ..adapters.browsecomp_client import BrowseCompConfig
from ..adapters.system_adapter import BrowseCompSystemAdapter
from ..evaluator import BrowseCompEvaluator
from ..benchmark import BROWSECOMP_SECTIONS, BROWSECOMP_CONFIG


def main():
    parser = argparse.ArgumentParser(description="BrowseComp+ baseline eval")
    parser.add_argument("--playbook", default=None, help="Path to playbook JSON")
    parser.add_argument("--model", default="google/gemini-3-flash-preview")
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--mcp-url", default="http://localhost:8081/mcp/")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/rcl_bc_test")
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load playbook
    if args.playbook:
        playbook = Playbook.load(args.playbook, allowed_sections=set(BROWSECOMP_SECTIONS.keys()))
    else:
        playbook = Playbook(allowed_sections=set(BROWSECOMP_SECTIONS.keys()))
    print(f"Playbook: {len(playbook)} entries")

    # Build system adapter
    bc_config = BrowseCompConfig(
        model=args.model,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        mcp_url=args.mcp_url,
        max_steps=args.max_steps,
        n_concurrent=args.n_concurrent,
    )
    trace_writer = TraceWriter(args.output)
    adapter = BrowseCompSystemAdapter(config=bc_config, trace_writer=trace_writer)
    evaluator = BrowseCompEvaluator()

    # Load and run
    task_ids = adapter.load_tasks(limit=args.limit)
    random.shuffle(task_ids)
    task_ids = task_ids[:args.limit]
    print(f"Running {len(task_ids)} queries...\n")

    traces = adapter.execute(task_ids, playbook, "baseline", max_workers=args.n_concurrent)
    result = evaluator.evaluate(traces)

    # Print results
    print(f"\n{'='*60}")
    print(f"Accuracy: {result.score*100:.1f}% ({result.metadata.get('correct_count', 0)}/{len(task_ids)})")
    print(f"{'='*60}")

    # Show decoupled trace structure for first result
    if traces:
        t = traces[0]
        print(f"\n--- TRACE (trace.trace) ---")
        print(t.trace[:2000])
        if len(t.trace) > 2000:
            print(f"... ({len(t.trace)} total chars)")
        print(f"\n--- EVALUATION DETAILS (trace.metadata['evaluation_details']) ---")
        print(t.metadata.get("evaluation_details", "(not set)"))

    # Save
    with open(f"{args.output}/results.json", "w") as f:
        json.dump({
            "accuracy": result.score,
            "queries": [{
                "query_id": t.task_id,
                "correct": t.metadata.get("correct"),
                "extracted_answer": t.metadata.get("extracted_answer"),
                "gold_answer": t.metadata.get("gold_answer"),
            } for t in traces],
        }, f, indent=2)
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()

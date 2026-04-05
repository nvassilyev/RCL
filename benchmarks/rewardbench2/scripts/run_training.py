#!/usr/bin/env python3
"""Run RCL training for RewardBench 2."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from rcl.core.config import RCLConfig
from rcl.core.data_structures import Playbook
from rcl.core.optimizer import RCLOptimizer
from rcl.core.trace_writer import TraceWriter
from rcl.components.reflector import RCLReflector
from rcl.components.mutator import RCLMutator
from rcl.prompts.mutator import build_mutator_prompt

from ..adapters.rewardbench2_client import RewardBench2Config
from ..adapters.system_adapter import RewardBench2SystemAdapter
from ..benchmark import REWARDBENCH2_CONFIG
from ..evaluator import RewardBench2Evaluator


def build_rewardbench2(args):
    thinking = args.thinking_level if args.thinking_level != "none" else None
    config = RewardBench2Config(
        model=args.model,
        n_concurrent=args.n_concurrent,
        max_output_tokens=args.max_output_tokens,
        thinking_level=thinking,
    )
    trace_writer = TraceWriter(args.output)
    adapter = RewardBench2SystemAdapter(config=config, trace_writer=trace_writer)
    evaluator = RewardBench2Evaluator()
    seed_playbook = Playbook.load(
        args.seed_playbook or "benchmarks/rewardbench2/playbooks/seed_playbook.json",
        allowed_sections=REWARDBENCH2_CONFIG.section_names,
    )
    train_ids = adapter.load_tasks(args.train_split)
    val_ids = adapter.load_tasks(args.val_split)
    if args.train_pool:
        train_ids = train_ids[: args.train_pool]
    if args.val_pool:
        val_ids = val_ids[: args.val_pool]
    return adapter, evaluator, seed_playbook, train_ids, val_ids, trace_writer


def main() -> None:
    parser = argparse.ArgumentParser(description="RCL training for RewardBench 2")
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--reflector-model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--mutator-model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch", type=int, default=1,
                        help="Number of failed traces to reflect on per iteration (batching primitive)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-pool", type=int, default=None)
    parser.add_argument("--val-pool", type=int, default=None)
    parser.add_argument("--n-concurrent", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--add-only", action="store_true", help="Restrict mutations to ADD only")
    parser.add_argument("--prune-threshold", type=int, default=0)
    parser.add_argument("--seed-playbook", default=None)
    parser.add_argument("--thinking-level", default="none")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    # Composable feature flags
    parser.add_argument("--pp", action="store_true", help="Enable PP perturbations")
    parser.add_argument("--grouped-rollouts", type=int, default=0, metavar="K",
                        help="Grouped rollouts: execute each task K times for contrastive signal (0 = off)")
    parser.add_argument("--perturbation-set", default="rich")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    if args.n_concurrent is None:
        args.n_concurrent = args.batch_size
    if args.output is None:
        model_dir = args.model.replace("/", "_")
        variant_tag = ""
        if args.pp:
            variant_tag += "_pp"
        if args.grouped_rollouts > 1:
            variant_tag += "_group"
        args.output = f"results/training/{model_dir}/rb2{variant_tag}"

    random.seed(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    adapter, evaluator, seed_playbook, train_ids, val_ids, trace_writer = build_rewardbench2(args)

    print(f"{'=' * 60}")
    print("RCL Training: rewardbench2")
    print(f"  Model:      {args.model}")
    print(f"  Reflector:  {args.reflector_model}")
    print(f"  Mutator:    {args.mutator_model}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Concurrent: {args.n_concurrent}")
    print(f"  Output:     {args.output}")
    print(f"  Train pool: {len(train_ids)}")
    print(f"  Val pool:   {len(val_ids)}")
    if args.pp:
        print(f"  PP:         {args.perturbation_set}")
    if args.grouped_rollouts > 1:
        print(f"  Grouped:    K={args.grouped_rollouts}")
    print(f"{'=' * 60}\n")

    # Build mutator
    mutator_prompt = build_mutator_prompt(
        sections=REWARDBENCH2_CONFIG.sections,
        add_only=args.add_only,
    )
    mutator = RCLMutator(
        model=args.mutator_model,
        add_only=args.add_only,
        prompt_template=mutator_prompt,
        allowed_sections=REWARDBENCH2_CONFIG.section_names,
    )

    reflector = RCLReflector(
        model=args.reflector_model,
        domain_description=REWARDBENCH2_CONFIG.domain_description,
    )

    # Build config
    config = RCLConfig(
        model=args.model,
        reflector_model=args.reflector_model,
        mutator_model=args.mutator_model,
        iterations=args.iterations,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        seed=args.seed,
        output_dir=args.output,
        skip_validation=args.skip_validation,
        prune_threshold=args.prune_threshold,
        max_workers=args.n_concurrent,
        perturbation_set=args.perturbation_set if args.pp else "",
        group_size=args.grouped_rollouts if args.grouped_rollouts > 1 else 1,
    )
    config.save_yaml(f"{args.output}/config.yaml")

    # Build optimizer
    optimizer = RCLOptimizer(
        system_adapter=adapter,
        evaluator=evaluator,
        reflector=reflector,
        mutator=mutator,
        config=config,
        trace_writer=trace_writer,
    )

    best_playbook = optimizer.optimize(
        seed_playbook=seed_playbook,
        train_task_ids=train_ids,
        val_task_ids=None if args.skip_validation else val_ids,
    )

    print(f"\nFinal playbook: {len(best_playbook)} entries")
    print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()

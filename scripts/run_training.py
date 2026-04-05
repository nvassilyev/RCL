#!/usr/bin/env python3
"""Run RCL training loop.

Usage:
    # AppWorld (base loop)
    python -m scripts.run_training --benchmark appworld --batch-size 15 --iterations 30

    # BrowseComp+
    python -m scripts.run_training --benchmark browsecomp --batch-size 5 --iterations 30

    # RewardBench2
    python -m scripts.run_training --benchmark rewardbench2 --batch-size 16 --iterations 30
"""

import argparse
import json
import random
from pathlib import Path

from rcl.core.config import RCLConfig
from rcl.core.data_structures import Playbook
from rcl.core.optimizer import RCLOptimizer
from rcl.core.trace_writer import TraceWriter
from rcl.components.reflector import RCLReflector
from rcl.components.mutator import RCLMutator
from rcl.prompts.mutator import build_mutator_prompt
from rcl.prompts.reflector import get_reflector_prompt_templates


def build_browsecomp(args):
    from benchmarks.browsecomp.benchmark import BROWSECOMP_CONFIG
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
    trace_writer = TraceWriter(args.output)
    adapter = BrowseCompSystemAdapter(config=bc_config, trace_writer=trace_writer)
    evaluator = BrowseCompEvaluator()
    seed_playbook = Playbook.load(args.seed_playbook or "benchmarks/browsecomp/playbooks/empty_playbook.json")

    # Use fixed train split
    train_split = Path("benchmarks/browsecomp/splits/train_100.json")
    if train_split.exists():
        with open(train_split) as f:
            train_ids = json.load(f)
        if args.train_pool:
            train_ids = train_ids[:args.train_pool]
    else:
        train_ids = adapter.load_tasks(limit=args.train_pool or 100)

    return adapter, evaluator, BROWSECOMP_CONFIG, seed_playbook, train_ids, trace_writer


def build_rewardbench2(args):
    from benchmarks.rewardbench2.adapters.rewardbench2_client import RewardBench2Config
    from benchmarks.rewardbench2.adapters.system_adapter import RewardBench2SystemAdapter
    from benchmarks.rewardbench2.benchmark import REWARDBENCH2_CONFIG
    from benchmarks.rewardbench2.evaluator import RewardBench2Evaluator

    thinking = args.thinking_level if args.thinking_level != "none" else None
    rb2_config = RewardBench2Config(
        model=args.model,
        n_concurrent=args.n_concurrent,
        max_output_tokens=8192,
        thinking_level=thinking,
    )
    trace_writer = TraceWriter(args.output)
    adapter = RewardBench2SystemAdapter(config=rb2_config, trace_writer=trace_writer)
    evaluator = RewardBench2Evaluator()
    seed_playbook = Playbook.load(
        args.seed_playbook or "benchmarks/rewardbench2/playbooks/seed_playbook.json",
        allowed_sections=REWARDBENCH2_CONFIG.section_names,
    )
    train_ids = adapter.load_tasks(args.split, limit=args.train_pool)

    return adapter, evaluator, REWARDBENCH2_CONFIG, seed_playbook, train_ids, trace_writer


def build_appworld(args):
    from benchmarks.appworld.benchmark import APPWORLD_CONFIG
    from benchmarks.appworld.adapters.system_adapter import AppWorldSystemAdapter
    from benchmarks.appworld.evaluator import AppWorldEvaluator

    trace_writer = TraceWriter(args.output)
    thinking = args.thinking_level if args.thinking_level != "none" else None
    adapter = AppWorldSystemAdapter(
        model=args.model,
        max_remote_calls=args.max_remote_calls,
        trace_writer=trace_writer,
        thinking_level=thinking,
        n_concurrent=args.n_concurrent,
        task_timeout=args.task_timeout,
    )
    evaluator = AppWorldEvaluator()
    bench_config = APPWORLD_CONFIG
    seed_playbook = Playbook.load(args.seed_playbook or "benchmarks/appworld/playbooks/seed_playbook.json")
    train_ids = adapter.load_tasks(args.split, limit=args.train_pool)

    return adapter, evaluator, bench_config, seed_playbook, train_ids, trace_writer


def main():
    parser = argparse.ArgumentParser(description="RCL training loop")
    parser.add_argument("--benchmark", required=True, choices=["appworld", "browsecomp", "rewardbench2"])
    parser.add_argument("--model", required=True, help="Agent model (e.g. openai/gpt-5.4-nano, anthropic/claude-opus-4-6, gemini/gemini-3-flash-preview)")
    parser.add_argument("--reflector-model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--mutator-model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--mini-batch", type=int, default=1,
                        help="Number of failed traces to reflect on per iteration (batching primitive)")
    parser.add_argument("--max-remote-calls", type=int, default=100)
    parser.add_argument("--split", default="train", help="AppWorld split")
    parser.add_argument("--train-pool", type=int, default=None, help="Limit train task pool size")
    parser.add_argument("--n-concurrent", type=int, default=None, help="Concurrency (default: batch-size)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--add-only", action="store_true", help="Restrict mutations to ADD only (no UPDATE/DELETE)")
    parser.add_argument("--prune-threshold", type=int, default=0, help="Auto-prune entries with harmful-helpful >= N (0=disabled)")
    parser.add_argument("--entry-char-cap", type=int, default=0,
                        help="Reject ADD/UPDATE entries longer than N chars (0=disabled)")
    parser.add_argument("--task-timeout", type=int, default=900, help="Per-task timeout in seconds (default: 900 = 15min)")
    parser.add_argument("--failure-replay-ratio", type=float, default=0.0,
                        help="Fraction of each batch reserved for replayed failure tasks")
    parser.add_argument("--failure-replay-max-size", type=int, default=0,
                        help="Maximum retained failure tasks in replay (0 = unbounded)")
    parser.add_argument("--failure-replay-no-unseen-first", action="store_true",
                        help="Do not prioritize unseen tasks for the fresh half of each batch")
    parser.add_argument("--failure-replay-passes-to-graduate", type=int, default=5,
                        help="Consecutive passes needed to graduate a task from replay (0 = never graduate via passes)")
    parser.add_argument("--failure-replay-failures-to-evict", type=int, default=3,
                        help="Consecutive failures (post-reflection) to evict stuck tasks from replay (0 = disabled)")
    parser.add_argument("--seed-playbook", default=None, help="Path to seed playbook (for resuming from checkpoint)")
    parser.add_argument(
        "--resume", default=None,
        help="Resume from an existing output directory. Auto-discovers latest iteration and loads all state.",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="0-based iteration index to resume from (e.g. 40 resumes from rcl_iter40 and runs iteration 41 onward)",
    )
    parser.add_argument(
        "--resume-history",
        default=None,
        help="Path to existing history.json to append to when resuming; entries beyond start-iteration are dropped",
    )
    parser.add_argument("--thinking-level", default="HIGH", help="Thinking level (HIGH/MEDIUM/LOW/none)")
    parser.add_argument("--auxiliary-losses", action="store_true", default=False,
                        help="Enable enriched reflector with failure attribution, root cause analysis, and coverage gaps")
    # Composable feature flags
    parser.add_argument("--pp", action="store_true", help="Enable PP perturbations")
    parser.add_argument("--grouped-rollouts", type=int, default=0, metavar="K",
                        help="Grouped rollouts: execute each task K times for contrastive signal (0 = off)")
    parser.add_argument("--batched-reflection", action="store_true", help="Reflect on all signal traces in a single LLM call (vs per-trace)")
    parser.add_argument("--perturbation-set", default="",
                        help="Perturbation set for PP (minimal/standard/rich/full, empty=off)")
    parser.add_argument("--single-pass", action="store_true",
                        help="Sequential 1-pass through all tasks (no replay, no repeats)")
    parser.add_argument("--reflect-all-traces", action="store_true",
                        help="Include passing traces in reflection signal (not just failures)")
    parser.add_argument("--playbook-budget", type=int, default=0,
                        help="Soft max entries for playbook (0 = unlimited, communicated to mutator as guidance)")
    parser.add_argument("--dual-trace", action="store_true", default=False,
                        help="Run each task twice (baseline + full PP) for credit-attribution reflection")
    parser.add_argument("--omit-mutator-traces", action="store_true", default=False,
                        help="Don't pass raw traces to the mutator (it only sees the reflector analysis)")
    parser.add_argument("--summarize-traces", action="store_true", default=False,
                        help="Condense raw traces using a cheap LLM before passing to the mutator")
    parser.add_argument("--trace-summarizer-model", type=str, default="anthropic/claude-opus-4-6",
                        help="Model for trace summarization (default: claude-opus-4-6)")

    # Optimization state
    parser.add_argument("--optimization-state", action="store_true", default=False,
                        help="Enable rolling optimization state tracking")
    args = parser.parse_args()

    # Validate thinking level for models that require it
    if ("3.1-lite" in args.model or "flash-lite" in args.model) and args.thinking_level.lower() == "none":
        print("WARNING: gemini-3.1-flash-lite-preview requires thinking=HIGH. Overriding thinking-level to HIGH.")
        args.thinking_level = "HIGH"

    if args.n_concurrent is None:
        args.n_concurrent = args.batch_size

    omit_mutator_trace = args.omit_mutator_traces

    # Default output dir
    if args.output is None:
        model_tag = args.model.replace("/", "_").replace("-", "_")
        args.output = f"results/training/{args.benchmark}_{model_tag}"

    random.seed(args.seed)

    # --resume auto-discovers latest iteration and sets output dir
    if args.resume:
        resume_dir = args.resume.rstrip("/")
        latest = RCLOptimizer.discover_latest_iteration(resume_dir)
        if latest is None:
            raise ValueError(f"No completed iterations found in {resume_dir}")
        args.start_iteration = latest
        args.output = resume_dir
        checkpoint_pb = Path(resume_dir) / "iterations" / f"iter_{latest}" / "playbook.json"
        if checkpoint_pb.exists() and not args.seed_playbook:
            args.seed_playbook = str(checkpoint_pb)
            print(f"Resuming from {resume_dir}, iteration {latest} (playbook: {checkpoint_pb})")
        else:
            print(f"Resuming from {resume_dir}, iteration {latest}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    existing_history = None
    if args.resume_history:
        history_path = Path(args.resume_history)
        if history_path.exists():
            with open(history_path) as f:
                loaded_history = json.load(f)
            if not isinstance(loaded_history, list):
                raise ValueError(f"Resume history must be a list: {history_path}")
            if args.start_iteration > 0:
                loaded_history = [
                    entry for entry in loaded_history
                    if int(entry.get("iteration", 0)) <= args.start_iteration
                ]
            existing_history = loaded_history
        else:
            raise FileNotFoundError(f"Resume history not found: {history_path}")

    print(f"{'='*60}")
    print(f"RCL Training: {args.benchmark}")
    print(f"  Model:      {args.model}")
    print(f"  Reflector:  {args.reflector_model}")
    print(f"  Mutator:    {args.mutator_model}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mini-batch: {args.mini_batch}")
    print(f"  Add only:   {args.add_only}")
    print(f"  Prune thr:  {args.prune_threshold}")
    print(f"  Concurrent: {args.n_concurrent}")
    print(f"  Thinking:   {args.thinking_level}")
    print(
        "  Replay:     "
        f"ratio={args.failure_replay_ratio:.2f} "
        f"max={args.failure_replay_max_size or 'inf'} "
        f"unseen_first={not args.failure_replay_no_unseen_first} "
        f"passes_to_graduate={args.failure_replay_passes_to_graduate} "
        f"failures_to_evict={args.failure_replay_failures_to_evict}"
    )
    reflector_prompt_style = "enriched" if args.auxiliary_losses else "standard"
    print(f"  Aux losses: {args.auxiliary_losses}")
    print(f"  Output:     {args.output}")
    if args.seed_playbook:
        print(f"  Seed PB:    {args.seed_playbook}")
    if args.resume:
        print(f"  Resume:     {args.resume} @ iter {args.start_iteration}")
    elif args.start_iteration:
        print(f"  Resume @:   iter {args.start_iteration}")
    if args.resume_history:
        print(f"  History:    {args.resume_history}")
    if args.pp:
        print(f"  PP:         {args.perturbation_set}")
    if args.grouped_rollouts > 1:
        print(f"  Grouped:    K={args.grouped_rollouts}")
    if args.single_pass:
        print(f"  Single-pass: True")
    if args.reflect_all_traces:
        print(f"  Refl all:   True")
    if args.playbook_budget > 0:
        print(f"  PB budget:  {args.playbook_budget}")
    if args.entry_char_cap > 0:
        print(f"  Entry cap:  {args.entry_char_cap}")
    if args.dual_trace:
        print(f"  Dual-trace: True")
    if args.batched_reflection:
        print(f"  Batched ref: True")
    print(f"{'='*60}\n")

    # Build benchmark components
    if args.benchmark == "browsecomp":
        adapter, evaluator, bench_config, seed_playbook, train_ids, trace_writer = build_browsecomp(args)
    elif args.benchmark == "rewardbench2":
        adapter, evaluator, bench_config, seed_playbook, train_ids, trace_writer = build_rewardbench2(args)
    else:
        adapter, evaluator, bench_config, seed_playbook, train_ids, trace_writer = build_appworld(args)

    if args.batch_size <= 0:
        args.batch_size = len(train_ids)
    if args.n_concurrent is None or args.n_concurrent <= 0:
        args.n_concurrent = args.batch_size

    print(f"Train pool: {len(train_ids)} tasks")
    print(f"Seed playbook: {len(seed_playbook)} entries")

    # Build mutator prompt template from benchmark config
    enriched_reflection = args.auxiliary_losses
    mutator_prompt = build_mutator_prompt(
        sections=bench_config.sections,
        add_only=args.add_only,
        include_trace=not omit_mutator_trace,
        playbook_budget=args.playbook_budget,
        include_optimization_state=args.optimization_state,
        enriched_reflection=enriched_reflection,
    )

    # Build reflector + mutator
    llm_thinking = args.thinking_level.lower() if args.thinking_level and args.thinking_level != "none" else None
    reflector_prompt_template = get_reflector_prompt_templates(reflector_prompt_style)
    reflector = RCLReflector(
        model=args.reflector_model,
        domain_description=bench_config.domain_description,
        prompt_template=reflector_prompt_template,
        thinking=llm_thinking,
        batched_reflection=args.batched_reflection,
    )

    mutator = RCLMutator(
        model=args.mutator_model,
        add_only=args.add_only,
        prompt_template=mutator_prompt,
        allowed_sections=bench_config.section_names,
        thinking=llm_thinking,
        include_trace_in_prompt=not omit_mutator_trace,
    )

    # Build optimizer config
    config = RCLConfig(
        model=args.model,
        reflector_model=args.reflector_model,
        mutator_model=args.mutator_model,
        iterations=args.iterations,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        seed=args.seed,
        output_dir=args.output,
        skip_validation=True,
        prune_threshold=args.prune_threshold,
        entry_char_cap=args.entry_char_cap,
        max_workers=args.n_concurrent,
        thinking_level=args.thinking_level,
        reflector_prompt_style=reflector_prompt_style,
        failure_replay_ratio=args.failure_replay_ratio,
        failure_replay_max_size=args.failure_replay_max_size,
        failure_replay_unseen_first=not args.failure_replay_no_unseen_first,
        single_pass=args.single_pass,
        reflect_all_traces=args.reflect_all_traces,
        playbook_budget=args.playbook_budget,
        failure_replay_passes_to_graduate=args.failure_replay_passes_to_graduate,
        failure_replay_failures_to_evict=args.failure_replay_failures_to_evict,
        dual_trace=args.dual_trace,
        perturbation_set=args.perturbation_set if args.pp else "",
        group_size=args.grouped_rollouts if args.grouped_rollouts > 1 else 1,
        batched_reflection=args.batched_reflection,
        use_optimization_state=args.optimization_state,
    )
    config.save_yaml(f"{args.output}/config.yaml")

    # Build trace summarizer (optional)
    trace_summarizer = None
    if args.summarize_traces:
        from rcl.components.trace_summarizer import TraceSummarizer
        trace_summarizer = TraceSummarizer(
            model=args.trace_summarizer_model,
        )
        print(f"Trace summarizer: {args.trace_summarizer_model}")

    # Build optimizer
    optimizer = RCLOptimizer(
        system_adapter=adapter,
        evaluator=evaluator,
        reflector=reflector,
        mutator=mutator,
        config=config,
        trace_writer=trace_writer,
        trace_summarizer=trace_summarizer,
    )

    # Run optimization
    best_playbook = optimizer.optimize(
        seed_playbook=seed_playbook,
        train_task_ids=train_ids,
        output_dir=args.output,
        start_iteration=args.start_iteration,
        existing_history=existing_history,
    )

    print(f"\nFinal playbook: {len(best_playbook)} entries")
    print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()

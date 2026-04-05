# RewardBench2

Benchmark adapter for [RewardBench 2](https://github.com/allenai/reward-bench-2) — a response ranking benchmark where an LLM judge rates multiple candidate responses for the same prompt.

## Metrics

- **`leaderboard_score`**: Average of per-subset scores, using official RewardBench 2 ties scoring for the `Ties` subset (primary metric).
- **`avg_pairwise_score`**: Prompt-level fraction of correct-vs-incorrect pairwise comparisons ranked correctly.
- **`prompt_accuracy`**: Fraction of prompts where all correct responses outrank all incorrect responses.

## Prerequisites

The dataset and splits are included in the repo — no external setup required.

Set the API key for your model provider (see [main README](../../README.md#supported-models)):

```bash
export OPENAI_API_KEY=sk-...        # for OpenAI models
export ANTHROPIC_API_KEY=sk-ant-... # for Anthropic models
export GEMINI_API_KEY=...           # for Gemini models
```

### Regenerating data (optional)

The dataset and splits ship with the repo. If you need to regenerate them:

```bash
# Download dataset from HuggingFace
python -m benchmarks.rewardbench2.scripts.download_dataset

# Create train/val/test splits (70/15/15, stratified by subset)
python -m benchmarks.rewardbench2.scripts.create_splits
```

## Usage

### Smoke test

```bash
python -m benchmarks.rewardbench2.scripts.run_baseline \
  --model openai/gpt-5.4-nano \
  --split test --limit 10 --n-concurrent 10
```

### Training

```bash
python -m scripts.run_training \
  --benchmark rewardbench2 \
  --model openai/gpt-5.4-nano \
  --batch-size 16 --n-concurrent 16 --iterations 30
```

### Evaluation

```bash
python -m scripts.run_eval \
  --benchmark rewardbench2 \
  --model openai/gpt-5.4-nano \
  --playbook results/training/<run_dir>/rcl_best.json \
  --output results/evals/rb2_eval \
  --split test --n-concurrent 32
```

## Files

```
rewardbench2/
  benchmark.py              # BenchmarkConfig (sections, domain description)
  evaluator.py              # Evaluator (leaderboard scoring with ties handling)
  data/
    rewardbench2.jsonl       # Full dataset (1,865 tasks)
  splits/
    train.json               # 1,307 tasks
    val.json                 # 277 tasks
    test.json                # 281 tasks
  adapters/
    system_adapter.py        # SystemAdapter (runs LLM judge on candidate responses)
    rewardbench2_client.py   # Client (prompt formatting, JSON parsing, scoring)
  playbooks/
    seed_playbook.json       # Starting playbook with judging guidelines
  scripts/
    run_baseline.py          # Quick smoke test
    run_eval.py              # Standalone evaluation script
    run_training.py          # Standalone training script
    download_dataset.py      # Download from HuggingFace
    create_splits.py         # Generate train/val/test splits
```

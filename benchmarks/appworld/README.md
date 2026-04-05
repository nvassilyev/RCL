# AppWorld

Benchmark adapter for [AppWorld](https://github.com/stonybrooknlp/appworld) — a multi-step interactive coding benchmark where an AI agent writes Python code to interact with app APIs (email, calendar, Venmo, Spotify, etc.) to accomplish user requests.

## Metrics

- **Task Goal Completion (TGC)**: Fraction of tasks where all unit tests pass. Primary metric.
- **Pass percentage**: Per-task fraction of unit tests passed (used as learning signal).

## Prerequisites

AppWorld requires a separate Python environment to run its local task server. The RCL training code connects to this server over HTTP.

1. **Install AppWorld** following the [official instructions](https://github.com/stonybrooknlp/appworld). This creates a data directory with task definitions and a server that hosts the app environments.

2. **Set environment variables**:
   ```bash
   export APPWORLD_ROOT=/path/to/appworld          # root dir containing data/tasks/ (733 tasks)
   export APPWORLD_SERVER_PYTHON=/path/to/appworld/venv/bin/python  # Python with appworld installed
   ```

3. **Set your model's API key** (see [main README](../../README.md#supported-models)):
   ```bash
   export OPENAI_API_KEY=sk-...        # for OpenAI models
   export ANTHROPIC_API_KEY=sk-ant-... # for Anthropic models
   export GEMINI_API_KEY=...           # for Gemini models
   ```

## How It Works

The `AppWorldSystemAdapter` manages a pool of AppWorld server instances (one per concurrent worker). Each server hosts an isolated app environment for a single task. The adapter:

1. Auto-launches AppWorld servers on free ports via `appworld_server.py`
2. Runs the agent with a manual tool-calling loop against the server's APIs
3. Evaluates results by running the task's unit tests
4. Returns `ExecutionTrace` objects containing the agent's ReAct trace and test results

Server lifecycle is fully automatic — servers start on demand and shut down when the adapter is garbage collected.

Supports **OpenAI**, **Anthropic**, and **Gemini** models. The adapter auto-detects the provider from the model name and dispatches to the appropriate SDK.

## Usage

### Smoke test (single task)

```bash
python -m benchmarks.appworld.scripts.run_baseline \
  --model openai/gpt-5.4-nano \
  --limit 1
```

### Training

```bash
python -m scripts.run_training \
  --benchmark appworld \
  --model openai/gpt-5.4-nano \
  --batch-size 10 --n-concurrent 10 --iterations 30 --seed 42
```

### Evaluation

```bash
python -m scripts.run_eval \
  --benchmark appworld \
  --model openai/gpt-5.4-nano \
  --playbook results/training/<run_dir>/rcl_best.json \
  --output results/evals/aw_eval \
  --split test_normal \
  --n-concurrent 20
```

Available eval splits: `test_normal` (168 tasks), `test_challenge` (417 tasks).

## Files

```
appworld/
  benchmark.py              # BenchmarkConfig (sections, domain description)
  evaluator.py              # Evaluator (aggregates per-task pass percentages)
  appworld_server.py        # HTTP server wrapping AppWorld's task environment
  adapters/
    system_adapter.py       # SystemAdapter (manages server pool, runs agent)
    appworld_client.py      # HTTP client for the AppWorld server
  playbooks/
    seed_playbook.json      # Starting playbook
  scripts/
    run_baseline.py         # Quick single-task smoke test
```

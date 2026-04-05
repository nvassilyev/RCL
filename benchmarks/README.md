# Adding a New Benchmark

This guide explains how to integrate a new benchmark into RCL. The framework treats your benchmark as a black box — it doesn't care how your agent runs, what tools it uses, or how it talks to external services. It only needs three things from you:

1. **A way to run tasks and get back traces** (`SystemAdapter`)
2. **A way to score those traces** (`Evaluator`)
3. **A description of your benchmark domain** (`BenchmarkConfig`)

The existing benchmarks (`appworld/` and `browsecomp/`) are good references.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RCL Optimizer Loop                        │
│                                                             │
│  for each iteration:                                        │
│    1. Sample batch of task_ids from train pool               │
│    2. adapter.execute(task_ids, playbook) → traces           │
│    3. evaluator.evaluate(traces) → scores                    │
│    4. reflector.reflect(failed_traces, playbook) → insights  │
│    5. mutator.mutate(playbook, insights) → updated playbook  │
└─────────────────────────────────────────────────────────────┘
         │                                       ▲
         │ You implement steps 2-3               │ Framework handles 4-5
         ▼                                       │
┌─────────────────────┐                ┌─────────────────────┐
│   SystemAdapter     │                │ Reflector + Mutator  │
│   (your code)       │                │ (framework code)     │
│                     │                │                      │
│ Runs agent on tasks │                │ Reads traces to      │
│ Returns traces with │───────────────→│ propose playbook     │
│ agent work + eval   │  ExecutionTrace│ improvements         │
│ feedback            │                │                      │
└─────────────────────┘                └──────────────────────┘
```

The reflector and mutator are benchmark-agnostic — they read traces and evaluation feedback as strings. Your job is to produce those strings.

## Step-by-step

### 1. Create the benchmark directory

```
benchmarks/
  my_benchmark/
    __init__.py
    benchmark.py          # BenchmarkConfig
    evaluator.py          # Evaluator
    adapters/
      __init__.py
      system_adapter.py   # SystemAdapter
    playbooks/
      seed_playbook.json  # Starting playbook (can be empty)
```

### 2. Define `BenchmarkConfig` (`benchmark.py`)

This tells the reflector/mutator about your domain and what playbook sections to use.

```python
from rcl.core.interfaces import BenchmarkConfig

MY_SECTIONS = {
    "strategies":      "High-level approaches for solving tasks",
    "common_mistakes": "Known failure patterns and how to avoid them",
    "others":          "Miscellaneous guidance",
}

MY_CONFIG = BenchmarkConfig(
    name="my_benchmark",
    sections=MY_SECTIONS,
    domain_description=(
        "The agent is an AI assistant that ... "
        "It uses tools to ... "
        "Success is measured by ..."
    ),
)
```

**`sections`**: Categories for playbook entries. The mutator will assign new entries to these sections. Pick 3-6 sections that make sense for your domain.

**`domain_description`**: A paragraph explaining what the agent does, what tools it has, and how success is measured. The reflector uses this to reason about failures.

### 3. Implement `SystemAdapter` (`adapters/system_adapter.py`)

This is the core integration point. It runs your agent on tasks and returns `ExecutionTrace` objects.

```python
from typing import List, Optional
from rcl.core.data_structures import ExecutionTrace, Playbook
from rcl.core.interfaces import SystemAdapter

class MySystemAdapter(SystemAdapter):
    def execute(
        self,
        task_ids: List[str],
        playbook: Playbook,
        **kwargs,
    ) -> List[ExecutionTrace]:
        """Run the agent on each task and return traces."""
        traces = []
        for task_id in task_ids:
            # 1. Build the prompt: your instructions + playbook
            system_prompt = YOUR_BASE_INSTRUCTIONS
            if len(playbook) > 0:
                system_prompt += "\n\n" + playbook.to_prompt()

            # 2. Run your agent however you want
            #    (API calls, tool loops, subprocess, HTTP, etc.)
            result = run_my_agent(task_id, system_prompt)

            # 3. Evaluate the result
            score, eval_feedback = evaluate_result(task_id, result)

            # 4. Build the trace
            traces.append(ExecutionTrace(
                task_id=task_id,
                input_query=result.task_description,
                system_output=result.answer,
                trace=result.agent_trace,          # see below
                metadata={
                    "pass_pct": score,              # float 0.0-1.0
                    "task_completed": score > 0.99,  # bool
                    "evaluation_details": eval_feedback,  # see below
                },
            ))
        return traces

    def get_ground_truth(self, task_id: str) -> Optional[str]:
        """Return ground truth for a task (used by some RCL variants)."""
        return load_ground_truth(task_id)
```

#### The two key strings

The reflector reads two strings from each trace to understand what happened and why it failed. Getting these right is the most important part of the integration.

**`trace.trace`** — The agent's work log. What the agent saw and did.

Include:
- The task/question
- The agent's actions (tool calls, code execution, search queries)
- Tool outputs / observations
- The agent's reasoning (if available)
- The agent's final answer

Do NOT include:
- Ground truth answers
- Evaluation results (pass/fail, scores)
- Post-hoc judgments

The reflector needs to see the agent's *process* to diagnose what went wrong. If you include the evaluation results here, the reflector can't distinguish what the agent knew during execution vs. what was determined after.

**`trace.metadata["evaluation_details"]`** — Feedback from your evaluator. What went wrong.

Include:
- Whether the task passed or failed
- Scores, verdicts, or pass percentages
- Test reports (which tests passed/failed and why)
- Correct answer vs. agent's answer
- Any domain-specific feedback

This is where you put everything the agent *didn't* see during execution but that helps explain the outcome. The reflector reads both strings together: the trace shows what the agent did, and evaluation_details shows why it was wrong.

**Examples from existing benchmarks:**

AppWorld `trace.trace`:
```
## Task
Send $50 to John via Venmo

Step 1:
```python
print(apis.supervisor.show_account_passwords())
```
Output: {'venmo': {'username': 'alice', 'password': 'pass123'}}

Step 2:
...
```

AppWorld `evaluation_details`:
```
Pass percentage: 66.7%
Task completed: False

Total: 3 tests, 2 passed, 1 failed

FAILED TESTS:
  [FAIL] Payment amount should be $50.00
         Expected 50.00, got 25.00
PASSED TESTS:
  [PASS] Payment recipient should be John
  [PASS] Payment should be from Alice's account
```

BrowseComp+ `trace.trace`:
```
## Query
Who was the first person to climb K2 in winter?

[Step 1] SEARCH: "K2 winter ascent first"
  → [doc_42] "K2 Winter Expedition History"
  → [doc_87] "Mountaineering Records"

[Thinking] The search results mention...

[Final Answer]
Exact Answer: Nirmal Purja
```

BrowseComp+ `evaluation_details`:
```
Verdict: INCORRECT
Gold answer: The first winter ascent was not completed until 2021
Extracted answer: Nirmal Purja

Judge reasoning: The agent identified a climber but the gold answer
describes the event differently...

Retrieval: 1/3 gold docs found
Missed gold documents:
  [doc_201] "2021 K2 Winter Summit"
    On January 16, 2021, a team of Nepali climbers...
```

### 4. Implement `Evaluator` (`evaluator.py`)

Scores a batch of traces. The optimizer uses this to track progress.

```python
from typing import List
from rcl.core.data_structures import EvaluationResult, ExecutionTrace
from rcl.core.interfaces import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(self, traces: List[ExecutionTrace]) -> EvaluationResult:
        per_scores = [t.metadata.get("pass_pct", 0.0) for t in traces]
        n_pass = sum(1 for s in per_scores if s > 0.99)
        return EvaluationResult(
            score=sum(per_scores) / len(per_scores),  # average score
            tgc=n_pass / len(traces),                  # task completion rate
            per_instance_scores=per_scores,
            traces=traces,
        )
```

The evaluator is usually simple because the real evaluation logic lives in your `SystemAdapter` (which sets `pass_pct` and `evaluation_details` on each trace). The `Evaluator` just aggregates.

### 5. Create a seed playbook

Create `playbooks/seed_playbook.json`:

```json
{
  "entries": [],
  "version": "1.0"
}
```

An empty playbook is fine — RCL will build it up from scratch. If you have known tips for your domain, you can seed them:

```json
{
  "entries": [
    {"content": "Always check pagination — most list APIs return only the first page", "section": "strategies"},
    {"content": "Use ISO 8601 format for all dates", "section": "common_mistakes"}
  ],
  "version": "1.0"
}
```

### 6. Wire it into the training script

Add your benchmark to `scripts/run_training.py`:

```python
def build_my_benchmark(args):
    from benchmarks.my_benchmark.benchmark import MY_CONFIG
    from benchmarks.my_benchmark.adapters.system_adapter import MySystemAdapter
    from benchmarks.my_benchmark.evaluator import MyEvaluator

    trace_writer = TraceWriter(args.output)
    adapter = MySystemAdapter(model=args.model, trace_writer=trace_writer)
    evaluator = MyEvaluator()
    seed_playbook = Playbook.load("benchmarks/my_benchmark/playbooks/seed_playbook.json")
    train_ids = adapter.load_tasks(limit=args.train_pool)

    return adapter, evaluator, MY_CONFIG, seed_playbook, train_ids, trace_writer
```

Then add `"my_benchmark"` to the `--benchmark` choices and add a branch in `main()`:

```python
parser.add_argument("--benchmark", choices=["appworld", "browsecomp", "my_benchmark"])
...
elif args.benchmark == "my_benchmark":
    adapter, evaluator, bench_config, seed_playbook, train_ids, trace_writer = build_my_benchmark(args)
```

### 7. Run it

```bash
python -m scripts.run_training \
  --benchmark my_benchmark \
  --model gemini-3.1-flash-lite-preview \
  --batch-size 10 --iterations 30
```

## Key design decisions

**The playbook is injected into the system prompt.** Your adapter calls `playbook.to_prompt()` which renders entries as markdown, grouped by section. You prepend your base instructions and append the playbook. The agent sees it as part of its instructions.

**Only failed traces produce learning signal.** By default, RCL samples failed traces (where `pass_pct < 1.0`) for reflection. Successes are skipped because there's nothing to improve. Make sure your evaluator produces a range of scores, not just 0/1, if partial credit is meaningful.

**The framework doesn't care about your inference engine.** Sync, async, subprocess, HTTP — it's all fine. The `execute()` method is blocking from the optimizer's perspective. Handle your own concurrency internally.

**Infrastructure errors should be marked.** If a task fails due to server crashes or network issues (not agent mistakes), set `trace.metadata["infra_error"] = True`. The evaluator will exclude these from scoring so they don't pollute the signal.

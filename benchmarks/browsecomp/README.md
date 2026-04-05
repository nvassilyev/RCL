# BrowseComp+

Benchmark adapter for [BrowseComp+](https://github.com/texttron/BrowseComp-Plus) — a web research benchmark where an AI agent answers hard factual questions by searching a document corpus via an MCP search server.

## Metrics

- **Accuracy**: Fraction of queries judged correct by an LLM judge comparing the agent's answer to the gold answer.
- **Retrieval recall**: Fraction of gold evidence documents retrieved during search (diagnostic, not primary).

## Prerequisites

BrowseComp+ requires a FAISS-based MCP search server running on a GPU node (for embedding-based retrieval).

### 1. Clone and prepare the dataset

```bash
git clone https://github.com/texttron/BrowseComp-Plus.git
cd BrowseComp-Plus
python scripts_build_index/decrypt_dataset.py
```

This produces `data/browsecomp_plus_decrypted.jsonl` (830 queries).

### 2. Build the search index

Follow the [BrowseComp-Plus repo](https://github.com/texttron/BrowseComp-Plus) instructions to build a FAISS index with Qwen3-Embedding-8B (requires 2x L4 or equivalent GPUs).

### 3. Start the MCP search server

The search server needs to run on a machine with GPUs:

```bash
cd /path/to/BrowseComp-Plus
python searcher/mcp_server.py \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize --port 8081 --transport streamable-http
```

If the MCP server runs on a different machine from the training script, you'll need the server to bind to `0.0.0.0` instead of localhost. The upstream repo doesn't expose a `--host` flag yet, so either patch `mcp_server.py` to add one or use a tunnel.

To verify the server is running:

```bash
# Should return 406 (healthy — bare GET not allowed on MCP endpoint)
curl -s -o /dev/null -w "%{http_code}" http://<gpu-host>:8081/mcp/
```

### 4. Set environment variables

```bash
export BROWSECOMP_PLUS_DIR=/path/to/BrowseComp-Plus
export BROWSECOMP_MCP_URL=http://<gpu-host>:8081/mcp/   # default: http://localhost:8081/mcp/
```

And set the API key for your model provider (see [main README](../../README.md#supported-models)).

## How It Works

The `BrowseCompSystemAdapter` runs each query through:

1. **Inference**: The agent uses a manual tool-calling loop with the MCP search server. It can call `search(query)` to find documents and `get_document(docid)` to read them, iterating until it produces a final answer.
2. **Judging**: An LLM judge compares the agent's extracted answer to the gold answer and determines correctness.
3. **Trace building**: The adapter constructs a clean agent trace (search steps, reasoning, final answer) and evaluation details (verdict, gold answer, retrieval recall, missed documents).

Supports **OpenAI**, **Anthropic**, and **Gemini** models for the agent. The provider is specified via the `provider/model-name` format (e.g. `openai/gpt-5.4-nano`).

The training splits are pre-defined in `splits/`:
- `train_100.json`: 100 training queries
- `val_30.json`: 30 validation queries
- `test_150.json`: 150 test queries

## Usage

### Smoke test (single query)

```bash
python -m benchmarks.browsecomp.scripts.run_baseline \
  --model openai/gpt-5.4-nano \
  --limit 1
```

### Training

```bash
python -m scripts.run_training \
  --benchmark browsecomp \
  --model openai/gpt-5.4-nano \
  --batch-size 5 --n-concurrent 18 --iterations 30 --seed 42
```

### Evaluation

```bash
python -m scripts.run_eval \
  --benchmark browsecomp \
  --model openai/gpt-5.4-nano \
  --playbook results/training/<run_dir>/rcl_best.json \
  --output results/evals/bc_eval \
  --n-concurrent 18
```

## Files

```
browsecomp/
  benchmark.py              # BenchmarkConfig (sections, domain description)
  evaluator.py              # Evaluator (aggregates correctness across queries)
  adapters/
    system_adapter.py       # SystemAdapter (multi-provider inference + MCP + judge)
    browsecomp_client.py    # Dataset loading, judging, query result types
    gemini_inference.py     # Gemini-specific inference helpers (legacy, used by client)
  playbooks/
    empty_playbook.json     # Empty starting playbook
  splits/
    train_100.json          # Training query IDs
    val_30.json             # Validation query IDs
    test_150.json           # Test query IDs
  scripts/
    run_baseline.py         # Quick single-query smoke test
```

"""BrowseComp+ client wrapping Gemini inference + MCP search + LLM judge.

Provides a synchronous interface for running inference on BrowseComp+ queries
with an injectable system prompt (for playbook injection).

Uses pure async with asyncio.gather + semaphore for concurrency.

Requires:
    pip install google-genai 'fastmcp==2.9.2' python-dotenv
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from .gemini_inference import run_manual_async as _run_manual_async_shared, AFCResult

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BROWSECOMP_DATA_DIR = Path(os.environ.get("BROWSECOMP_PLUS_DIR", REPO_ROOT / "BrowseComp-Plus"))
DEFAULT_DATASET = _BROWSECOMP_DATA_DIR / "data" / "browsecomp_plus_decrypted.jsonl"
DEFAULT_QREL = _BROWSECOMP_DATA_DIR / "topics-qrels" / "qrel_evidence.txt"

# Default query template — the {system_prompt} placeholder is prepended when a playbook is provided
BASE_QUERY_TEMPLATE = """\
You are a deep research agent. You need to answer the given question by \
interacting with a search engine, using the search tool provided. Please \
perform reasoning and use the tool step by step, in an interleaved manner. \
You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation \
section only, you should cite your evidence documents inline by enclosing \
their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

PLAYBOOK_QUERY_ADDENDUM = """\

IMPORTANT: You have been given a Playbook in your system instructions. \
Read it carefully and actively apply its guidance throughout your search and reasoning process."""

GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


@dataclass
class BrowseCompConfig:
    """Configuration for BrowseComp+ evaluation."""

    model: str = "google/gemini-3-flash-preview"
    judge_model: str = "anthropic/claude-sonnet-4-6"
    max_tokens: int = 65536
    dataset: str = str(DEFAULT_DATASET)
    qrel: str = str(DEFAULT_QREL)
    mcp_url: str = os.environ.get("BROWSECOMP_MCP_URL", "http://localhost:8081/mcp/")
    snippet_max_chars: int = 3000
    n_concurrent: int = 16
    max_steps: int = 50
    query_timeout: float = 0  # per-query timeout in seconds (0 = no timeout)
    thinking_level: str = "HIGH"  # thinking level for AFC (None to disable)
    afc_retries: int = 0  # number of AFC retries (0 = no retries, like official impl)


@dataclass
class QueryResult:
    """Result from a single BrowseComp+ query."""

    query_id: str
    query: str
    gold_answer: str
    model_response: str = ""
    correct: Optional[bool] = None
    judge_response: str = ""
    extracted_answer: Optional[str] = None
    confidence: Optional[float] = None
    retrieval_recall: Optional[float] = None
    retrieved_docids: List[str] = field(default_factory=list)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    full_trace: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None
    duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str, num_examples: Optional[int] = None) -> List[Dict]:
    """Load browsecomp+ JSONL and return list of {query_id, query, answer}."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            examples.append({
                "query_id": str(obj["query_id"]),
                "query": obj["query"],
                "answer": obj["answer"],
                "evidence_docs": obj.get("evidence_docs", []),
            })
            if num_examples and len(examples) >= num_examples:
                break
    return examples


def load_qrel(path: str) -> Dict[str, List[str]]:
    """Load TREC-format qrel -> {query_id: [docid, ...]}."""
    qrel: Dict[str, List[str]] = defaultdict(list)
    p = Path(path)
    if not p.exists():
        return dict(qrel)
    with p.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qrel[parts[0]].append(parts[2])
    return dict(qrel)


# ---------------------------------------------------------------------------
# Internal helpers (from eval script)
# ---------------------------------------------------------------------------

def _extract_retrieved_docids(results: List[Dict]) -> List[str]:
    docids: set = set()
    for item in results:
        if item.get("type") != "tool_call":
            continue
        output = item.get("output")
        parsed = None
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except Exception:
                pass
        elif isinstance(output, list):
            parsed = output
        if isinstance(parsed, list):
            for elem in parsed:
                if isinstance(elem, dict) and "docid" in elem:
                    docids.add(str(elem["docid"]))
        elif isinstance(output, str):
            for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', output):
                docids.add(str(m))
    return sorted(docids)


def _parse_judge_response(text: str) -> Dict:
    result = {"extracted_final_answer": None, "reasoning": None, "correct": None, "confidence": None, "parse_error": False}
    if not text:
        result["parse_error"] = True
        return result

    for pat in [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            result["extracted_final_answer"] = m.group(1).strip()
            break

    for pat in [
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            result["reasoning"] = m.group(1).strip()
            break

    for pat in [r"\*\*correct:\*\*\s*(yes|no)", r"\*\*correct\*\*:\s*(yes|no)", r"correct:\s*(yes|no)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["correct"] = m.group(1).lower() == "yes"
            break

    for pat in [r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)", r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)", r"confidence:\s*(\d+(?:\.\d+)?)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["confidence"] = min(float(m.group(1)), 100.0)
            break

    if result["correct"] is None:
        result["parse_error"] = True
    return result


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

async def _judge_answer(
    judge_model: str,
    question: str,
    response_text: str,
    correct_answer: str,
) -> Dict:
    """Use an LLM judge to grade a response via create_generate_fn."""
    from rcl.components.llm_client import create_generate_fn

    prompt = GRADER_TEMPLATE.format(
        question=question,
        response=response_text,
        correct_answer=correct_answer,
    )

    judge_fn = create_generate_fn(judge_model, max_output_tokens=2048)
    judge_text = await asyncio.get_event_loop().run_in_executor(None, judge_fn, prompt)

    return {
        "judge_response": judge_text,
        **_parse_judge_response(judge_text),
    }


# ---------------------------------------------------------------------------
# BrowseCompClient — synchronous facade
# ---------------------------------------------------------------------------

class BrowseCompClient:
    """Client for running BrowseComp+ queries with Gemini + MCP search.

    Connects to a remote MCP search server (e.g. FAISS/Qwen3-Embedding)
    for document retrieval during agent execution.

    Usage:
        client = BrowseCompClient(BrowseCompConfig())
        results = client.run_queries(
            queries=[{"query_id": "1", "query": "...", "answer": "..."}],
            system_prompt="## Playbook\\n..."
        )
    """

    def __init__(self, config: BrowseCompConfig):
        self.config = config

        # Clean up bad GOOGLE_APPLICATION_CREDENTIALS
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if gac and not Path(gac).exists():
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    def load_dataset(self, limit: Optional[int] = None) -> List[Dict]:
        """Load queries from the dataset JSONL."""
        return load_dataset(self.config.dataset, num_examples=limit)

    def load_qrel(self) -> Dict[str, List[str]]:
        """Load TREC-format qrel for retrieval recall."""
        return load_qrel(self.config.qrel)

    def run_queries(
        self,
        queries: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> List[QueryResult]:
        """Run inference + judge on a batch of queries.

        Uses pure async with asyncio.gather + semaphore for concurrency.

        Args:
            queries: List of {query_id, query, answer} dicts.
            system_prompt: Optional system prompt (playbook) to prepend.

        Returns:
            List of QueryResult objects (order matches input).
        """
        return asyncio.run(self._run_queries_async(queries, system_prompt))

    async def _run_queries_async(
        self,
        queries: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> List[QueryResult]:
        """Async implementation of run_queries.

        Uses a shared Gemini client and semaphore for concurrency control.
        Each query gets its own MCP client session (required by AFC).
        """
        mcp_server = self.config.mcp_url

        # Shared Gemini client for all queries
        from rcl.components.llm_client import _create_genai_client
        gemini_client = _create_genai_client(genai, self.config.model)

        sem = asyncio.Semaphore(self.config.n_concurrent)
        completed_count = 0
        total = len(queries)

        async def run_one(idx: int, example: Dict) -> tuple:
            nonlocal completed_count
            async with sem:
                try:
                    result = await self._process_single_query_shared(
                        gemini_client, mcp_server, example, system_prompt
                    )
                except Exception as e:
                    logger.error("Worker error for %s: %s", example["query_id"], e)
                    result = QueryResult(
                        query_id=example["query_id"],
                        query=example["query"],
                        gold_answer=example["answer"],
                        status="error",
                        error=str(e),
                        correct=False,
                    )
                completed_count += 1
                print(
                    f"  [{completed_count}/{total}] {result.query_id} "
                    f"correct={result.correct} {result.duration_sec:.1f}s",
                    flush=True,
                )
                return idx, result

        tasks = [run_one(i, q) for i, q in enumerate(queries)]
        pair_results = await asyncio.gather(*tasks)

        results: List[QueryResult] = [None] * total
        for idx, result in pair_results:
            results[idx] = result
        return results

    async def _process_single_query_shared(
        self,
        gemini_client,
        mcp_server,
        example: Dict,
        system_prompt: Optional[str],
    ) -> QueryResult:
        """Process a single query using a shared Gemini client.

        Each query still gets its own MCP session (required by AFC).
        """
        from fastmcp import Client

        qid = example["query_id"]
        mcp_client = Client(mcp_server)
        timeout = self.config.query_timeout

        async with mcp_client:
            try:
                if timeout and timeout > 0:
                    return await asyncio.wait_for(
                        self._process_query(
                            gemini_client, mcp_client, example, system_prompt
                        ),
                        timeout=timeout,
                    )
                else:
                    return await self._process_query(
                        gemini_client, mcp_client, example, system_prompt
                    )
            except asyncio.TimeoutError:
                logger.warning("Query %s timed out after %.0fs", qid, timeout)
                return QueryResult(
                    query_id=qid,
                    query=example["query"],
                    gold_answer=example["answer"],
                    status="error",
                    error=f"Timeout after {timeout}s",
                    correct=False,
                )
            except Exception as e:
                logger.error("Query %s failed: %s", qid, e)
                return QueryResult(
                    query_id=qid,
                    query=example["query"],
                    gold_answer=example["answer"],
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    correct=False,
                )

    async def _process_query(
        self,
        gemini_client,
        mcp_client,
        example: Dict,
        system_prompt: Optional[str],
    ) -> QueryResult:
        """Process a single query: inference + judge.

        Uses the shared manual tool loop via MCP, with per-step timeouts
        and partial-progress recovery.
        """
        qid = example["query_id"]
        query = example["query"]
        answer = example["answer"]

        user_content = BASE_QUERY_TEMPLATE.format(Question=query)
        if system_prompt:
            user_content += PLAYBOOK_QUERY_ADDENDUM

        start = time.time()
        try:
            step_timeout = self.config.query_timeout if self.config.query_timeout > 0 else 120
            retry_count = max(1, self.config.afc_retries) if self.config.afc_retries else 5
            afc_result = await _run_manual_async_shared(
                gemini_client, self.config.model, user_content,
                mcp_client=mcp_client,
                system_prompt=system_prompt,
                max_steps=self.config.max_steps,
                max_output_tokens=self.config.max_tokens,
                step_timeout=step_timeout,
                max_retries=retry_count,
                thinking_level=self.config.thinking_level,
            )
        except Exception as exc:
            logger.warning("Inference error for %s: %s", qid, exc)
            return QueryResult(
                query_id=qid, query=query, gold_answer=answer,
                status="error", error=str(exc), correct=False,
                duration_sec=time.time() - start,
            )

        # Convert AFCResult to the dict format expected by downstream code
        inf = self._afc_result_to_dict(afc_result)

        # Extract final text
        final_text = afc_result.final_text or ""

        # Judge
        try:
            judge = await _judge_answer(
                self.config.judge_model,
                query, final_text, answer,
            )
        except Exception as exc:
            logger.warning("Judge error for %s: %s", qid, exc)
            judge = {"correct": None, "parse_error": True, "judge_response": str(exc)}

        elapsed = time.time() - start

        return QueryResult(
            query_id=qid,
            query=query,
            gold_answer=answer,
            model_response=final_text,
            correct=judge.get("correct"),
            judge_response=judge.get("judge_response", ""),
            extracted_answer=judge.get("extracted_final_answer"),
            confidence=judge.get("confidence"),
            retrieved_docids=inf.get("retrieved_docids", []),
            tool_call_counts=inf.get("tool_call_counts", {}),
            usage=inf.get("usage", {}),
            full_trace=inf.get("result", []),
            status=inf.get("status", "unknown"),
            error=afc_result.error,
            duration_sec=elapsed,
        )

    @staticmethod
    def _afc_result_to_dict(afc_result: AFCResult) -> Dict:
        """Convert AFCResult to the legacy dict format for downstream compatibility."""
        tool_counts: Dict[str, int] = {}
        for tc in afc_result.tool_calls:
            if tc.get("type") == "tool_call":
                name = tc.get("tool_name", "")
                tool_counts[name] = tool_counts.get(name, 0) + 1

        status = "error" if afc_result.error else "completed"

        return {
            "tool_call_counts": tool_counts,
            "status": status,
            "retrieved_docids": _extract_retrieved_docids(afc_result.tool_calls),
            "result": afc_result.tool_calls,
            "usage": afc_result.usage,
        }

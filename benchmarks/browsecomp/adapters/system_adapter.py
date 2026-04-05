"""SystemAdapter for BrowseComp+ via manual MCP tool-calling loop."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from rcl.core.data_structures import ExecutionTrace, Playbook
from rcl.core.interfaces import SystemAdapter
from rcl.core.trace_writer import (
    TraceWriter,
    build_rollout_descriptors,
    rollout_metadata,
)
from rcl.components.inference import (
    AFCResult,
    create_gemini_client,
    run_inference,
    _parse_provider,
)

from .browsecomp_client import BrowseCompConfig, load_dataset, load_qrel

logger = logging.getLogger(__name__)

# Default data paths (relative to browsecomp_client.py SCRIPT_DIR)
from .browsecomp_client import DEFAULT_DATASET, DEFAULT_QREL

SYSTEM_INSTRUCTIONS = """\
You are a deep research agent. You need to answer the given question by \
interacting with a search engine, using the search tool provided. Please \
perform reasoning and use the tool step by step, in an interleaved manner. \
You may use the search tool multiple times.

Your response should be in the following format:
Explanation: {{your explanation for your final answer. Cite evidence documents \
inline by enclosing their docids in square brackets [].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

PLAYBOOK_HEADER = """

# Playbook

You have been given a curated playbook of strategies, common mistakes, and proven solutions. \
Read it carefully and actively apply its guidance throughout your search and reasoning.

"""


def _build_system_prompt(playbook: Playbook) -> str:
    """Build system prompt: instructions + playbook."""
    parts = [SYSTEM_INSTRUCTIONS]
    if len(playbook) > 0:
        parts.append(PLAYBOOK_HEADER + playbook.to_prompt())
    return "\n".join(parts)


# Judge prompt
JUDGE_PROMPT = """You are a judge evaluating whether an extracted answer matches a correct answer.

Correct answer: {gold_answer}
Extracted answer: {extracted_answer}

Respond in this exact format:
extracted_final_answer: <the answer being evaluated>
reasoning: <brief comparison>
correct: <yes or no>
confidence: <0-100>"""



def _extract_title(text: str) -> str:
    """Extract title from doc text, handling frontmatter."""
    if text.startswith("---"):
        for line in text.split("\n")[1:]:
            if line.startswith("title:"):
                return line[6:].strip()
            if line.strip() == "---":
                break
    for line in text.split("\n"):
        line = line.strip()
        if line and line != "---":
            return line[:100]
    return "(untitled)"


class BrowseCompSystemAdapter(SystemAdapter):
    """System adapter for BrowseComp+ using unified AFC inference."""

    def __init__(
        self,
        config: BrowseCompConfig,
        trace_writer: Optional[TraceWriter] = None,
    ):
        self.config = config
        self.trace_writer = trace_writer
        provider, _ = _parse_provider(config.model)
        self._gemini = create_gemini_client() if provider == "google" else None
        self._dataset_cache = None
        self._qrel_cache = None
        self._job_counter = 0

    def execute(
        self,
        task_ids: List[str],
        playbook: Playbook,
        experiment_prefix: str = "eval",
        max_workers: int = 1,
        verbose: bool = True,
        trace_subdir: str = "traces",
        playbook_overrides: Optional[Dict[str, "Playbook"]] = None,
    ) -> List[ExecutionTrace]:
        self._job_counter += 1

        # Build system prompt from playbook (with per-task overrides for PP)
        system_prompt = _build_system_prompt(playbook)
        _overrides = playbook_overrides or {}
        system_prompts = {tid: _build_system_prompt(_overrides[tid]) for tid in _overrides}
        # Default prompt for tasks without overrides
        for tid in task_ids:
            if tid not in system_prompts:
                system_prompts[tid] = system_prompt

        dataset = self._get_dataset()
        query_map = {ex["query_id"]: ex for ex in dataset}
        # Strip __pp__ prefix for dataset lookup (optimizer adds it for playbook_overrides routing)
        pp_prefix = "__pp__"
        rollout_descriptors = build_rollout_descriptors(task_ids)
        query_specs = []
        for tid, rollout in zip(task_ids, rollout_descriptors):
            raw_tid = tid[len(pp_prefix):] if tid.startswith(pp_prefix) else tid
            query_specs.append(
                {
                    "task_id": tid,
                    "raw_task_id": raw_tid,
                    "query_data": query_map.get(raw_tid),
                    "system_prompt": system_prompts.get(tid, system_prompt),
                    "rollout": rollout,
                    "result": None,
                }
            )
        runnable_specs = [spec for spec in query_specs if spec["query_data"] is not None]

        if verbose:
            print(
                f"    Running {len(runnable_specs)} queries via AFC ({experiment_prefix}_{self._job_counter})",
                flush=True,
            )

        batch_start = time.time()
        results = asyncio.run(self._run_batch(runnable_specs))
        batch_elapsed = time.time() - batch_start
        for spec, result in zip(runnable_specs, results):
            spec["result"] = result

        qrel = self._get_qrel()
        evidence_docs_map = {ex["query_id"]: ex.get("evidence_docs", []) for ex in dataset}

        traces = []
        for spec in query_specs:
            tid = spec["task_id"]
            raw_tid = spec["raw_task_id"]
            afc_result = spec["result"]
            query_data = spec["query_data"] or {}
            rollout = spec["rollout"]
            if afc_result and (not afc_result.error or afc_result.n_tool_calls > 0):
                # Run judge even on partial results — if the agent did work, evaluate it
                trace = self._build_trace(
                    tid,
                    raw_tid,
                    query_data,
                    afc_result,
                    qrel,
                    evidence_docs_map.get(raw_tid, []),
                )
            else:
                error = afc_result.error if afc_result else "missing_result"
                is_timeout = error == "timeout"
                trace = ExecutionTrace(
                    task_id=tid, input_query=query_data.get("question", tid),
                    system_output=None, trace=f"Error: {error}",
                    metadata={"pass_pct": 0.0, "task_completed": False, "error": error,
                              "timed_out": is_timeout,
                              "evaluation_details": f"Error: {error}"},
                )
            trace.metadata.update(rollout_metadata(rollout))
            traces.append(trace)

            if verbose:
                status = "PASS" if trace.metadata.get("task_completed") else "FAIL"
                print(f"    [{tid}] correct={trace.metadata.get('correct')} {status}", flush=True)

            if self.trace_writer and afc_result:
                self.trace_writer.write_trace(
                    tid,
                    {
                        "task_id": tid,
                        "query": query_data.get("query", ""),
                        "gold_answer": query_data.get("answer", ""),
                        "result": {
                            "correct": trace.metadata.get("correct"),
                            "extracted_answer": trace.metadata.get("extracted_answer"),
                            "retrieval_recall": trace.metadata.get("retrieval_recall"),
                            "n_tool_calls": afc_result.n_tool_calls,
                            "duration_sec": afc_result.duration_sec,
                        },
                        "judge_response": trace.metadata.get("judge_response"),
                        "afc_trace": afc_result.tool_calls,
                        "final_text": afc_result.final_text,
                        "usage": afc_result.usage,
                        "rollout": rollout_metadata(rollout),
                    },
                    subdir=trace_subdir,
                    artifact_id=str(rollout["artifact_id"]),
                )

        if verbose and traces:
            correct_count = sum(1 for t in traces if t.metadata.get("task_completed"))
            print(f"    Batch: {len(traces)} in {batch_elapsed:.1f}s | accuracy={correct_count}/{len(traces)}", flush=True)

        return traces

    async def _run_batch(self, query_specs):
        """Run queries concurrently via manual tool loop."""
        from fastmcp import Client as MCPClient

        sem = asyncio.Semaphore(self.config.n_concurrent)
        query_timeout = self.config.query_timeout if self.config.query_timeout > 0 else 900
        completed = 0

        async def _run_one(spec):
            nonlocal completed
            tid = spec["task_id"]
            query_data = spec["query_data"]
            async with sem:
                try:
                    mcp = MCPClient(self.config.mcp_url)
                    async with mcp:
                        question = query_data.get("query", "")
                        cancelled = asyncio.Event()
                        task_prompt = spec["system_prompt"]

                        try:
                            inference_coro = run_inference(
                                model=self.config.model,
                                prompt=question,
                                mcp_client=mcp,
                                system_prompt=task_prompt,
                                max_steps=self.config.max_steps,
                                max_output_tokens=self.config.max_tokens,
                                step_timeout=120,
                                thinking_level=self.config.thinking_level,
                                cancelled=cancelled,
                                gemini_client=self._gemini,
                            )
                            result = await asyncio.wait_for(
                                inference_coro,
                                timeout=query_timeout,
                            )
                        except asyncio.TimeoutError:
                            cancelled.set()
                            result = AFCResult(error="timeout", duration_sec=query_timeout)
                        completed += 1
                        print(
                            f"    [{tid}] done {completed}/{len(query_specs)} "
                            f"({result.n_tool_calls} tools, {result.duration_sec:.0f}s)",
                            flush=True,
                        )
                        return result
                except Exception as exc:
                    logger.warning("Query %s failed (MCP/infra): %s", tid, str(exc)[:200])
                    completed += 1
                    print(
                        f"    [{tid}] ERROR {completed}/{len(query_specs)}: {str(exc)[:100]}",
                        flush=True,
                    )
                    return AFCResult(error=f"infra_error: {exc}", duration_sec=0)

        return await asyncio.gather(*[_run_one(spec) for spec in query_specs])

    def _build_trace(self, tid, qrel_task_id, query_data, afc_result: AFCResult, qrel, evidence_docs):
        """Build ExecutionTrace with decoupled trace and evaluation_details."""
        question = query_data.get("query", "")
        gold_answer = query_data.get("answer", "")

        # Extract answer from final text
        extracted_answer = self._extract_answer(afc_result.final_text)

        # Judge
        correct, judge_response = self._judge(extracted_answer, gold_answer)
        pass_pct = 1.0 if correct else 0.0

        # Retrieval recall
        evidence_docids = qrel.get(qrel_task_id, [])
        retrieved_docids = [
            str((tc.get("arguments") or {}).get("docid", (tc.get("arguments") or {}).get("query", "")))
            for tc in afc_result.tool_calls
            if tc.get("type") == "tool_call" and tc.get("tool_name") == "get_document"
        ]
        # Also get docids from search results
        for tc in afc_result.tool_calls:
            if tc.get("type") == "tool_call" and tc.get("tool_name") == "search" and tc.get("output"):
                try:
                    search_results = json.loads(tc["output"])
                    if isinstance(search_results, list):
                        for r in search_results:
                            retrieved_docids.append(str(r.get("docid", "")))
                except (json.JSONDecodeError, TypeError):
                    pass

        gold_set = set(str(d) for d in evidence_docids)
        retrieved_set = set(retrieved_docids)
        found = sorted(gold_set & retrieved_set)
        missed = sorted(gold_set - retrieved_set)
        retrieval_recall = len(found) / len(gold_set) if gold_set else None

        # === Build clean trace ===
        trace_parts = [f"## Query\n{question}", ""]
        trace_parts.append(self._format_search_trace(afc_result, gold_set))
        trace_str = "\n".join(trace_parts)

        # === Build evaluation details ===
        eval_parts = [
            f"Verdict: {'CORRECT' if correct else 'INCORRECT'}",
            f"Gold answer: {gold_answer}",
            f"Extracted answer: {extracted_answer}",
        ]
        if judge_response:
            eval_parts.append(f"\nJudge reasoning:\n{judge_response}")
        if evidence_docids:
            eval_parts.append(f"\nRetrieval: {len(found)}/{len(evidence_docids)} gold docs found")
            if missed and evidence_docs:
                eval_parts.append("Missed gold documents:")
                missed_set = set(missed)
                for doc in evidence_docs:
                    if str(doc.get("docid", "")) in missed_set:
                        title = _extract_title(doc.get("text", ""))
                        body = doc.get("text", "")[:500]
                        eval_parts.append(f"  [{doc['docid']}] \"{title}\"\n    {body}")

        return ExecutionTrace(
            task_id=tid, input_query=question, system_output=correct,
            trace=trace_str,
            metadata={
                "pass_pct": pass_pct,
                "task_completed": correct is True,
                "correct": correct,
                "gold_answer": gold_answer,
                "extracted_answer": extracted_answer,
                "retrieval_recall": retrieval_recall,
                "judge_response": judge_response,
                "evaluation_details": "\n".join(eval_parts),
                "duration_s": afc_result.duration_sec,
                "n_tool_calls": afc_result.n_tool_calls,
                "usage": afc_result.usage,
            },
        )

    def _format_search_trace(self, afc_result: AFCResult, gold_docid_set: set) -> str:
        """Format AFC tool calls into readable search trace."""
        parts = []
        step = 0
        for tc in afc_result.tool_calls:
            if tc.get("type") == "reasoning":
                text = tc.get("output", "")
                if text:
                    parts.append(f"[Thinking] {text}")

            elif tc.get("type") == "tool_call":
                step += 1
                name = tc.get("tool_name", "")
                args = tc.get("arguments", {}) or {}
                output = tc.get("output", "") or ""

                if name == "search":
                    query_text = args.get("query", str(args))
                    parts.append(f"\n[Step {step}] SEARCH: \"{query_text}\"")
                    try:
                        results = json.loads(output) if isinstance(output, str) else output
                        if isinstance(results, list):
                            for r in results:
                                docid = str(r.get("docid", ""))
                                snippet = r.get("snippet", r.get("text", ""))
                                title = _extract_title(snippet) if snippet else "(no title)"
                                body = snippet[:300] if snippet else ""
                                gold = " ★ GOLD" if docid in gold_docid_set else ""
                                parts.append(f"  → [{docid}]{gold} \"{title}\"")
                                if body:
                                    parts.append(f"    {body}")
                    except (json.JSONDecodeError, TypeError):
                        if output:
                            parts.append(f"  → {output[:200]}")

                elif name == "get_document":
                    docid = args.get("docid", str(args))
                    gold = " ★ GOLD" if str(docid) in gold_docid_set else ""
                    parts.append(f"\n[Step {step}] GET_DOCUMENT: [{docid}]{gold}")

                else:
                    parts.append(f"\n[Step {step}] {name}: {json.dumps(args)[:200]}")

        # Append final answer
        if afc_result.final_text:
            parts.append(f"\n[Final Answer]\n{afc_result.final_text}")

        n_searches = sum(1 for tc in afc_result.tool_calls
                         if tc.get("type") == "tool_call" and tc.get("tool_name") == "search")
        return f"## Agent Search Trace ({n_searches} searches)\n" + "\n".join(parts)

    def _extract_answer(self, text: str) -> str:
        """Extract answer from model's final text."""
        import re
        # Try structured formats first (order: most explicit → least)
        for pattern in [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Exact Answer:\s*(.+?)(?:\n|Confidence:|$)",
            r"(?:^|\n)\s*The answer is\s+(.+?)\.?\s*(?:\n|$)",
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # Fallback: last non-metadata line
        lines = text.strip().split("\n")
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not re.match(
                r"^(Confidence|Note|Sources?|Citations?)\s*:", stripped, re.IGNORECASE
            ):
                return stripped
        return lines[-1].strip() if lines else ""

    def _judge(self, extracted: str, gold: str):
        """Judge correctness using Claude."""
        from rcl.components.llm_client import create_generate_fn
        judge_fn = create_generate_fn(self.config.judge_model)
        prompt = JUDGE_PROMPT.format(gold_answer=gold, extracted_answer=extracted)
        try:
            response = judge_fn(prompt)
            correct = "correct: yes" in response.lower()
            return correct, response
        except Exception as e:
            logger.warning("Judge failed: %s", e)
            return extracted.lower().strip() == gold.lower().strip(), f"Judge error: {e}"

    def get_ground_truth(self, task_id: str) -> Optional[str]:
        for ex in self._get_dataset():
            if ex["query_id"] == task_id:
                return ex.get("answer")
        return None

    def load_tasks(self, split: str = "train", limit: Optional[int] = None) -> List[str]:
        return [ex["query_id"] for ex in self._get_dataset(limit=limit)]

    def _get_dataset(self, limit=None):
        if self._dataset_cache is None:
            self._dataset_cache = load_dataset(self.config.dataset)
        return self._dataset_cache[:limit] if limit else self._dataset_cache

    def _get_qrel(self):
        if self._qrel_cache is None:
            self._qrel_cache = load_qrel(self.config.qrel)
        return self._qrel_cache

"""BrowseComp+ benchmark configuration.

The BrowseComp system adapter should populate ExecutionTrace with:

- trace.trace: Clean agent trace containing:
    - The question
    - Search steps (queries, retrieved doc titles/snippets with ★ GOLD markers)
    - Agent's reasoning between searches
    - Agent's final answer
    NOTE: No ground truth answer, verdict, or retrieval gap in the trace.

- trace.metadata["evaluation_details"]: Evaluation feedback string containing:
    - Verdict (CORRECT / INCORRECT)
    - Gold answer
    - Agent's extracted answer
    - Judge reasoning (if available)
    - Retrieval gap: which gold docs were found vs missed (with snippets of missed docs)

- trace.metadata["pass_pct"]: 1.0 if correct, 0.0 if not
- trace.metadata["task_completed"]: Bool
- trace.metadata["correct"]: Bool (judge result)
- trace.metadata["gold_answer"]: str
- trace.metadata["extracted_answer"]: str
- trace.metadata["retrieval_recall"]: float or None
- trace.metadata["judge_response"]: str (judge's reasoning)
"""

from rcl.core.interfaces import BenchmarkConfig

BROWSECOMP_SECTIONS = {
    "search_strategies": "High-level approaches for finding information",
    "query_formulation": "How to construct effective search queries",
    "answer_extraction": "How to extract and verify answers from search results",
    "common_mistakes": "Known failure patterns and how to avoid them",
}

BROWSECOMP_CONFIG = BenchmarkConfig(
    name="browsecomp",
    sections=BROWSECOMP_SECTIONS,
    domain_description=(
        "The agent is an AI research assistant that answers hard factual questions by searching "
        "the web. It uses a search tool (via MCP) to find relevant documents, reads snippets, "
        "and synthesizes an answer. Success is measured by a judge LLM comparing the agent's "
        "answer to a gold answer.\n\n"
        "The agent's required answer format is: 'Exact Answer: <answer>' followed by "
        "'Confidence: <0-100%>'. Do NOT add entries that change this format — the scoring "
        "extractor depends on it."
    ),
)

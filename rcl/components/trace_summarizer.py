"""TraceSummarizer — condenses raw traces using reflection context.

Sits between reflection and mutation in the pipeline:
    Execute → Group merge → Signal select → Reflect → [TraceSummarizer] → Mutate

One summarizer call per reflection group:
- No group: 1 trace → 1 reflection → 1 summarizer call
- Group(K): K traces (same task) → 1 reflection → 1 summarizer call
- Batch(M): M of the above → M summarizer calls (parallelizable)

The output replaces the raw trace strings on ExecutionTrace objects so
the mutator receives condensed traces without any code changes.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..core.data_structures import ExecutionTrace
from ..prompts.trace_summarizer import TRACE_SUMMARIZER_PROMPT
from .llm_client import create_generate_fn

logger = logging.getLogger(__name__)


class TraceSummarizer:
    """Condense execution traces based on reflection analysis."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: Optional[float] = None,
        parallelism: int = 1,
    ):
        self.model_name = model
        self.temperature = temperature
        self.parallelism = max(1, parallelism)
        self._generate = create_generate_fn(model, temperature)

    def summarize(
        self,
        signal_traces: List[ExecutionTrace],
        reflection: Dict[str, Any],
    ) -> List[ExecutionTrace]:
        """Condense signal traces using per-group reflection analysis.

        Args:
            signal_traces: The traces that were reflected on (1:1 with
                reflection["analyses"]).
            reflection: Output from the reflector, containing an "analyses"
                list aligned with signal_traces.

        Returns:
            The same ExecutionTrace objects with their trace text replaced
            by condensed versions. Originals are preserved in
            metadata["raw_trace"].
        """
        analyses = reflection.get("analyses", [])
        if not analyses or not signal_traces:
            return signal_traces

        # Build (trace, analysis) pairs — 1:1 mapping
        pairs = list(zip(signal_traces, analyses))

        if self.parallelism <= 1 or len(pairs) <= 1:
            for trace, analysis in pairs:
                self._summarize_one(trace, analysis)
        else:
            max_workers = min(self.parallelism, len(pairs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._summarize_one,
                        trace,
                        analysis,
                        create_generate_fn(self.model_name, self.temperature),
                    ): idx
                    for idx, (trace, analysis) in enumerate(pairs)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        future.result()
                    except Exception:
                        logger.exception(
                            "Trace summarization failed for task %s; keeping raw trace",
                            pairs[idx][0].task_id,
                        )

        return signal_traces

    def _summarize_one(
        self,
        trace: ExecutionTrace,
        analysis: Dict[str, Any],
        generate_fn=None,
    ) -> None:
        """Summarize a single trace group in-place."""
        gen = generate_fn or self._generate

        # Build the traces block — may include contrastive rollouts
        raw_trace = trace.get_afc_trace_str()
        eval_details = trace.metadata.get("evaluation_details", "")
        task_id = trace.task_id
        pass_pct = trace.metadata.get("pass_pct", 0.0)
        completed = trace.metadata.get("task_completed", False)

        traces_block = (
            f"### Task {task_id} "
            f"({'passed' if completed else 'failed'}, {pass_pct:.0f}% pass)\n\n"
            f"{raw_trace}"
        )

        analysis_text = analysis.get("analysis", "")

        prompt = TRACE_SUMMARIZER_PROMPT.format(
            analysis=analysis_text,
            traces_block=traces_block,
            eval_block=eval_details,
        )

        condensed = gen(prompt)

        # Preserve original and replace
        trace.metadata["raw_trace"] = raw_trace
        trace.trace = condensed
        # Also update afc_trace so get_afc_trace_str() returns condensed
        trace.metadata.pop("afc_trace", None)

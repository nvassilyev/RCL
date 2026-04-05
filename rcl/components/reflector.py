"""RCL Reflector — analyzes failures and assesses playbook entries.

Analyzes individual failed traces (one at a time) and outputs structured JSON with:
- entry_assessments: per-entry helpful/harmful/neutral tags (counts accumulate)
- analysis: free-form diagnostic text for the mutator

The system adapter is responsible for populating trace.trace and
trace.metadata["evaluation_details"] in the format appropriate for its benchmark.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..core.data_structures import EvaluationResult, ExecutionTrace, Playbook
from ..core.interfaces import Reflector
from .llm_client import create_generate_fn, extract_json_from_response

logger = logging.getLogger(__name__)

# Default prompts from unified templates
from ..prompts.reflector import BATCHED_REFLECTOR_PROMPT, REFLECTOR_PROMPT


class RCLReflector(Reflector):
    """Reflector that analyzes each failed trace individually.

    The reflector outputs structured JSON with entry_assessments and analysis.
    Entry assessments are used to update helpful/harmful counts on PlaybookEntry.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: Optional[float] = None,
        prompt_template: Optional[str] = None,
        domain_description: str = "",
        thinking: Optional[str] = "high",
        trace_parallelism: int = 1,
        optimization_state_context: str = "",
        batched_reflection: bool = False,
        batched_reflection_prompt: Optional[str] = None,
    ):
        self.model_name = model
        self.temperature = temperature
        self.prompt_template = prompt_template or REFLECTOR_PROMPT
        self.domain_description = domain_description
        self.thinking = thinking
        self.trace_parallelism = max(1, trace_parallelism)
        self.optimization_state_context = optimization_state_context
        self.batched_reflection = batched_reflection
        self.batched_reflection_prompt = batched_reflection_prompt or BATCHED_REFLECTOR_PROMPT
        self._generate = create_generate_fn(model, temperature, thinking=thinking)

    def _create_generate_fn(self):
        return create_generate_fn(
            self.model_name,
            self.temperature,
            thinking=self.thinking,
        )

    @staticmethod
    def _get_evaluation_details(trace: ExecutionTrace) -> str:
        """Get evaluation details from trace metadata."""
        details = trace.metadata.get("evaluation_details")
        if details:
            return details
        pass_pct = trace.metadata.get("pass_pct", 0.0)
        return f"Pass percentage: {pass_pct * 100:.1f}%."

    @staticmethod
    def _append_pp_trace(trace_text: str, pp_trace_text: str) -> str:
        """Append the supplementary PP trace when dual-trace metadata is present."""
        if not pp_trace_text:
            return trace_text
        return trace_text + (
            "\n\n" + "=" * 40 + "\n"
            "## Supplementary: PP Execution Trace\n\n"
            "The same task was also run with behavioral annotations. This trace\n"
            "contains XML tags revealing the agent's internal reasoning:\n"
            "- <playbook_cite>: which entries the agent consulted\n"
            "- <uncertainty>: where the agent was unsure\n"
            "- <reflection>: agent's self-assessment of playbook helpfulness\n"
            "- <missing_guidance>: where guidance was missing\n\n"
            + pp_trace_text + "\n\n"
            "This is supplementary context. Your analysis should be grounded in the\n"
            "baseline trace above. Use the PP trace to understand WHY decisions were made.\n"
            + "=" * 40
        )

    def _compose_reflection_context(self) -> str:
        ctx = str(self.optimization_state_context or "").strip()
        return ctx

    @staticmethod
    def _inject_context(prompt: str, context: str) -> str:
        context = str(context or "").strip()
        if not context:
            return prompt
        if "## Your Task\n" in prompt:
            return prompt.replace("## Your Task\n", context + "\n\n## Your Task\n", 1)
        return prompt.rstrip() + "\n\n" + context

    @staticmethod
    def _parse_reflection(raw: str) -> Dict[str, Any]:
        """Parse structured reflector output into assessments + analysis.

        Expected format:
        {"entry_assessments": [...], "analysis": "..."}

        Falls back gracefully: if parsing fails, treats entire response as analysis.
        """
        parsed = extract_json_from_response(raw)
        if isinstance(parsed, dict) and "analysis" in parsed:
            assessments = parsed.get("entry_assessments", [])
            # Validate assessments
            valid = []
            for a in assessments:
                if isinstance(a, dict) and "entry_id" in a and "tag" in a:
                    tag = a["tag"].lower()
                    if tag in ("helpful", "harmful", "neutral"):
                        valid.append({"entry_id": a["entry_id"], "tag": tag})
            result = {
                "entry_assessments": valid,
                "analysis": parsed["analysis"],
            }
            # Preserve principles if present (generalization_single_pass style)
            if "principles" in parsed and isinstance(parsed["principles"], list):
                result["principles"] = parsed["principles"]
            # Preserve enriched auxiliary fields if present
            for aux_key in ("failure_type", "root_cause", "coverage_gaps"):
                if aux_key in parsed:
                    result[aux_key] = parsed[aux_key]
            return result
        # Fallback: treat entire response as analysis
        logger.warning("Reflector output not valid JSON — treating as free-form analysis")
        return {
            "entry_assessments": [],
            "analysis": raw,
        }

    def reflect(
        self,
        traces: List[ExecutionTrace],
        playbook: Playbook,
        evaluation: EvaluationResult,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze traces — batched (single LLM call) or per-trace then merge."""
        if self.batched_reflection and len(traces) > 1:
            return self._reflect_batched(traces, playbook)
        return self._reflect_standard(
            traces,
            playbook,
        )

    def _reflect_single_trace(
        self,
        trace: ExecutionTrace,
        playbook_str: str,
        *,
        generate_fn=None,
    ) -> Dict[str, Any]:
        """Reflect on a single trace. Thread-safe when generate_fn is provided."""
        gen = generate_fn or self._generate
        task_id = trace.task_id
        pass_pct = trace.metadata.get("pass_pct", 0.0)
        evaluation_details = self._get_evaluation_details(trace)
        trace_text = self._append_pp_trace(
            trace.get_afc_trace_str(),
            trace.metadata.get("pp_trace", ""),
        )

        prompt = self.prompt_template.format(
            domain_description=self.domain_description,
            trace=trace_text,
            evaluation_details=evaluation_details,
            playbook=playbook_str,
            optimization_state_context=self._compose_reflection_context(),
        )
        if "{optimization_state_context}" not in self.prompt_template:
            prompt = self._inject_context(
                prompt,
                self._compose_reflection_context(),
            )

        try:
            raw = gen(prompt)
        except Exception as exc:
            err_text = str(exc).lower()
            if "content filtering" in err_text or "blocked" in err_text:
                print(f"  [reflector] Content filter blocked output for {task_id}, skipping trace", flush=True)
                return {
                    "task_id": task_id,
                    "pass_pct": pass_pct,
                    "task_completed": trace.metadata.get("task_completed", False),
                    "analysis": "(Reflection skipped: content filter triggered on this trace)",
                    "entry_assessments": [],
                }
            raise
        parsed = self._parse_reflection(raw)

        entry = {
            "task_id": task_id,
            "pass_pct": pass_pct,
            "task_completed": trace.metadata.get("task_completed", False),
            "analysis": parsed["analysis"],
            "entry_assessments": parsed["entry_assessments"],
        }
        if "principles" in parsed:
            entry["principles"] = parsed["principles"]
        # Preserve enriched auxiliary fields
        for aux_key in ("failure_type", "root_cause", "coverage_gaps"):
            if aux_key in parsed:
                entry[aux_key] = parsed[aux_key]
        return entry

    def _reflect_standard(
        self,
        traces: List[ExecutionTrace],
        playbook: Playbook,
    ) -> Dict[str, Any]:
        playbook_str = playbook.to_prompt_with_counts()

        if self.trace_parallelism <= 1 or len(traces) <= 1:
            analyses = [
                self._reflect_single_trace(t, playbook_str)
                for t in traces
            ]
        else:
            analyses = [None] * len(traces)
            max_workers = min(self.trace_parallelism, len(traces))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._reflect_single_trace,
                        trace,
                        playbook_str,
                        generate_fn=self._create_generate_fn(),
                    ): idx
                    for idx, trace in enumerate(traces)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        analyses[idx] = future.result()
                    except Exception:
                        logger.exception(
                            "Parallel trace reflection failed for task %s; retrying sequentially",
                            traces[idx].task_id,
                        )
                        analyses[idx] = self._reflect_single_trace(
                            traces[idx], playbook_str
                        )

        # Filter out None entries from failed parallel reflections
        analyses = [a for a in analyses if a is not None]

        all_assessments = []
        for a in analyses:
            all_assessments.extend(a["entry_assessments"])

        # Build combined_analysis: use structured merger when batching multiple traces
        has_principles = any("principles" in a for a in analyses)
        if has_principles:
            combined = self._merge_reflections_structured(analyses)
        elif len(analyses) > 1:
            combined = self._merge_reflections_general(analyses)
        else:
            # Single trace — concatenate
            analysis_parts = [a.get("analysis", "") for a in analyses]
            combined = "\n\n---\n\n".join(analysis_parts)

        # Append enriched auxiliary fields when present
        aux_parts = []
        for a in analyses:
            task_id = a.get("task_id", "")
            prefix = f"[{task_id}] " if task_id and len(analyses) > 1 else ""
            ft = a.get("failure_type", "")
            rc = a.get("root_cause", "")
            cg = a.get("coverage_gaps", "")
            if ft or rc or cg:
                lines = []
                if ft:
                    lines.append(f"{prefix}Failure type: {ft}")
                if rc:
                    lines.append(f"{prefix}Root cause: {rc}")
                if cg:
                    lines.append(f"{prefix}Coverage gaps: {cg}")
                aux_parts.append("\n".join(lines))
        if aux_parts:
            combined += "\n\n---\n\n" + "\n\n".join(aux_parts)

        result = {
            "analyses": analyses,
            "combined_analysis": combined,
            "all_entry_assessments": all_assessments,
        }
        return result

    def _reflect_batched(
        self,
        traces: List[ExecutionTrace],
        playbook: Playbook,
    ) -> Dict[str, Any]:
        """Reflect on all traces in a single LLM call."""
        playbook_str = playbook.to_prompt_with_counts()

        # Format all traces into a single block
        trace_parts = []
        for i, trace in enumerate(traces, 1):
            task_id = trace.task_id
            pass_pct = trace.metadata.get("pass_pct", 0.0)
            status = "PASS" if pass_pct >= 1.0 else f"FAIL (pass_pct={pass_pct:.0%})"
            evaluation_details = self._get_evaluation_details(trace)
            trace_text = self._append_pp_trace(
                trace.get_afc_trace_str(),
                trace.metadata.get("pp_trace", ""),
            )

            trace_parts.append(
                f"### Trace {i}: {task_id} — {status}\n\n"
                f"**Evaluation Feedback:**\n{evaluation_details}\n\n"
                f"**Execution Trace:**\n{trace_text}"
            )

        traces_block = "\n\n" + ("=" * 60 + "\n\n").join(trace_parts)

        prompt = self.batched_reflection_prompt.format(
            domain_description=self.domain_description,
            traces_block=traces_block,
            playbook=playbook_str,
        )
        context = self._compose_reflection_context()
        if context:
            prompt = self._inject_context(prompt, context)

        try:
            raw = self._generate(prompt)
        except Exception as exc:
            err_text = str(exc).lower()
            if "content filtering" in err_text or "blocked" in err_text:
                task_ids = [t.task_id for t in traces]
                print(f"  [reflector] Content filter blocked batched output for {task_ids}, falling back to per-trace", flush=True)
                return self._reflect_standard(traces, playbook)
            raise

        parsed = self._parse_reflection(raw)
        combined = parsed["analysis"]

        # Build per-trace analysis stubs for downstream consumers (trace writer,
        # trace summarizer, optimization state).  Each stub carries
        # the unified analysis so that consumers relying on per-trace analysis
        # text still get useful context.
        analyses = []
        for trace in traces:
            analyses.append({
                "task_id": trace.task_id,
                "pass_pct": trace.metadata.get("pass_pct", 0.0),
                "task_completed": trace.metadata.get("task_completed", False),
                "analysis": combined,
                "entry_assessments": [],
            })

        return {
            "analyses": analyses,
            "combined_analysis": combined,
            "all_entry_assessments": parsed["entry_assessments"],
        }

    @staticmethod
    def _merge_reflections_structured(analyses: list) -> str:
        """Merge per-trace reflections into a structured document with principles.

        Organizes by: batch summary -> deduplicated principles (grouped by
        coverage) -> entry assessment tally -> per-task evidence briefs.
        """
        parts = []

        # Section 1: Batch overview
        n_total = len(analyses)
        n_pass = sum(1 for a in analyses if a.get("pass_pct", 0) >= 1.0)
        n_partial = sum(1 for a in analyses if 0 < a.get("pass_pct", 0) < 1.0)
        n_fail = sum(1 for a in analyses if a.get("pass_pct", 0) == 0)
        parts.append(f"## Batch Summary\n\n{n_total} tasks analyzed: "
                      f"{n_pass} full pass, {n_partial} partial, {n_fail} fail.")

        # Section 2: Extracted principles, grouped by coverage
        all_principles = []
        for a in analyses:
            for p in a.get("principles", []):
                if isinstance(p, dict) and "statement" in p:
                    all_principles.append({
                        **p,
                        "source_task": a["task_id"],
                        "source_pass_pct": a.get("pass_pct", 0),
                    })

        if all_principles:
            # Group by coverage
            missing = [p for p in all_principles if p.get("coverage") == "MISSING"]
            weak = [p for p in all_principles if p.get("coverage") == "WEAK"]
            covered = [p for p in all_principles if p.get("coverage") == "COVERED"]

            parts.append("## Extracted Principles\n")

            if missing:
                parts.append("### MISSING from playbook (need ADD)")
                for p in missing:
                    level = p.get("transfer_level", p.get("generality", "?"))
                    parts.append(f"- [{level}] \"{p['statement']}\"")
                    if p.get("evidence"):
                        parts.append(f"  Evidence ({p['source_task']}, {p['source_pass_pct']*100:.0f}%): {p['evidence']}")
                parts.append("")

            if weak:
                parts.append("### WEAK in playbook (need UPDATE)")
                for p in weak:
                    level = p.get("transfer_level", p.get("generality", "?"))
                    parts.append(f"- [{level}] \"{p['statement']}\"")
                    if p.get("evidence"):
                        parts.append(f"  Evidence ({p['source_task']}, {p['source_pass_pct']*100:.0f}%): {p['evidence']}")
                parts.append("")

            if covered:
                parts.append("### Already COVERED (no action needed)")
                for p in covered:
                    level = p.get("transfer_level", p.get("generality", "?"))
                    parts.append(f"- [{level}] \"{p['statement']}\"")
                parts.append("")
        else:
            parts.append("## Extracted Principles\n\nNo new principles extracted from this batch.")

        # Section 3: Entry assessment tally
        assessment_counts: dict = {}  # entry_id -> {helpful: N, harmful: N, neutral: N}
        for a in analyses:
            for ea in a.get("entry_assessments", []):
                eid = ea.get("entry_id", "")
                tag = ea.get("tag", "")
                if eid not in assessment_counts:
                    assessment_counts[eid] = {"helpful": 0, "harmful": 0, "neutral": 0}
                if tag in assessment_counts[eid]:
                    assessment_counts[eid][tag] += 1

        if assessment_counts:
            parts.append("## Entry Assessment Tally\n")
            for eid, counts in sorted(assessment_counts.items()):
                tags = []
                if counts["helpful"]:
                    tags.append(f"+{counts['helpful']}")
                if counts["harmful"]:
                    tags.append(f"-{counts['harmful']}")
                if counts["neutral"]:
                    tags.append(f"~{counts['neutral']}")
                parts.append(f"- {eid}: {' '.join(tags)}")
            parts.append("")

        # Section 4: Per-task analysis briefs
        parts.append("## Per-Task Analysis\n")
        for a in analyses:
            status = "PASS" if a.get("pass_pct", 0) >= 1.0 else f"{a.get('pass_pct', 0)*100:.0f}%"
            parts.append(f"### {a['task_id']} ({status})\n")
            parts.append(a.get("analysis", "(no analysis)"))
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _merge_reflections_general(analyses: list) -> str:
        """Merge per-trace reflections into a structured document (no principles).

        Used when the reflector doesn't output principles (e.g., standard prompt)
        but we still want structured merging for the mutator.

        Organizes by: batch summary -> entry assessment tally -> per-task briefs.
        """
        parts = []

        # Section 1: Batch overview
        n_total = len(analyses)
        n_pass = sum(1 for a in analyses if a.get("pass_pct", 0) >= 1.0)
        n_partial = sum(1 for a in analyses if 0 < a.get("pass_pct", 0) < 1.0)
        n_fail = sum(1 for a in analyses if a.get("pass_pct", 0) == 0)
        parts.append(f"## Batch Summary\n\n{n_total} tasks analyzed: "
                      f"{n_pass} full pass, {n_partial} partial, {n_fail} fail.")

        # Section 2: Entry assessment tally
        assessment_counts: dict = {}
        for a in analyses:
            for ea in a.get("entry_assessments", []):
                eid = ea.get("entry_id", "")
                tag = ea.get("tag", "")
                if eid not in assessment_counts:
                    assessment_counts[eid] = {"helpful": 0, "harmful": 0, "neutral": 0}
                if tag in assessment_counts[eid]:
                    assessment_counts[eid][tag] += 1

        if assessment_counts:
            parts.append("## Entry Assessment Tally\n")
            for eid, counts in sorted(assessment_counts.items()):
                tags = []
                if counts["helpful"]:
                    tags.append(f"+{counts['helpful']}")
                if counts["harmful"]:
                    tags.append(f"-{counts['harmful']}")
                if counts["neutral"]:
                    tags.append(f"~{counts['neutral']}")
                parts.append(f"- {eid}: {' '.join(tags)}")
            parts.append("")

        # Section 3: Per-task analysis briefs
        parts.append("## Per-Task Analysis\n")
        for a in analyses:
            status = "PASS" if a.get("pass_pct", 0) >= 1.0 else f"{a.get('pass_pct', 0)*100:.0f}%"
            parts.append(f"### {a['task_id']} ({status})\n")
            parts.append(a.get("analysis", "(no analysis)"))
            parts.append("")

        return "\n".join(parts)



"""RCL Mutator — proposes playbook mutations based on reflection."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.data_structures import Playbook
from ..core.interfaces import Mutator
from .llm_client import create_generate_fn, extract_json_from_response

logger = logging.getLogger(__name__)

# Placeholder prompts — will be replaced by unified prompt templates.
_DEFAULT_ADD_ONLY = "MUTATOR ADD-ONLY PROMPT NOT SET"
_DEFAULT_FULL = "MUTATOR FULL PROMPT NOT SET"


class RCLMutator(Mutator):
    """Mutator that proposes playbook mutations based on reflector analysis.

    Modes:
    - add_only=True: Only ADD operations (paper-faithful)
    - add_only=False: ADD/UPDATE/DELETE operations
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: Optional[float] = None,
        add_only: bool = True,
        prompt_template: Optional[str] = None,
        allowed_sections: Optional[set] = None,
        thinking: Optional[str] = "high",
        include_trace_in_prompt: bool = True,
    ):
        self.model_name = model
        self.temperature = temperature
        self.add_only = add_only
        self.allowed_sections = allowed_sections or set()
        self.include_trace_in_prompt = include_trace_in_prompt

        # prompt_template is the primary prompt built by build_mutator_prompt()
        # which encodes add_only vs full operations via the operations block.
        self._prompt = prompt_template or (
            _DEFAULT_ADD_ONLY if add_only else _DEFAULT_FULL
        )

        self._generate = create_generate_fn(model, temperature, thinking=thinking)

    @staticmethod
    def _normalize_source_tasks(value: Any, allowed_task_ids: Optional[set[str]] = None) -> List[str]:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            candidates = []

        out: List[str] = []
        seen = set()
        for item in candidates:
            task_id = str(item or "").strip()
            if not task_id or task_id in seen:
                continue
            if allowed_task_ids is not None and task_id not in allowed_task_ids:
                continue
            seen.add(task_id)
            out.append(task_id)
        return out

    def mutate(
        self,
        playbook: Playbook,
        reflection: Dict[str, Any],
        recently_assessed: Optional[set] = None,
        signal_traces: Optional[List] = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """Generate mutations based on reflection.

        Args:
            playbook: Current playbook
            reflection: Output from Reflector (with combined_analysis)
            recently_assessed: Entry IDs assessed this round (for highlighting)
            signal_traces: The traces that were reflected on (for context)
        """
        reflector_analysis = reflection.get("combined_analysis", "")
        playbook_str = playbook.to_prompt_with_counts(recently_assessed)

        # Build trace + evaluation_details from signal traces
        if signal_traces and self.include_trace_in_prompt:
            trace_parts = []
            eval_parts = []
            for t in signal_traces:
                trace_parts.append(t.get_afc_trace_str())
                eval_parts.append(t.metadata.get("evaluation_details", ""))
            trace_str = "\n\n---\n\n".join(trace_parts)
            eval_str = "\n\n---\n\n".join(eval_parts)
        else:
            trace_str = "(omitted by configuration)" if not self.include_trace_in_prompt else "(not available)"
            if signal_traces:
                eval_str = "\n\n---\n\n".join(
                    t.metadata.get("evaluation_details", "") for t in signal_traces
                )
            else:
                eval_str = "(not available)"

        signal_task_ids = []
        if signal_traces:
            seen_task_ids = set()
            for t in signal_traces:
                task_id = str(getattr(t, "task_id", "") or "").strip()
                if task_id and task_id not in seen_task_ids:
                    seen_task_ids.add(task_id)
                    signal_task_ids.append(task_id)

        fmt_kwargs = dict(
            playbook=playbook_str,
            trace=trace_str,
            evaluation_details=eval_str,
            reflector_analysis=reflector_analysis,
            current_count=len(playbook),
            signal_task_ids_context=(
                "## Reflected Task IDs\n\n"
                "Use these task IDs when filling each operation's `source_tasks` field.\n"
                + "\n".join(f"- {task_id}" for task_id in signal_task_ids)
            ) if signal_task_ids else "",
        )
        extra_prompt_fields = kwargs.get("extra_prompt_fields")
        if isinstance(extra_prompt_fields, dict):
            fmt_kwargs.update(extra_prompt_fields)

        # Fill optimization state placeholders
        fmt_kwargs.setdefault("optimization_state_context", kwargs.get("optimization_state_context", ""))
        prompt = self._prompt.format(**fmt_kwargs)

        raw_response = self._generate(prompt)

        # Parse JSON
        parsed = extract_json_from_response(raw_response)
        mutation_summary = ""
        if isinstance(parsed, dict):
            mutations = parsed.get("operations", [])
            mutation_summary = parsed.get("mutation_summary", "")
        elif isinstance(parsed, list):
            mutations = parsed
        else:
            mutations = []
            logger.warning("Mutator output was not valid JSON; treating as zero mutations")

        # Validate
        validated = []
        allowed_task_ids = set(signal_task_ids)
        default_source_tasks = list(signal_task_ids)
        for m in mutations:
            op = (m.get("op") or m.get("type") or "").upper()

            if self.add_only and op != "ADD":
                print(f"    Warning: Skipping non-ADD operation: {op}")
                continue

            if op not in ("ADD", "UPDATE", "DELETE"):
                continue

            if op == "ADD":
                section = m.get("section", "others")
                section = section.lower().replace(" ", "_")
                if self.allowed_sections and section not in self.allowed_sections:
                    section = "others"
                entry = {"op": "ADD", "section": section, "content": m.get("content", "")}
                if m.get("rationale"):
                    entry["rationale"] = m["rationale"]
                if m.get("expected_effect"):
                    entry["expected_effect"] = m["expected_effect"]
                if m.get("scope_hint"):
                    entry["scope_hint"] = m["scope_hint"]
                source_tasks = self._normalize_source_tasks(m.get("source_tasks"), allowed_task_ids)
                if source_tasks:
                    entry["source_tasks"] = source_tasks
                elif "source_tasks" not in m and default_source_tasks:
                    entry["source_tasks"] = list(default_source_tasks)
                validated.append(entry)
            elif op == "UPDATE":
                entry = {"op": "UPDATE", "entry_id": m.get("entry_id", ""), "content": m.get("content", "")}
                if m.get("rationale"):
                    entry["rationale"] = m["rationale"]
                if m.get("expected_effect"):
                    entry["expected_effect"] = m["expected_effect"]
                if m.get("scope_hint"):
                    entry["scope_hint"] = m["scope_hint"]
                source_tasks = self._normalize_source_tasks(m.get("source_tasks"), allowed_task_ids)
                if source_tasks:
                    entry["source_tasks"] = source_tasks
                elif "source_tasks" not in m and default_source_tasks:
                    entry["source_tasks"] = list(default_source_tasks)
                validated.append(entry)
            elif op == "DELETE":
                entry = {"op": "DELETE", "entry_id": m.get("entry_id", "")}
                if m.get("rationale"):
                    entry["rationale"] = m["rationale"]
                if m.get("expected_effect"):
                    entry["expected_effect"] = m["expected_effect"]
                source_tasks = self._normalize_source_tasks(m.get("source_tasks"), allowed_task_ids)
                if source_tasks:
                    entry["source_tasks"] = source_tasks
                elif "source_tasks" not in m and default_source_tasks:
                    entry["source_tasks"] = list(default_source_tasks)
                validated.append(entry)

        return validated, raw_response, mutation_summary

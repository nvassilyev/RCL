"""Optimization state manager for RCL."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from .data_structures import Playbook, PlaybookEntry
from ..components.llm_client import create_generate_fn, extract_json_from_response


DEFAULT_OPTIMIZATION_STATE: Dict[str, Any] = {
    "playbook_assessment": {
        "health": "unknown",
        "playbook_size_trend": "unknown",
        "coverage_inventory": {
            "total_entries": 0,
            "sections_present": [],
            "section_entry_counts": {},
            "underrepresented_sections": [],
            "observed_gap_areas": [],
        },
        "entry_maturity": {
            "battle_tested": [],
            "developing": [],
            "untested_or_new": [],
            "at_risk": [],
        },
        "coherence_issues": [],
    },
    "change_ledger": [],
    "open_hypotheses": [],
    "preserve_until_more_evidence": [],
    "interference_patterns": [],
    "strategy_memory": {
        "questions": [],
        "notes": [],
    },
    "model_observations": {
        "capability_gaps_observed": [],
        "strategies_that_work": [],
        "entries_the_model_struggles_with": [],
        "interference_observed": [],
    },
    "agent_reasoning_patterns": {
        "reliable_behaviors": [],
        "common_mistakes": [],
        "entry_styles_that_work": {},
        "scaffolding_effectiveness": {},
    },
    "optimization_velocity": {
        "stage": "exploration",
        "stage_rationale": "",
        "iterations_at_current_stage": 0,
        "recent_mutation_success_rate": "",
        "recommendation": "",
        "recurring_failure_patterns": [],
        "single_occurrence_failures": [],
    },
}

RECENT_LEDGER_CONTEXT_ITEMS = 5


def _deepcopy_default_state() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_OPTIMIZATION_STATE)


def _trim_list(values: Any, limit: int) -> List[Any]:
    if not isinstance(values, list):
        return []
    return values[:limit]


def _dedupe_trim_text(values: Any, limit: int) -> List[str]:
    if not isinstance(values, list):
        return []
    seen = set()
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _dedupe_trim_task_ids(values: Any, limit: int) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    seen = set()
    out: List[str] = []
    for value in values:
        task_id = str(value or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        out.append(task_id)
        if len(out) >= limit:
            break
    return out


def _merge_dict(default_value: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(default_value)
    for key, value in candidate.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            inner = dict(merged[key])
            inner.update(value)
            merged[key] = inner
        else:
            merged[key] = value
    return merged


def _trim_text(value: Any, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_strategy_memory(values: Any, max_history: int) -> Dict[str, List[str]]:
    if not isinstance(values, dict):
        values = {}
    return {
        "questions": _dedupe_trim_text(values.get("questions"), max_history),
        "notes": _dedupe_trim_text(values.get("notes"), max_history),
    }


def _normalize_change_ledger(values: Any, max_history: int) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []

    allowed_status = {"untested", "watch", "mixed", "validated", "harmful", "superseded"}
    allowed_evidence = {"single_trace", "multi_trace", "repeated", "mixed", "unclear"}
    allowed_reversal = {"high", "medium", "low"}
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in values:
        if not isinstance(item, dict):
            continue

        raw_iteration = item.get("iteration")
        if isinstance(raw_iteration, bool):
            raw_iteration = None
        elif isinstance(raw_iteration, str):
            raw_iteration = int(raw_iteration) if raw_iteration.isdigit() else None
        elif not isinstance(raw_iteration, int):
            raw_iteration = None

        summary = _trim_text(item.get("summary") or item.get("change"), 240)
        source_tasks = _dedupe_trim_text(
            item.get("source_tasks") or item.get("source_reflections"),
            6,
        )
        reflection_summary = _trim_text(
            item.get("reflection_summary") or item.get("analysis") or item.get("lesson"),
            320,
        )
        observed_effect = _trim_text(item.get("observed_effect") or item.get("result"), 280)
        notes = _trim_text(item.get("notes") or item.get("note"), 240)
        status = str(item.get("status") or "watch").strip().lower() or "watch"
        if status not in allowed_status:
            status = "watch"
        evidence_strength = str(item.get("evidence_strength") or "unclear").strip().lower() or "unclear"
        if evidence_strength not in allowed_evidence:
            evidence_strength = "unclear"
        reversal_risk = str(item.get("reversal_risk") or "medium").strip().lower() or "medium"
        if reversal_risk not in allowed_reversal:
            reversal_risk = "medium"

        operations: List[Dict[str, str]] = []
        for op_item in item.get("operations", []) or []:
            if not isinstance(op_item, dict):
                continue
            op = str(op_item.get("op") or "").upper()
            if op not in {"ADD", "UPDATE", "DELETE"}:
                continue
            normalized_op: Dict[str, str] = {"op": op}
            if op_item.get("entry_id"):
                normalized_op["entry_id"] = str(op_item["entry_id"]).strip()
            if op_item.get("section"):
                normalized_op["section"] = str(op_item["section"]).strip()
            rationale = _trim_text(op_item.get("rationale"), 180)
            if rationale:
                normalized_op["rationale"] = rationale
            expected_effect = _trim_text(op_item.get("expected_effect"), 180)
            if expected_effect:
                normalized_op["expected_effect"] = expected_effect
            scope_hint = _trim_text(op_item.get("scope_hint"), 120)
            if scope_hint:
                normalized_op["scope_hint"] = scope_hint
            op_source_tasks = _dedupe_trim_text(op_item.get("source_tasks"), 6)
            if op_source_tasks:
                normalized_op["source_tasks"] = op_source_tasks
            operations.append(normalized_op)
            if len(operations) >= 6:
                break

        if not any((summary, source_tasks, reflection_summary, operations, observed_effect, notes)):
            continue

        dedupe_key = json.dumps(
            {
                "iteration": raw_iteration,
                "summary": summary,
                "source_tasks": source_tasks,
                "operations": operations,
            },
            sort_keys=True,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        out.append(
            {
                "iteration": raw_iteration,
                "summary": summary,
                "source_tasks": source_tasks,
                "reflection_summary": reflection_summary,
                "operations": operations,
                "observed_effect": observed_effect,
                "status": status,
                "evidence_strength": evidence_strength,
                "reversal_risk": reversal_risk,
                "notes": notes,
            }
        )
        if len(out) >= max_history:
            break
    return out


def _legacy_learning_buckets(values: Any, *, limit: int) -> Dict[str, List[str]]:
    buckets = {"keep": [], "watch": [], "avoid": []}
    if not isinstance(values, list):
        return buckets

    for item in values:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "watch").strip().lower()
        if status not in buckets:
            status = "watch"
        lesson = _trim_text(item.get("lesson") or item.get("summary"), 220)
        if lesson and lesson not in buckets[status]:
            buckets[status].append(lesson)
        if len(buckets[status]) >= limit:
            buckets[status] = buckets[status][:limit]
    return buckets


def _summarize_operations(operations: List[Dict[str, Any]], *, max_ops: int = 3) -> str:
    pieces = []
    for op in operations[:max_ops]:
        op_name = str(op.get("op") or "").upper()
        if not op_name:
            continue
        target = op.get("entry_id") or op.get("section") or "playbook"
        frag = f"{op_name} {target}"
        rationale = _trim_text(op.get("rationale"), 90)
        expected = _trim_text(op.get("expected_effect"), 90)
        op_source_tasks = _dedupe_trim_text(op.get("source_tasks"), 3)
        extras = []
        if rationale:
            extras.append(f"why: {rationale}")
        if expected:
            extras.append(f"goal: {expected}")
        if op_source_tasks:
            extras.append("tasks: " + ", ".join(op_source_tasks))
        if extras:
            frag += " (" + "; ".join(extras) + ")"
        pieces.append(frag)
    return " | ".join(pieces)


def _entry_total_signal(entry: PlaybookEntry) -> int:
    return int(entry.helpful_count or 0) + int(entry.harmful_count or 0)


def _entry_net_signal(entry: PlaybookEntry) -> int:
    return int(entry.helpful_count or 0) - int(entry.harmful_count or 0)


def _entry_brief(entry: PlaybookEntry) -> str:
    total = _entry_total_signal(entry)
    return (
        f"{entry.entry_id} [{entry.section}] "
        f"+{int(entry.helpful_count or 0)}/-{int(entry.harmful_count or 0)}"
        f", tested={total}"
    )


def _build_inventory_snapshot(
    playbook: Playbook,
    sampling_stats: Dict[str, Any],
    max_history: int,
) -> Dict[str, Any]:
    section_counts: Dict[str, int] = {}
    battle_tested: List[tuple] = []
    developing: List[tuple] = []
    untested_or_new: List[tuple] = []
    at_risk: List[tuple] = []

    for entry in playbook:
        section_counts[entry.section] = section_counts.get(entry.section, 0) + 1
        total = _entry_total_signal(entry)
        net = _entry_net_signal(entry)
        brief = _entry_brief(entry)

        if total >= 5:
            battle_tested.append((-net, -int(entry.helpful_count or 0), brief))
        elif total >= 3:
            developing.append((-net, -int(entry.helpful_count or 0), brief))
        else:
            untested_or_new.append((-net, brief))

        harm_margin = int(entry.harmful_count or 0) - int(entry.helpful_count or 0)
        if harm_margin >= 2:
            at_risk.append((-harm_margin, brief))

    battle_tested.sort()
    developing.sort()
    untested_or_new.sort()
    at_risk.sort()

    return {
        "coverage_inventory": {
            "total_entries": len(playbook),
            "sections_present": sorted(section_counts.keys()),
            "section_entry_counts": section_counts,
            "underrepresented_sections": sorted(
                section for section, count in section_counts.items() if count <= 1
            ),
            "observed_gap_areas": [],
        },
        "entry_maturity": {
            "battle_tested": [brief for _, _, brief in battle_tested[: max_history + 2]],
            "developing": [brief for _, _, brief in developing[: max_history + 2]],
            "untested_or_new": [brief for _, brief in untested_or_new[: max_history + 4]],
            "at_risk": [brief for _, brief in at_risk[: max_history + 2]],
        },
    }


def _normalize_state(raw: Dict[str, Any], max_history: int) -> Dict[str, Any]:
    state = _deepcopy_default_state()
    if not isinstance(raw, dict):
        return state

    for top_key, default_value in state.items():
        candidate = raw.get(top_key)
        if isinstance(default_value, dict) and isinstance(candidate, dict):
            state[top_key] = _merge_dict(default_value, candidate)
        elif candidate is not None:
            state[top_key] = candidate

    legacy_assessment = (
        raw.get("playbook_assessment", {})
        if isinstance(raw.get("playbook_assessment"), dict)
        else {}
    )
    playbook_assessment = state.get("playbook_assessment", {}) or {}
    coverage_inventory = playbook_assessment.get("coverage_inventory", {}) or {}
    entry_maturity = playbook_assessment.get("entry_maturity", {}) or {}

    legacy_gap_areas = legacy_assessment.get("coverage_gaps") or []
    if legacy_gap_areas and not coverage_inventory.get("observed_gap_areas"):
        coverage_inventory["observed_gap_areas"] = _dedupe_trim_text(legacy_gap_areas, max_history)

    strongest = legacy_assessment.get("strongest_entries") or []
    if strongest and not entry_maturity.get("battle_tested"):
        entry_maturity["battle_tested"] = _dedupe_trim_text(strongest, max_history)

    weakest = legacy_assessment.get("weakest_entries") or []
    if weakest and not entry_maturity.get("at_risk"):
        entry_maturity["at_risk"] = _dedupe_trim_text(weakest, max_history)

    coverage_inventory["sections_present"] = _dedupe_trim_text(
        coverage_inventory.get("sections_present"), max_history * 3,
    )
    coverage_inventory["underrepresented_sections"] = _dedupe_trim_text(
        coverage_inventory.get("underrepresented_sections"), max_history * 2,
    )
    coverage_inventory["observed_gap_areas"] = _dedupe_trim_text(
        coverage_inventory.get("observed_gap_areas"), max_history,
    )

    for key in (
        "battle_tested",
        "developing",
        "untested_or_new",
        "at_risk",
    ):
        entry_maturity[key] = _dedupe_trim_text(entry_maturity.get(key), max_history + 2)

    playbook_assessment["coverage_inventory"] = coverage_inventory
    playbook_assessment["entry_maturity"] = entry_maturity
    playbook_assessment["coherence_issues"] = _dedupe_trim_text(
        playbook_assessment.get("coherence_issues"), max_history,
    )
    state["playbook_assessment"] = playbook_assessment

    legacy_direction = raw.get("optimization_direction", {}) if isinstance(raw.get("optimization_direction"), dict) else {}
    legacy_learning_log = raw.get("learning_log") if isinstance(raw.get("learning_log"), list) else None
    legacy_what_we_tried = raw.get("what_we_tried") if isinstance(raw.get("what_we_tried"), list) else None

    state["change_ledger"] = _normalize_change_ledger(
        raw.get("change_ledger"),
        max_history,
    )

    questions = raw.get("open_hypotheses") or legacy_direction.get("open_questions") or []
    preserve = raw.get("preserve_until_more_evidence") or legacy_direction.get("preserve_or_watch") or []
    interference = raw.get("interference_patterns") or []

    legacy_buckets = _legacy_learning_buckets(legacy_learning_log or legacy_what_we_tried or [], limit=max_history)
    state["open_hypotheses"] = _dedupe_trim_text(questions or legacy_buckets["watch"], max_history)
    state["preserve_until_more_evidence"] = _dedupe_trim_text(preserve, max_history)
    state["interference_patterns"] = _dedupe_trim_text(interference, max_history)

    strategy_memory = raw.get("strategy_memory")
    if not isinstance(strategy_memory, dict):
        strategy_memory = {
            "questions": raw.get("strategy_questions") or [],
            "notes": raw.get("strategy_notes") or [],
        }
    state["strategy_memory"] = _normalize_strategy_memory(strategy_memory, max_history)

    observations = state.get("model_observations", {}) or {}
    for key in (
        "capability_gaps_observed",
        "strategies_that_work",
        "entries_the_model_struggles_with",
        "interference_observed",
    ):
        observations[key] = _dedupe_trim_text(observations.get(key), max_history)
    state["model_observations"] = observations

    reasoning = state.get("agent_reasoning_patterns", {}) or {}
    for key in ("reliable_behaviors", "common_mistakes"):
        reasoning[key] = _dedupe_trim_text(reasoning.get(key), max_history)
    if not isinstance(reasoning.get("entry_styles_that_work"), dict):
        reasoning["entry_styles_that_work"] = {}
    if not isinstance(reasoning.get("scaffolding_effectiveness"), dict):
        reasoning["scaffolding_effectiveness"] = {}
    state["agent_reasoning_patterns"] = reasoning

    return state


def _apply_deterministic_snapshot(
    state: Dict[str, Any],
    snapshot: Dict[str, Any],
    max_history: int,
) -> Dict[str, Any]:
    normalized = _normalize_state(state, max_history)
    assessment = normalized.get("playbook_assessment", {}) or {}
    snapshot_cov = snapshot.get("coverage_inventory", {}) or {}
    snapshot_maturity = snapshot.get("entry_maturity", {}) or {}
    current_cov = assessment.get("coverage_inventory", {}) or {}
    current_gap_areas = current_cov.get("observed_gap_areas") or []
    snapshot_cov = dict(snapshot_cov)
    snapshot_cov["observed_gap_areas"] = _dedupe_trim_text(current_gap_areas, max_history)

    assessment["coverage_inventory"] = snapshot_cov
    assessment["entry_maturity"] = snapshot_maturity
    normalized["playbook_assessment"] = assessment
    return normalized


def _build_iteration_memory_seed(
    *,
    iteration: int,
    reflection: Optional[Dict[str, Any]],
    applied_mutations: List[Dict[str, Any]],
    train_eval: Any,
    mutation_summary: str = "",
) -> Dict[str, Any]:
    analyses = reflection.get("analyses", []) if isinstance(reflection, dict) else []
    reflection_source_tasks: List[str] = []
    if isinstance(analyses, list):
        for item in analyses:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or "").strip()
            if task_id and task_id not in reflection_source_tasks:
                reflection_source_tasks.append(task_id)
            if len(reflection_source_tasks) >= 8:
                break

    assessment_summary = {"helpful": 0, "harmful": 0, "neutral": 0, "unique_entries": 0}
    if isinstance(reflection, dict):
        assessments = reflection.get("all_entry_assessments")
        if isinstance(assessments, list):
            seen_entries = set()
            for item in assessments:
                if not isinstance(item, dict):
                    continue
                tag = str(item.get("tag") or "").strip().lower()
                if tag in assessment_summary:
                    assessment_summary[tag] += 1
                entry_id = str(item.get("entry_id") or "").strip()
                if entry_id:
                    seen_entries.add(entry_id)
            assessment_summary["unique_entries"] = len(seen_entries)

    operations = []
    op_source_tasks_merged: List[str] = []
    for mutation in applied_mutations[:8]:
        if not isinstance(mutation, dict):
            continue
        op = str(mutation.get("op") or "").upper()
        if op not in {"ADD", "UPDATE", "DELETE"}:
            continue
        entry: Dict[str, Any] = {"op": op}
        if mutation.get("entry_id"):
            entry["entry_id"] = str(mutation.get("entry_id")).strip()
        if mutation.get("section"):
            entry["section"] = str(mutation.get("section")).strip()
        rationale = _trim_text(mutation.get("rationale"), 180)
        if rationale:
            entry["rationale"] = rationale
        expected_effect = _trim_text(mutation.get("expected_effect"), 180)
        if expected_effect:
            entry["expected_effect"] = expected_effect
        scope_hint = _trim_text(mutation.get("scope_hint"), 120)
        if scope_hint:
            entry["scope_hint"] = scope_hint
        op_source_tasks = _dedupe_trim_task_ids(mutation.get("source_tasks"), 6)
        if op_source_tasks:
            entry["source_tasks"] = op_source_tasks
            for task_id in op_source_tasks:
                if task_id not in op_source_tasks_merged:
                    op_source_tasks_merged.append(task_id)
                if len(op_source_tasks_merged) >= 8:
                    break
        operations.append(entry)

    memory_seed_source_tasks = _dedupe_trim_task_ids(
        reflection_source_tasks + op_source_tasks_merged,
        8,
    )

    return {
        "iteration": iteration,
        "source_tasks": memory_seed_source_tasks,
        "reflection_summary": _trim_text(
            reflection.get("combined_analysis", "") if isinstance(reflection, dict) else "",
            1200,
        ),
        "mutation_summary": _trim_text(mutation_summary, 320),
        "assessment_summary": assessment_summary,
        "applied_operations": operations,
        "train_snapshot": {
            "train_pass": round(float(getattr(train_eval, "score", 0.0) or 0.0), 4),
            "train_tgc": round(float(getattr(train_eval, "tgc", 0.0) or 0.0), 4),
        },
    }


class OptimizationStateManager:
    """Maintains a rolling optimization-state JSON document."""

    def __init__(
        self,
        model: str,
        target_model_name: str,
        thinking: Optional[str] = "high",
        max_history: int = 10,
    ):
        self.model_name = model
        self.target_model_name = target_model_name
        self.max_history = max(1, int(max_history))
        self._generate = create_generate_fn(model, thinking=thinking)
        self.state: Dict[str, Any] = _deepcopy_default_state()

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def load_dict(self, payload: Dict[str, Any]) -> None:
        self.state = _normalize_state(payload, self.max_history)

    def get_shared_context(self) -> str:
        state = self.state
        assessment = state.get("playbook_assessment", {}) or {}
        observations = state.get("model_observations", {}) or {}
        coverage = assessment.get("coverage_inventory", {}) or {}
        maturity = assessment.get("entry_maturity", {}) or {}
        change_ledger = state.get("change_ledger", []) or []
        strategy_memory = state.get("strategy_memory", {}) or {}
        open_hypotheses = state.get("open_hypotheses", []) or []
        preserve = state.get("preserve_until_more_evidence", []) or []
        interference = state.get("interference_patterns", []) or []

        lines = [
            "## Optimization State",
            "",
            f"- Playbook health: {assessment.get('health', 'unknown')}",
            f"- Playbook size trend: {assessment.get('playbook_size_trend', 'unknown')}",
            f"- Total entries: {coverage.get('total_entries', 0)}",
            "- Treat this state as historical memory for the mutator.",
            "- Use it to remember what changed, why it changed, what seemed to help, and what still needs evidence.",
            "- It is not a command source: the reflector's current analysis should drive this iteration's edits.",
        ]

        sections_present = coverage.get("sections_present") or []
        if sections_present:
            lines.append(f"- Sections present: {', '.join(str(v) for v in sections_present[:8])}")
        underrepresented = coverage.get("underrepresented_sections") or []
        if underrepresented:
            lines.append(
                f"- Thin coverage sections: {', '.join(str(v) for v in underrepresented[:6])}"
            )
        gap_areas = coverage.get("observed_gap_areas") or []
        if gap_areas:
            lines.append(f"- Observed gap areas: {', '.join(str(v) for v in gap_areas[:6])}")

        battle_tested = maturity.get("battle_tested") or []
        if battle_tested:
            lines.append(f"- Stable anchor entries: {', '.join(str(v) for v in battle_tested[:4])}")
        untested = maturity.get("untested_or_new") or []
        if untested:
            lines.append(
                f"- Untested/new entries to preserve: {', '.join(str(v) for v in untested[:4])}"
            )
        if open_hypotheses:
            lines.append(
                f"- Open hypotheses: {', '.join(str(v) for v in open_hypotheses[:3])}"
            )
        if preserve:
            lines.append(
                f"- Preserve until more evidence: {', '.join(str(v) for v in preserve[:4])}"
            )
        if interference:
            lines.append(f"- Known interference patterns: {', '.join(str(v) for v in interference[:3])}")

        questions = strategy_memory.get("questions") or []
        if questions:
            lines.append(f"- Strategy questions: {', '.join(str(v) for v in questions[:3])}")
        notes = strategy_memory.get("notes") or []
        if notes:
            lines.append(f"- Strategy notes: {', '.join(str(v) for v in notes[:3])}")

        strategies = observations.get("strategies_that_work") or []
        if strategies:
            lines.append(f"- Strategies that work: {', '.join(str(v) for v in strategies[:4])}")
        gaps = observations.get("capability_gaps_observed") or []
        if gaps:
            lines.append(f"- Capability gaps observed: {', '.join(str(v) for v in gaps[:4])}")

        # Only emit factual failure pattern data from velocity — no stage/recommendation
        velocity = state.get("optimization_velocity", {})
        if velocity:
            recurring = velocity.get("recurring_failure_patterns", [])
            if recurring:
                lines.append(f"- Recurring failure patterns: {', '.join(str(v) for v in recurring[:4])}")
            single = velocity.get("single_occurrence_failures", [])
            if single:
                lines.append(f"- Single-occurrence failure patterns: {', '.join(str(v) for v in single[:4])}")

        if change_ledger:
            lines.append("- Recent reflection -> mutation memory:")
            recent_entries = list(change_ledger[-RECENT_LEDGER_CONTEXT_ITEMS:])
            for entry in reversed(recent_entries):
                bits = []
                if entry.get("iteration") is not None:
                    bits.append(f"iter {entry['iteration']}")
                if entry.get("summary"):
                    bits.append(str(entry["summary"]))
                source_tasks = entry.get("source_tasks") or []
                if source_tasks:
                    bits.append(f"tasks: {', '.join(str(v) for v in source_tasks[:4])}")
                ops_summary = _summarize_operations(entry.get("operations") or [], max_ops=2)
                if ops_summary:
                    bits.append(f"ops: {ops_summary}")
                if entry.get("observed_effect"):
                    bits.append(f"observed: {entry['observed_effect']}")
                if entry.get("evidence_strength"):
                    bits.append(f"evidence: {entry['evidence_strength']}")
                if entry.get("reversal_risk"):
                    bits.append(f"reversal_risk: {entry['reversal_risk']}")
                if entry.get("notes"):
                    bits.append(f"notes: {entry['notes']}")
                if entry.get("status"):
                    bits.append(f"status: {entry['status']}")
                if bits:
                    lines.append(f"  - {' | '.join(bits)}")

        return "\n".join(lines).strip()

    def update(
        self,
        *,
        iteration: int,
        playbook: Playbook,
        train_eval: Any,
        reflection: Optional[Dict[str, Any]],
        applied_mutations: List[Dict[str, Any]],
        mutation_summary: str = "",
        recent_history: List[Dict[str, Any]],
        sampling_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update the rolling optimization state via one model call."""
        deterministic_snapshot = _build_inventory_snapshot(
            playbook, sampling_stats, self.max_history,
        )
        recent_history = recent_history[-5:] if recent_history else []
        reflection_summary = ""
        if isinstance(reflection, dict):
            reflection_summary = str(reflection.get("combined_analysis", ""))[:6000]
        iteration_memory_seed = _build_iteration_memory_seed(
            iteration=iteration,
            reflection=reflection,
            applied_mutations=applied_mutations,
            train_eval=train_eval,
            mutation_summary=mutation_summary,
        )

        prompt = f"""You maintain a rolling optimization-state document for an iterative playbook optimizer.

Update the JSON state using the latest evidence. Keep the state concise, cumulative, and useful for future reflector/mutator calls.

## Core Principles

The playbook is a training manual. Your state document should preserve institutional knowledge and keep coverage broad enough for the full task space.

- COMPREHENSIVENESS IS A FEATURE. A healthy playbook has enough chapters to cover the task space. Missing coverage is more dangerous than carrying an imperfect but unique entry.
- ZERO SIGNAL IS NOT DEAD WEIGHT. Untested or lightly tested entries need time and exposure, not panic deletion.
- DELETION REQUIRES REPEATED EVIDENCE. Use rewrite/refine by default. Recommend removal only when harm clearly outweighs help across multiple iterations.
- METRIC DIPS ARE NOT A CRISIS. Single-iteration dips often come from batch variance. Do not encode "crisis revert" or "manual should be tiny" lessons.
- TRACK COVERAGE, NOT JUST WINNERS. The state must remember which sections are thin, which gaps are still uncovered, and which new entries deserve more time.
- THIS STATE IS MEMORY, NOT CONTROL. Never issue tactical controller directives such as stop-changing edicts, hard pause commands, or target-size rules.
- CHANGE_LEDGER IS THE MAIN MEMORY SURFACE. Preserve change -> motivation -> hoped effect -> observed effect chains so future iterations understand what was tried, what helped, what hurt, and what still needs exposure.
- SHARED STATE. This JSON is consumed by the mutator as historical memory. It should preserve context and factual observations, not try to overrule current evidence.
- FACTS > STORIES. Use deterministic inputs and explicit task-linked evidence; do not invent confident causal narratives from weak signals.
- HYPOTHESES ARE CHEAP. A one-iteration metric move or one tricky trace belongs in `open_hypotheses`. Only promote to the change_ledger's `observed_effect` when there is a direct causal trace or repeated support.
- BE SPECIFIC ABOUT REVERSAL RISK. New or unique-coverage changes should usually carry high reversal risk until they get more exposure. That is memory, not control.
- NO SIZE OR EDIT-BUDGET DOCTRINE. Never write target entry counts, pause/hold commands, or hard no-edit / no-add / edit-budget directives.
- PRESERVE PROVENANCE. Keep task-level motivation attached to each change whenever it is available. Do not smear batch-wide edits across unrelated tasks.
- LEARNING RATE IS EMERGENT. Track whether the playbook is still exploring (many gaps, fast growth) or refining (few gaps, stable entries). Record this in `optimization_velocity`. Focus on trends across iterations, not single-batch metrics — batch performance varies heavily based on which tasks were sampled. Ask: are failures mostly easy mistakes (missing coverage, avoidable errors) or hard ones (variance, capability limits, edge cases)? A failure seen only once is an observation, not an action item. A failure seen repeatedly across different tasks is a real gap.

Focus on transferable lessons about:
- playbook health and coverage
- entry maturity and whether coverage is stabilizing or collapsing
- what changes were tried, why they were tried, and what should be preserved or revisited
- strategy principles/questions/notes that help future reasoning without dictating exact edits
- the target model's capability gaps, reasoning patterns, and entry styles that work

## Previous Optimization State
```json
{json.dumps(self.state, indent=2)}
```

## Deterministic Coverage Snapshot
Treat this block as the source of truth for coverage counts, maturity buckets, and sampling statistics. Do not invent contradictory counts.
```json
{json.dumps(deterministic_snapshot, indent=2)}
```

## Current Iteration
- iteration: {iteration}
- train_pass: {getattr(train_eval, 'score', 0.0):.4f}
- train_tgc: {getattr(train_eval, 'tgc', 0.0):.4f}
- playbook_size: {len(playbook)}

## Current Playbook
{playbook.to_prompt_with_counts()}

## Reflection Summary
{reflection_summary or "(none)"}

## Current Iteration Memory Seed
Treat this as a deterministic seed for the new ledger entry. Preserve its key facts.
```json
{json.dumps(iteration_memory_seed, indent=2)}
```

## Applied Mutations
```json
{json.dumps(applied_mutations[:12], indent=2)}
```

## Recent History
```json
{json.dumps(recent_history, indent=2)}
```

Output ONLY valid JSON with this schema:
{{
  "playbook_assessment": {{
    "health": "healthy|fragile|overloaded|interfering",
    "playbook_size_trend": "growing|stable|shrinking|overloaded",
    "coverage_inventory": {{
      "total_entries": 0,
      "sections_present": ["copy from snapshot"],
      "section_entry_counts": {{}},
      "underrepresented_sections": ["copy from snapshot"],
      "observed_gap_areas": ["missing chapter or scenario to add next"]
    }},
    "entry_maturity": {{
      "battle_tested": ["copy or summarize from snapshot"],
      "developing": ["copy or summarize from snapshot"],
      "untested_or_new": ["copy or summarize from snapshot"],
      "at_risk": ["entries that need rewrite or closer watch"]
    }},
    "coherence_issues": ["contradictions or interference to resolve"]
  }},
  "change_ledger": [
    {{
      "iteration": 0,
        "summary": "neutral summary of what changed this iteration",
      "source_tasks": ["task ids or reflection sources"],
      "reflection_summary": "what pattern or failure motivated the change",
      "operations": [
        {{
          "op": "ADD|UPDATE|DELETE",
          "entry_id": "optional",
          "section": "optional",
          "rationale": "why this change was made",
          "expected_effect": "what improvement it was meant to produce",
          "source_tasks": ["task ids directly motivating this op"]
        }}
      ],
      "observed_effect": "\"too early to judge\" if there is no later evidence yet; otherwise what later evidence suggests so far",
      "status": "untested|watch|mixed|validated|harmful|superseded",
      "evidence_strength": "single_trace|multi_trace|repeated|mixed|unclear",
      "reversal_risk": "high|medium|low",
      "notes": "what to remember before revisiting or undoing this; reference concrete entries or task families"
    }}
  ],
  "open_hypotheses": ["non-binding hypotheses or unresolved questions to verify; this is the default home for tentative interpretations"],
  "preserve_until_more_evidence": ["entries/changes with unique coverage or too little evidence to reverse yet; be concrete, not abstract"],
  "interference_patterns": ["known conflicts, overlaps, or trigger collisions to watch"],
  "strategy_memory": {{
    "questions": ["strategic questions to keep in mind"],
    "notes": ["neutral working notes; no commands, no edit budgets, no hold language"]
  }},
  "model_observations": {{
    "capability_gaps_observed": ["..."],
    "strategies_that_work": ["..."],
    "entries_the_model_struggles_with": ["..."],
    "interference_observed": ["..."]
  }},
  "agent_reasoning_patterns": {{
    "reliable_behaviors": ["..."],
    "common_mistakes": ["..."],
    "entry_styles_that_work": {{
      "style": "short note"
    }},
    "scaffolding_effectiveness": {{
      "scaffold": "short note"
    }}
  }},
  "optimization_velocity": {{
    "stage": "exploration|development|refinement|converged",
    "stage_rationale": "Trend-based: what patterns are we seeing across iterations? Are failures mostly easy mistakes (missing coverage) or hard ones (variance, capability limits)? Are we sampling the right tasks to get signal?",
    "iterations_at_current_stage": 0,
    "recent_mutation_success_rate": "e.g. '3/5 recent changes validated' or '1/4 — most changes untested'",
    "recommendation": "What trends suggest about mutation approach. NOT a command — observation that helps calibrate future reflections and mutations.",
    "recurring_failure_patterns": ["failure patterns confirmed across 2+ iterations or 2+ tasks — these are real gaps worth acting on"],
    "single_occurrence_failures": ["failure patterns seen only once — observe, don't act yet; batch variance is high"]
  }}
}}

Keep lists short (<= {self.max_history}).
When in doubt:
- prefer `open_hypotheses` over premature conclusions
- prefer `observed_effect: "too early to judge"` over invented causal claims
- prefer concrete preserve notes over abstract strategy slogans
- prefer evidence-linked memory over optimizer policy
Never issue optimizer commands. Record evidence, hypotheses, and change memory instead.
"""

        raw = self._generate(prompt)
        parsed = extract_json_from_response(raw)
        if not isinstance(parsed, dict):
            fallback = _apply_deterministic_snapshot(self.to_dict(), deterministic_snapshot, self.max_history)
            fallback_entry = {
                "iteration": iteration,
                "summary": "State update fallback preserved the current iteration memory seed",
                "source_tasks": iteration_memory_seed.get("source_tasks", []),
                "reflection_summary": _trim_text(iteration_memory_seed.get("reflection_summary"), 240),
                "operations": iteration_memory_seed.get("applied_operations", []),
                "observed_effect": "State update returned invalid JSON; prior memory retained and deterministic snapshot refreshed.",
                "status": "watch",
                "evidence_strength": "unclear",
                "reversal_risk": "high",
                "notes": "Do not treat this fallback as a controller. Revisit after more evidence.",
            }
            fallback["change_ledger"] = _normalize_change_ledger(
                [fallback_entry] + fallback.get("change_ledger", []),
                self.max_history,
            )
            fallback["preserve_until_more_evidence"] = _dedupe_trim_text(
                [
                    "Preserve unique coverage and recent changes while the state update is degraded.",
                    *fallback.get("preserve_until_more_evidence", []),
                ],
                self.max_history,
            )
            self.state = fallback
            return self.to_dict()

        normalized = _normalize_state(parsed, self.max_history)
        normalized = _apply_deterministic_snapshot(normalized, deterministic_snapshot, self.max_history)
        self.state = normalized
        return self.to_dict()

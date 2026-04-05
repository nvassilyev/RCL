"""Unified mutator (curator) prompt templates.

The mutator prompt is assembled by build_mutator_prompt() which combines:
- A base template (context, instructions, inputs)
- An operations block (ADD-only or full ADD/UPDATE/DELETE)
- Dynamic sections from BenchmarkConfig

Template variables:
- {playbook}: from Playbook.to_prompt_with_counts()
- {evaluation_details}: benchmark feedback (how it was scored)
- {signal_task_ids_context}: optional block listing the currently reflected task IDs
- {reflector_analysis}: combined analysis from reflector
- {sections_block}: formatted from BenchmarkConfig.sections (injected by builder)
- {operations_block}: ADD-only or full operations instructions (injected by builder)
- {trace_block}: optional raw execution trace section
- {reflection_mode_block}: optional guidance for auxiliary losses reflector
"""

from typing import Dict


MUTATOR_BASE = """\
You are a playbook curator in an iterative optimization loop. Your role \
is to improve a shared playbook based on analysis of the agent's recent \
performance.

## What is a Playbook?

A playbook is a training manual — institutional knowledge passed from \
experienced practitioners to a new hire. The agent reads the playbook as \
part of its system instructions and actively applies its guidance during \
execution. The playbook is organized into sections, each covering a different \
aspect of the agent's workflow. A good manual has chapters for every scenario \
the agent will encounter.

Important: The reflector analysis below was generated using ground truth / \
evaluation feedback that will NOT be available to the agent at test time. \
Your entries must help the agent succeed without access to these answers — \
focus on strategies and patterns, not task-specific solutions.

## Current Playbook

{playbook}
{trace_block}

## Evaluation Feedback

{evaluation_details}
{signal_task_ids_context}

## Reflector Analysis

{reflector_analysis}

## Entry Design Principles

The agent that will follow these entries is not a perfect reasoner. It may:
- misunderstand vague triggers or overloaded rules
- lose track of long multi-step procedures
- take plausible shortcuts that skip decisive checks
- produce confident but wrong outputs when asked to do brittle mechanical work

When curating entries, follow these principles:
- EXPLICIT > IMPLICIT: prefer direct triggers and concrete action words over subtle phrasing
- PROCEDURAL > DECLARATIVE: encode what to do, not just what is true
- CHECKPOINTED > MONOLITHIC: break fragile multi-step behavior into short stages with checkpoints
- GROUNDED > ABSTRACT: prefer concrete criteria, scratch artifacts, and verification gates over vague doctrine
- BALANCED > ONE-SIDED: use the most general formulation that still preserves \
the concrete anchors the model needs; avoid both brittle task-local hacks and \
elegant abstractions that remove actionable detail
- DEFENSIVE > OPTIMISTIC: assume the agent may skip a step unless the entry forces a check
- RECOVERABLE > FRAGILE: prefer entries that help the agent notice and correct mistakes, not just avoid them in theory

{optimization_state_block}
## Instructions

Each entry should read like advice from an experienced colleague:
- WHAT to do (concrete procedure or checklist)
- WHEN to do it (clear trigger)
- WHY (what failure mode this prevents)
- HOW CONFIDENT: use ALWAYS / TYPICALLY / WATCH FOR / UNCERTAIN markers

- Identify insights that are NEW and MISSING from the current playbook
- Avoid redundancy — if similar advice already exists, don't add near-duplicates
- Do NOT regenerate the entire playbook — only propose targeted changes
- Focus on quality over quantity — a focused playbook outperforms an exhaustive one
- Propose 0-3 operations per batch:
  - 0: The playbook already covers this failure pattern, or the evidence is \
execution variance / noise. Proposing nothing is often correct.
  - 1: One clear gap or one entry that needs rewriting. The most common case.
  - 2: Two genuinely independent issues — e.g., a missing coverage gap plus a \
separate entry that needs tightening, or a split (ADD + DELETE) to decompose \
an overloaded entry.
  - 3: Rare. Multiple independent failure patterns in one batch that each need \
their own entry. Never propose 3 edits for the same underlying issue.
- Each entry should be concise, actionable, and generalizable (not task-specific)
- When the failure is really about execution control, prefer short triggerable entries that enforce task completion, full instruction coverage, qualifying-set completeness, explicit tool/doc inspection, entity/account grounding, and final verification
- If an existing broad entry is close but weakly triggerable, prefer UPDATE, stronger emphasis, or splitting it into a shorter checklist-style rule over adding a near-duplicate sibling rule
- If the agent stopped after a partial solve, completed with unresolved criteria, or acted on the wrong entity/account/object, treat that as evidence for a reusable control rule rather than a task-specific patch
{reflection_mode_block}
{operations_block}

## Available Sections
{sections_block}

Output ONLY a valid JSON array (no other text):
```json
{json_example}
```

If no changes are needed, output:
```json
[]
```\
"""



OPTIMIZATION_STATE_MUTATOR_GUIDANCE = """

{optimization_state_context}

Treat the optimization state as historical memory and supporting context.
It should help you preserve validated lessons, remember open hypotheses,
review the recent reflection->mutation ledger, and avoid undoing useful work.
It is NOT a hard controller: let the current reflection evidence determine the
exact mutation batch.
Use it to preserve both high-level lessons and concrete load-bearing scaffolds
without forcing the playbook to express everything at one abstraction level.

Decision framework:
- ADD when the reflector identifies a task category with no playbook coverage. \
Missing a chapter is the highest-impact gap.
- UPDATE when an existing entry partially covers the scenario but its trigger, \
procedure, or framing needs revision. Rewrite as one coherent unit.
- DELETE only when an entry has consistent evidence of harm (harmful > helpful \
by 3+ across multiple iterations), OR is genuinely redundant with another \
entry that fully subsumes its coverage.
- Before proposing DELETE, ask: "Would I remove this chapter from a training \
manual? Or would I rewrite it to be clearer?"
- Do NOT delete zero-signal entries (+0/-0) — they cover scenarios that haven't \
been sampled yet.
- Review the section distribution of the current playbook. If an underrepresented \
section naturally fits the failure pattern, prefer adding a focused entry there \
instead of overloading an unrelated entry
"""


# --- Operations blocks ---

ADD_ONLY_OPS = """\

Available operations:
1. ADD — add a new entry to a section

Each operation needs: op, section, content.\
"""

FULL_OPS = """\

Available operations:
1. ADD — add a new entry to a section
2. UPDATE — rewrite an existing entry from scratch (reference by entry_id from the playbook above). Read the existing entry and the new evidence, then write the entry you wish had been there from the start — a coherent, self-contained unit that integrates both. A strong updated entry states a general rule in 1-2 sentences, then optionally adds a short checklist, example, or verification gate.
3. DELETE — remove an entry that is wrong, harmful, redundant, superseded, or overloaded enough that clean replacement is better than another rewrite (reference by entry_id)

UPDATE is the right choice when the new insight genuinely improves the same rule. Write the whole entry fresh each time — do not paste new paragraphs onto the old text.

When an existing entry partially covers a new insight:
- If the new insight belongs in the same trigger family: UPDATE it as one clean rewrite
- If the new insight is a genuinely separate concern: ADD a new entry, and optionally DELETE the old one if it is now redundant
- If an entry has accumulated multiple distinct triggers, gates, or examples: split it via ADD + DELETE instead of growing a mega-entry

Treat repeated evidence that an entry is confusing, contradictory, weakly helpful, or too bloated to apply quickly as a signal to rewrite or replace it.\
"""

OPTIMIZATION_STATE_OPS_SUFFIX = """
- For each operation, include a "rationale" field (1-2 sentences: why this change is needed, \
what evidence supports it)
- For each operation, include a "source_tasks" field (array of task IDs from the CURRENT \
reflected batch that directly motivated the change). Cite only task IDs supported by the \
current evidence. If the support is batch-level but not attributable to one task, use an \
empty array instead of guessing.\
"""


# --- JSON examples ---

_ADD_ONLY_EXAMPLE = """\
[
  {{"op": "ADD", "section": "<section_name>", "content": "Your new entry..."}}
]\
"""

_FULL_EXAMPLE = """\
[
  {{"op": "ADD", "section": "<section_name>", "content": "New entry..."}},
  {{"op": "UPDATE", "entry_id": "<id>", "content": "Rewritten entry..."}},
  {{"op": "DELETE", "entry_id": "<id>"}}
]\
"""

_OPT_STATE_ADD_ONLY_EXAMPLE = """\
{{
  "mutation_summary": "1-2 sentence summary of what you changed and why",
  "operations": [
    {{"op": "ADD", "section": "<section_name>", "content": "Your new entry...", \
"rationale": "Why this entry is needed...", "source_tasks": ["task:123"]}}
  ]
}}\
"""

_OPT_STATE_FULL_EXAMPLE = """\
{{
  "mutation_summary": "1-2 sentence summary of what you changed and why",
  "operations": [
    {{"op": "ADD", "section": "<section_name>", "content": "New entry...", \
"rationale": "...", "source_tasks": ["task:123"]}},
    {{"op": "UPDATE", "entry_id": "<id>", "content": "Rewritten entry...", \
"rationale": "Why this rewrite is needed...", "source_tasks": ["task:123", "task:456"]}},
    {{"op": "DELETE", "entry_id": "<id>", "rationale": "Why this should be removed...", \
"source_tasks": []}}
  ]
}}\
"""

AUXILIARY_LOSSES_MUTATOR_GUIDANCE = """
- The reflector analysis includes structured auxiliary fields (failure_type, \
root_cause, coverage_gaps) alongside the main analysis. Use all of them.
- If failure_type is EXECUTION_VARIANCE, bias toward 0 operations — the \
playbook likely already covers this pattern.
- If the evidence suggests a capability gap or complexity overload, redesign \
the entry into something the target model can execute reliably rather than \
making the same instruction longer or more demanding.
- Good entries often combine a broad lesson with concrete anchors when the \
concrete detail reduces inference burden for the weak model. Canonical \
API/tool names, exact trigger phrases, source-of-truth rules, intermediate \
artifacts, and verification steps are all valid if they are repeatedly \
evidenced and load-bearing.
- When the failure is procedural, prefer entries that encode: (a) the trigger, \
(b) the procedure or checklist to run, and (c) the verification gate or \
intermediate artifact that should exist before completion.
- Do not generalize away a concrete domain-native scaffold just because a \
broader slogan sounds cleaner. If removing the concrete anchor would make the \
weaker model guess what to do, keep it.
- Prefer operational process over slogans: decomposition recipes, comparison \
tables, scratch calculations, branch guards, and reasoning-to-output \
consistency checks are all valid playbook content when they generalize.
"""

TRACE_BLOCK = """
## Agent's Execution Trace

{trace}
"""

TRACE_OMITTED_BLOCK = """
## Agent's Execution Trace

Raw execution trace intentionally omitted for this curator variant.
Rely on the evaluation feedback and the structured reflection bundle below.
"""



def format_sections(sections: Dict[str, str]) -> str:
    """Format BenchmarkConfig.sections into a prompt block.

    Args:
        sections: {section_id: description} from BenchmarkConfig
    """
    lines = []
    for name, desc in sections.items():
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def build_mutator_prompt(
    sections: Dict[str, str],
    add_only: bool = False,
    include_trace: bool = True,
    playbook_budget: int = 0,
    include_optimization_state: bool = False,
    enriched_reflection: bool = False,
    **_kwargs,
) -> str:
    """Build the complete mutator prompt.

    Args:
        sections: {section_id: description} from BenchmarkConfig.sections
        add_only: If True, only ADD operations. If False, ADD/UPDATE/DELETE.
        include_trace: If True, include the raw execution trace section.
        include_optimization_state: If True, use optimization-state-aware JSON
            format (rationale + source_tasks) and include variance awareness.
        enriched_reflection: If True, inject entry-design guidance paired with
            enriched reflector's structured auxiliary fields.

    Returns:
        Prompt template string with {playbook}, {reflector_analysis},
        {task_descriptions} still as unfilled placeholders.
    """
    operations_block = ADD_ONLY_OPS if add_only else FULL_OPS
    if include_optimization_state:
        json_example = _OPT_STATE_ADD_ONLY_EXAMPLE if add_only else _OPT_STATE_FULL_EXAMPLE
        operations_block += OPTIMIZATION_STATE_OPS_SUFFIX
    else:
        json_example = _ADD_ONLY_EXAMPLE if add_only else _FULL_EXAMPLE
    sections_block = format_sections(sections)

    trace_block = TRACE_BLOCK if include_trace else TRACE_OMITTED_BLOCK
    if enriched_reflection:  # auxiliary losses primitive
        reflection_mode_block = AUXILIARY_LOSSES_MUTATOR_GUIDANCE
    else:
        reflection_mode_block = ""
    optimization_state_block = (
        OPTIMIZATION_STATE_MUTATOR_GUIDANCE if include_optimization_state else ""
    )

    fmt_kwargs = dict(
        operations_block=operations_block,
        json_example=json_example,
        sections_block=sections_block,
        trace_block=trace_block,
        reflection_mode_block=reflection_mode_block,
        optimization_state_block=optimization_state_block,
        # Leave these as literal placeholders for the caller to fill
        playbook="{playbook}",
        evaluation_details="{evaluation_details}",
        reflector_analysis="{reflector_analysis}",
        trace="{trace}",
        optimization_state_context="{optimization_state_context}",
        signal_task_ids_context="{signal_task_ids_context}",
    )
    return MUTATOR_BASE.format(**fmt_kwargs)

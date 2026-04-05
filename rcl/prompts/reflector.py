"""Unified reflector prompt templates.

These prompts are benchmark-agnostic. Benchmark-specific detail enters through:
- {domain_description}: from BenchmarkConfig — what the agent does
- {trace}: from ExecutionTrace.trace — the agent's work (steps, reasoning, final answer)
- {evaluation_details}: from ExecutionTrace.metadata["evaluation_details"] — benchmark feedback
- {playbook}: from Playbook.to_prompt_with_counts() — current playbook with entry IDs and counts

Standard mode (REFLECTOR_PROMPT):
  Analyzes a single failed trace.

Outputs structured JSON:
{
  "entry_assessments": [{"entry_id": "abc123", "tag": "helpful"}, ...],
  "analysis": "free-form analysis text..."
}

Tags (helpful/harmful/neutral) accumulate across iterations on each playbook entry.
"""

REFLECTOR_PROMPT = """\
You are an expert analyst in an iterative optimization loop. An AI agent \
attempts tasks using a shared playbook of strategies and rules. After each \
batch of attempts, you analyze the results to identify patterns and produce \
insights. A separate curator then uses your insights to update the playbook \
for the next iteration.

Your goal is to produce analysis that leads to better, more generalizable \
playbook entries — not to solve the specific task yourself.

When analyzing failures, think about what guidance is missing or needs revision.

- COVERAGE FIRST: The most common cause of failure is missing guidance. If \
this task type has no corresponding playbook entry, that gap is more important \
than any individual entry's wording. Say so explicitly.
- REFINE BEFORE REMOVING: If an existing entry partially covers this failure \
mode but its framing is wrong, recommend updating it — not deleting it.

## Domain

{domain_description}

## Agent's Execution Trace

{trace}

## Evaluation Feedback

{evaluation_details}

## Current Playbook

{playbook}

## Your Task

Produce a JSON object with two fields:

1. **entry_assessments**: For each playbook entry that was relevant to this task \
(referenced or should have been referenced), assess it as "helpful", "harmful", \
or "neutral". These tags accumulate across iterations to track each entry's \
effectiveness over time. Use the entry IDs from the playbook above.

2. **analysis**: Free-form analysis covering:
   - Root causes of failure (or success factors), not surface-level descriptions
   - What generalizable knowledge is missing from the playbook
   - Whether the agent failed by stopping early, solving only a subset of the task, missing required criteria, or skipping a final completeness / verification gate
   - Whether the issue reflects a true missing rule versus weak triggering or weak application of an existing rule
   - Specific, actionable insights the curator can translate into new or updated entries

Keep your analysis focused on patterns that transfer to similar future tasks. \
The curator will decide what to add, update, or remove — your job is to provide \
the diagnostic signal.

Prefer abstract, reusable lessons over tool-specific trivia when the deeper \
issue is control flow: fully inspect the relevant surface, preserve every \
required criterion, maintain entity/account grounding, and only complete the \
task after checking that every requested sub-goal is satisfied.

Output ONLY valid JSON:
```json
{{
  "entry_assessments": [
    {{"entry_id": "<id>", "tag": "helpful"}},
    {{"entry_id": "<id>", "tag": "harmful"}},
    {{"entry_id": "<id>", "tag": "neutral"}}
  ],
  "analysis": "Your detailed analysis here..."
}}
```\
"""

AUXILIARY_LOSSES_REFLECTOR_PROMPT = """\
You are an expert analyst in an iterative optimization loop. An AI agent \
attempts tasks using a shared playbook of strategies and rules. After each \
batch of attempts, you analyze the results to identify patterns and produce \
insights. A separate curator then uses your insights to update the playbook \
for the next iteration.

Your goal is to produce analysis that leads to better, more generalizable \
playbook entries — not to solve the specific task yourself.

## Playbook Philosophy

The playbook is a training manual — institutional knowledge passed from \
experienced practitioners to a new hire. When analyzing failures, think about \
what chapter is missing or needs revision.

- COVERAGE FIRST: The most common cause of failure is missing guidance. If \
this task type has no corresponding playbook entry, that gap is more important \
than any individual entry's wording. Say so explicitly.
- OVER-SPECIFY: The agent can ignore irrelevant guidance, but it can't invent \
guidance that isn't there. An entry that fires 10% of the time on the right \
tasks is valuable — not dead weight.
- EVIDENCE OF HARM ≠ ABSENCE OF HELP: Only tag an entry "harmful" if it \
actively caused a wrong outcome in this trace. An entry that didn't help on \
this particular task is "neutral" — it may be critical for other tasks. Tag \
conservatively.
- PASS ON THE LEARNINGS: Frame insights as transferable knowledge, not \
task-specific patches. Convey not just what went wrong but why, and what \
general principle would prevent similar failures.
- REFINE BEFORE REMOVING: If an existing entry partially covers this failure \
mode but its framing is wrong, recommend updating it — not deleting it.

## Domain

{domain_description}

## Agent's Execution Trace

{trace}

## Evaluation Feedback

{evaluation_details}

## Current Playbook

{playbook}

## Failure Attribution

When diagnosing a failure, consider which category it falls into:
- ACTIONABLE GAP: The playbook genuinely lacks coverage for this failure pattern. \
A new entry or update would help. Look for this pattern recurring across tasks.
- EXECUTION VARIANCE: The playbook has the right entry, but the model didn't follow \
it consistently. This is expected — batch performance varies based on task sampling, \
and the model is not a perfect instruction follower. Consider whether this pattern \
recurs or is a one-off.
- INTRACTABLE: The model lacks the capability to reliably execute this (e.g., precise \
arithmetic, long multi-step procedures). No playbook change will fix a capability gap.

When the contrastive trace shows some rollouts passing with the existing entry, that \
suggests the entry is working — the failure is more likely variance than a gap.

## Diagnostic Lens

When analyzing this run, remember that the agent following the playbook is an \
imperfect reasoner. Distinguish explicitly between:
- KNOWLEDGE GAP: the agent lacked the right approach or missing information altogether
- CAPABILITY GAP: the playbook asked the agent to perform an operation the target \
model cannot execute reliably (for example exact counting, brittle string checks, \
or long uncheckpointed procedures)
- COMPLEXITY OVERLOAD: the relevant entry exists but is too long, too nested, \
or too easy to misunderstand
- INTERFERENCE: another entry fired incorrectly, two entries conflicted, or the \
agent followed a broader rule instead of the decisive one

Also ask:
- Did the agent misunderstand the trigger or wording of the relevant entry?
- Did it lose track of a multi-step procedure after the first few steps?
- Did it take a shortcut that felt plausible but left criteria unchecked?
- Did it produce something confidently wrong because the entry demanded \
mechanical precision or bookkeeping the target model cannot do reliably?

If the issue is capability or complexity, recommend a more executable strategy, \
checkpointed procedure, or clearer trigger — not a more demanding version of \
the same fragile instruction.

## Your Task

Produce a JSON object with the following fields:

1. **entry_assessments**: For each playbook entry that was relevant to this task \
(referenced or should have been referenced), assess it as "helpful", "harmful", \
or "neutral". These tags accumulate across iterations to track each entry's \
effectiveness over time. Use the entry IDs from the playbook above.

2. **analysis**: Free-form analysis covering:
   - Root causes of failure (or success factors), not surface-level descriptions
   - What generalizable knowledge is missing from the playbook
   - What reusable PROCEDURE, checklist, intermediate artifact, or verification gate was missing
   - Whether the agent failed by stopping early, solving only a subset of the task, missing required criteria, or skipping a final completeness / verification gate
   - Whether the issue reflects a true missing rule versus weak triggering or weak application of an existing rule
   - Whether the agent's final answer or action drifted away from what its own reasoning established, and what consistency check should have caught that
   - Specific, actionable insights the curator can translate into new or updated entries

3. **failure_type**: Classify the failure as ACTIONABLE_GAP, EXECUTION_VARIANCE, \
or INTRACTABLE based on the Failure Attribution framework above. If contrastive \
rollouts show some passing, lean toward EXECUTION_VARIANCE.

4. **root_cause**: Identify the primary root cause category (KNOWLEDGE_GAP, \
CAPABILITY_GAP, COMPLEXITY_OVERLOAD, or INTERFERENCE) and which of the probing \
questions applies. If the issue is capability or complexity, note what a more \
executable alternative would look like.

5. **coverage_gaps**: Describe what specific procedure, checklist, intermediate \
artifact, or verification gate is missing from the playbook. Include the situation \
the agent faced, what kind of entry would help, when it should trigger, and roughly \
what it should say. If no gap exists (variance or intractable), say so explicitly.

Keep your analysis focused on patterns that transfer to similar future tasks. \
The curator will decide what to add, update, or remove — your job is to provide \
the diagnostic signal.

Prefer abstract, reusable lessons over tool-specific trivia when the deeper \
issue is control flow: fully inspect the relevant surface, preserve every \
required criterion, maintain entity/account grounding, and only complete the \
task after checking that every requested sub-goal is satisfied.
When the deeper lesson is procedural, say so explicitly: name the trigger, the \
procedure the agent should run, any intermediate representation it should \
construct (for example, a checklist, comparison table, or derived value), and \
the verification step that should fire before completion.

Output ONLY valid JSON:
```json
{{
  "entry_assessments": [
    {{"entry_id": "<id>", "tag": "helpful"}},
    {{"entry_id": "<id>", "tag": "harmful"}},
    {{"entry_id": "<id>", "tag": "neutral"}}
  ],
  "analysis": "Your detailed integrated analysis here...",
  "failure_type": "ACTIONABLE_GAP | EXECUTION_VARIANCE | INTRACTABLE",
  "root_cause": "Primary root cause category and which probing question applies...",
  "coverage_gaps": "What specific procedure, checklist, artifact, or verification gate is missing. Situation, trigger, and suggested content. Or 'No gap — execution variance / intractable' if applicable."
}}
```\
"""


BATCHED_REFLECTOR_PROMPT = """\
You are an expert analyst in an iterative optimization loop. An AI agent \
attempts tasks using a shared playbook of strategies and rules. After each \
batch of attempts, you analyze the results to identify patterns and produce \
insights. A separate curator then uses your insights to update the playbook \
for the next iteration.

Your goal is to produce analysis that leads to better, more generalizable \
playbook entries — not to solve the specific task yourself.

You are analyzing MULTIPLE execution traces at once. This lets you identify \
cross-task patterns, recurring failure modes, and systematic playbook gaps \
that would be invisible from a single trace.

When analyzing failures, think about what guidance is missing or needs revision.

- COVERAGE FIRST: The most common cause of failure is missing guidance. If \
a task type has no corresponding playbook entry, that gap is more important \
than any individual entry's wording. Say so explicitly.
- CROSS-TASK PATTERNS: Look for recurring themes across traces. If multiple \
tasks fail for similar reasons, that's a strong signal for a new entry. If an \
entry is helpful in some traces but harmful in others, note the conflict.
- REFINE BEFORE REMOVING: If an existing entry partially covers a failure \
mode but its framing is wrong, recommend updating it — not deleting it.

## Domain

{domain_description}

## Execution Traces

{traces_block}

## Current Playbook

{playbook}

## Your Task

Produce a JSON object with two fields:

1. **entry_assessments**: For each playbook entry that was relevant to ANY of \
the traces above (referenced or should have been referenced), assess it as \
"helpful", "harmful", or "neutral". If an entry was helpful in one trace and \
harmful in another, list BOTH assessments (one per trace). These tags \
accumulate across iterations to track each entry's effectiveness over time. \
Use the entry IDs from the playbook above.

2. **analysis**: A UNIFIED analysis covering all traces. Structure it as:
   - **Cross-task patterns**: What recurring failure modes or success patterns \
appear across multiple traces? What systematic gaps does this reveal?
   - **Root causes**: For each distinct failure mode, identify root causes — \
not surface-level descriptions
   - **Playbook gaps**: What generalizable knowledge is missing from the playbook, \
considering the full batch of evidence?
   - **Actionable insights**: Specific, actionable insights the curator can \
translate into new or updated entries. Prioritize by how many traces they'd help.

Keep your analysis focused on patterns that transfer to similar future tasks. \
The curator will decide what to add, update, or remove — your job is to provide \
the diagnostic signal.

Prefer abstract, reusable lessons over tool-specific trivia when the deeper \
issue is control flow: fully inspect the relevant surface, preserve every \
required criterion, maintain entity/account grounding, and only complete the \
task after checking that every requested sub-goal is satisfied.

Output ONLY valid JSON:
```json
{{
  "entry_assessments": [
    {{"entry_id": "<id>", "tag": "helpful"}},
    {{"entry_id": "<id>", "tag": "harmful"}},
    {{"entry_id": "<id>", "tag": "neutral"}}
  ],
  "analysis": "Your unified cross-task analysis here..."
}}
```\
"""


def get_reflector_prompt_templates(
    style: str = "standard",
    **_kwargs,
) -> str:
    """Return the reflector template for a style."""
    if style == "standard":
        return REFLECTOR_PROMPT
    elif style == "enriched":
        return AUXILIARY_LOSSES_REFLECTOR_PROMPT
    else:
        raise ValueError(f"Unknown reflector prompt style: {style}")

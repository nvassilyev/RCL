"""Prompt template for the TraceSummarizer component.

The summarizer condenses raw execution traces into focused excerpts
relevant to the reflector's diagnosis. It runs after reflection and
before mutation, so it can use the reflection output to decide what
trace moments matter.

Template variables:
- {analysis}: the reflector's analysis for this trace group
- {traces_block}: one or more raw traces (with metadata headers)
- {eval_block}: evaluation feedback for each trace
"""

TRACE_SUMMARIZER_PROMPT = """\
You are condensing execution traces for a downstream curator who will \
use them alongside a reflector's analysis to update a playbook of \
strategies for an AI agent.

The curator needs to verify the reflector's claims and see concrete \
failure patterns — not read the full execution history.

## Reflector's Analysis

{analysis}

## Evaluation Feedback

{eval_block}

## Execution Traces

{traces_block}

## Instructions

- You may be given multiple traces of the same task (contrastive rollouts). \
Preserve trace boundaries and labels (pass/fail, pass percentage).
- Focus on moments relevant to the reflector's diagnosis: decision points, \
failure moments, divergence between pass and fail rollouts.
- Preserve exact agent reasoning at critical moments — paraphrase \
everything else.
- For verbose tool outputs, describe what was returned, don't reproduce it.
- Omit successful steps unrelated to the diagnosis.
- A pass rollout's main value is showing what the agent did differently \
at the divergence point — condense accordingly.
- Output the condensed trace(s) directly as text. If there are multiple \
traces, use clear headers to separate them.\
"""

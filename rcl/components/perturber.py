"""Behavioral perturbation engine for dual-trace credit assignment.

Perturbation instructions are injected as a playbook suffix during training
execution. They ask the agent to verbalize its reasoning using XML tags,
producing richer traces for the reflector to analyze.

The definitions are benchmark-agnostic — the same tags work for any agent task.
"""

from typing import Dict, List

from ..core.data_structures import Playbook


PERTURBATIONS = {
    "cite_playbook": {
        "name": "Cite Playbook",
        "instruction": (
            "## Citation Mode\n"
            "IMPORTANT: Before each action, use <playbook_cite> tags to note which "
            "playbook entry (if any) is guiding your decision:\n"
            "- <playbook_cite>Following entry [abc123]: applying this strategy</playbook_cite>\n"
            "- <playbook_cite>No relevant entry — improvising</playbook_cite>"
        ),
        "tag": "<playbook_cite>",
        "tag_description": "shows which playbook entries the agent actually used (or found lacking)",
    },
    "ask_questions": {
        "name": "Ask Questions",
        "instruction": (
            "## Uncertainty Mode\n"
            "IMPORTANT: Before each action, use <uncertainty> tags to verbalize "
            "what you're unsure about:\n"
            "- <uncertainty>Not sure about the right approach here</uncertainty>\n"
            "- <uncertainty>Unclear whether this will work — making an assumption</uncertainty>"
        ),
        "tag": "<uncertainty>",
        "tag_description": "reveals what the agent found ambiguous or what assumptions it made",
    },
    "rewrite_instructions": {
        "name": "Rewrite Instructions",
        "instruction": (
            "## Reflection Mode\n"
            "IMPORTANT: After each action, use <reflection> tags to assess "
            "how well the playbook guided you:\n"
            "- <reflection>Entry [abc123] was helpful — it worked as expected</reflection>\n"
            "- <reflection>No playbook entry covered this situation</reflection>"
        ),
        "tag": "<reflection>",
        "tag_description": "contains the agent's own assessment of what worked and what didn't",
    },
    "structured_feedback": {
        "name": "Structured Feedback",
        "instruction": (
            "## Feedback Mode\n"
            "IMPORTANT: When you encounter a gap in the playbook, use <missing_guidance> tags:\n"
            "- <missing_guidance>Need a strategy for handling this case</missing_guidance>\n"
            "- <missing_guidance>No guidance on how to deal with this type of problem</missing_guidance>"
        ),
        "tag": "<missing_guidance>",
        "tag_description": "flags gaps in the playbook where the agent needed help but found none",
    },
}

PERTURBATION_SETS = {
    "minimal": ["cite_playbook"],
    "standard": ["cite_playbook", "structured_feedback"],
    "rich": ["ask_questions", "rewrite_instructions"],
    "full": list(PERTURBATIONS.keys()),
}


def build_reflector_tag_note(perturbation_names: List[str]) -> str:
    """Build a note for the reflector prompt describing the behavioral XML tags
    and how to use them for credit attribution analysis.

    The agent was instructed to use these tags during execution, so the trace
    will contain them. This note tells the reflector what each tag means and
    how to leverage them for stronger failure attribution.
    """
    tag_lines = []
    for name in perturbation_names:
        bp = PERTURBATIONS.get(name)
        if bp:
            tag_lines.append(f"- {bp['tag']} — {bp['tag_description']}")

    if not tag_lines:
        return ""

    return (
        "\n\n## Behavioral Tags in the Trace\n\n"
        "The agent was instructed to verbalize its reasoning using XML tags. "
        "These are direct signals from the agent about which entries it "
        "consulted and where it felt unsupported. Use them as primary evidence:\n"
        + "\n".join(tag_lines)
        + "\n\n"
        "Cross-reference these signals with the task outcome. An entry that was "
        "cited but led to a wrong action is harmful; an entry that was cited and "
        "contributed to correct behavior is helpful. An entry that didn't help on "
        "this particular task is neutral — it may be critical for other tasks.\n\n"
        "If the trace does NOT contain these tags, infer credit from the narrative. "
        "This is inherently noisier — only tag entries where evidence is reasonably "
        "clear. Omitting an entry with weak signal is better than guessing.\n\n"
        "## Credit Attribution\n\n"
        "When assessing entries, apply these principles:\n"
        "- OVER-SPECIFY: The agent can ignore irrelevant guidance, but it can't "
        "invent guidance that isn't there. An entry that fires 10% of the time on "
        "the right tasks is valuable — not dead weight.\n"
        "- EVIDENCE OF HARM ≠ ABSENCE OF HELP: Only tag an entry \"harmful\" if it "
        "actively caused a wrong outcome in this trace.\n\n"
        "Classify the failure into one of:\n"
        "- ACTIONABLE GAP: The playbook genuinely lacks coverage for this failure "
        "pattern. A new entry or update would help.\n"
        "- EXECUTION VARIANCE: The playbook has the right entry, but the model "
        "didn't follow it consistently. This is expected — the model is not a "
        "perfect instruction follower.\n"
        "- INTRACTABLE: The model lacks the capability to reliably execute this "
        "(e.g., precise arithmetic, long multi-step procedures). No playbook "
        "change will fix a capability gap.\n\n"
        "When the contrastive trace shows some rollouts passing with the existing "
        "entry, that suggests the entry is working — the failure is more likely "
        "variance than a gap.\n\n"
        "When attributing credit, distinguish explicitly between:\n"
        "- KNOWLEDGE GAP: the agent lacked the right approach or information\n"
        "- CAPABILITY GAP: the playbook asked the agent to perform an operation "
        "the target model cannot execute reliably\n"
        "- COMPLEXITY OVERLOAD: the relevant entry exists but is too long, too "
        "nested, or too easy to misunderstand\n"
        "- INTERFERENCE: another entry fired incorrectly, two entries conflicted, "
        "or the agent followed a broader rule instead of the decisive one\n\n"
        "For coverage gaps, describe the situation the agent faced and what kind "
        "of entry is missing: a procedure (reusable workflow), checklist (explicit "
        "gate/set of checks), artifact (intermediate state to construct), or "
        "verification_gate (final consistency check). When the gap is procedural, "
        "specify the trigger and the missing workflow rather than only naming an "
        "abstract idea."
    )


def make_perturbed_playbook(
    playbook: Playbook,
    perturbation_names: List[str],
    perturbation_registry: Dict[str, Dict] = None,
) -> Playbook:
    """Create a playbook copy with behavioral instructions appended as suffix.

    Args:
        playbook: Original playbook
        perturbation_names: Which perturbations to apply (e.g. ["ask_questions", "rewrite_instructions"])
        perturbation_registry: Custom registry. Defaults to the built-in PERTURBATIONS.
    """
    registry = perturbation_registry or PERTURBATIONS
    perturbed = playbook.copy()
    instruction_blocks = []
    for name in perturbation_names:
        bp = registry.get(name)
        if bp:
            instruction_blocks.append(bp["instruction"])

    if instruction_blocks:
        combined = "\n\n".join(instruction_blocks)
        perturbed._prompt_suffix = (
            "\n---\n\n"
            "# Behavioral Instructions (for this run only)\n\n"
            "The following instructions are NOT part of the playbook. "
            "They are temporary behavioral requirements for this run.\n\n"
            + combined
        )
    return perturbed

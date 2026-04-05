"""AppWorld benchmark configuration.

The AppWorld system adapter should populate ExecutionTrace with:
- trace.trace: Task instruction + agent's ReAct execution (code + observations)
- trace.metadata["evaluation_details"]: Test report + pass percentage
- trace.metadata["pass_pct"]: Float 0.0-1.0
- trace.metadata["task_completed"]: Bool
"""

from rcl.core.interfaces import BenchmarkConfig

APPWORLD_SECTIONS = {
    "strategies_and_hard_rules": "Core strategies and invariant rules for task execution",
    "apis_to_use_for_specific_information": "Which APIs to call for specific data needs",
    "useful_code_snippets_and_templates": "Ready-to-copy code patterns",
    "common_mistakes_and_correct_strategies": "Known pitfalls and how to avoid them",
    "problem_solving_heuristics_and_workflows": "Step-by-step approaches for common task types",
    "verification_checklist": "Checks to run before submitting",
    "troubleshooting_and_pitfalls": "Debugging strategies and runtime failure recovery",
    "others": "Miscellaneous guidance",
}

APPWORLD_CONFIG = BenchmarkConfig(
    name="appworld",
    sections=APPWORLD_SECTIONS,
    domain_description=(
        "The agent is an AI coding assistant that solves tasks in the AppWorld environment. "
        "It writes Python code to interact with various app APIs (email, calendar, Venmo, Spotify, etc.) "
        "to accomplish user requests. The agent uses a ReAct loop: it writes code, observes execution "
        "output, and iterates until the task is complete. Success is measured by unit tests."
    ),
)


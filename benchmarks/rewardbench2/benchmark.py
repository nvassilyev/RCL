"""RewardBench 2 benchmark configuration."""

from rcl.core.interfaces import BenchmarkConfig

REWARDBENCH2_SECTIONS = {
    "core_ranking_criteria": "General rules for scoring candidate responses against the user prompt",
    "instruction_following_checks": "How to verify constraints, formatting rules, and precise compliance",
    "factuality_and_math_checks": "How to detect hallucinations, factual errors, and mathematical mistakes",
    "safety_and_refusal_calibration": "How to judge appropriate compliance, refusal quality, and unsafe behavior",
    "tie_handling_and_score_calibration": "How to rate multiple valid answers and calibrate margins between good and bad answers",
    "common_mistakes": "Failure patterns that produce bad rankings or unstable scores",
}

REWARDBENCH2_CONFIG = BenchmarkConfig(
    name="rewardbench2",
    sections=REWARDBENCH2_SECTIONS,
    domain_description=(
        "The agent is an LLM judge that evaluates multiple candidate assistant responses for the "
        "same user prompt. It must independently rate each candidate, identify the best response "
        "or responses, and maintain good calibration across instruction following, factuality, "
        "math correctness, safety, focus, and cases with many equally valid answers. Success is "
        "measured by whether correct responses are scored above incorrect ones, with special "
        "official scoring for the Ties subset."
    ),
)

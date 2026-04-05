"""Unified prompt templates for RCL optimization.

All prompts are benchmark-agnostic. Benchmark-specific information enters
through template variables populated by BenchmarkConfig and ExecutionTrace.
"""

from .reflector import REFLECTOR_PROMPT
from .mutator import build_mutator_prompt

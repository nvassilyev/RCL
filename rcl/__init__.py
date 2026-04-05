"""RCL — Reflective Context Learning for LLM agents."""

from .core.data_structures import (
    EvaluationResult,
    ExecutionTrace,
    Playbook,
    PlaybookEntry,
)
from .core.config import RCLConfig
from .core.interfaces import Evaluator, Mutator, Reflector, SystemAdapter
from .core.optimizer import RCLOptimizer

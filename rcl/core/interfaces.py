"""Abstract interfaces for RCL optimization components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .data_structures import EvaluationResult, ExecutionTrace, Playbook


class SystemAdapter(ABC):
    """Execute tasks with a given playbook and return execution traces."""

    @abstractmethod
    def execute(
        self,
        task_ids: List[str],
        playbook: Playbook,
        **kwargs,
    ) -> List[ExecutionTrace]:
        pass

    @abstractmethod
    def get_ground_truth(self, task_id: str) -> Optional[str]:
        pass

    def clone_for_parallel(self) -> "SystemAdapter":
        """Return an isolated adapter instance for parallel execution.

        Adapters with internal mutable state or dedicated worker/server pools
        should override this. The default returns self, which signals that the
        adapter is not safely cloneable for true parallel slot execution.
        """
        return self


class Evaluator(ABC):
    """Score execution results."""

    @abstractmethod
    def evaluate(self, traces: List[ExecutionTrace]) -> EvaluationResult:
        pass


class Reflector(ABC):
    """Analyze failures and propose improvements."""

    @abstractmethod
    def reflect(
        self,
        traces: List[ExecutionTrace],
        playbook: Playbook,
        evaluation: EvaluationResult,
        **kwargs,
    ) -> Dict[str, Any]:
        pass


class Mutator(ABC):
    """Generate playbook mutations from reflection output."""

    @abstractmethod
    def mutate(
        self,
        playbook: Playbook,
        reflection: Dict[str, Any],
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str, str]:
        """Returns (mutations_list, raw_response_str, mutation_summary)."""
        pass


@dataclass
class BenchmarkConfig:
    """Configuration provided by each benchmark to parameterize prompts.

    Each benchmark (appworld, browsecomp, etc.) creates one of these.
    The system adapter is responsible for putting structured data into
    ExecutionTrace.metadata so the reflector can build prompts generically:

        trace.metadata["evaluation_details"]  — str, benchmark-specific feedback
            AppWorld: test report + pass percentage
            BrowseComp: verdict, gold answer, extracted answer, missed docs, judge reasoning

    The trace itself (trace.trace) should contain:
        - The question/task
        - The agent's work (search steps, code execution, reasoning)
        - The agent's final answer
        - Annotations are OK (e.g. ★ GOLD markers on retrieved docs)
        - But NOT evaluation results or ground truth answers
    """
    name: str
    sections: Dict[str, str]  # section_id -> human-readable description
    domain_description: str   # What the agent does (one paragraph)

    @property
    def section_names(self) -> set:
        return set(self.sections.keys())

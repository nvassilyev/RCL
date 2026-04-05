"""Configuration classes for RCL optimization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RCLConfig:
    """Configuration for RCL optimizer."""

    # Model settings
    model: str = "google/gemini-3-flash-preview"
    reflector_model: Optional[str] = None
    mutator_model: Optional[str] = None
    thinking_level: str = "none"
    reflector_prompt_style: str = "standard"

    # Optimization settings
    iterations: int = 10
    batch_size: int = 5
    mini_batch: int = 1  # how many failed traces to sample for reflection per iteration
    max_workers: int = 1
    reflection_threshold: float = 1.0  # reflect on tasks with pass_pct < this
    prune_threshold: int = 0  # auto-prune entries where harmful - helpful >= N (0 = disabled)
    entry_char_cap: int = 0  # reject ADD/UPDATE entries longer than N chars (0 = disabled)
    failure_replay_ratio: float = 0.0  # fraction of each batch reserved for replayed failures
    failure_replay_max_size: int = 0  # max retained failure tasks (0 = unbounded)
    failure_replay_unseen_first: bool = True  # prefer unseen tasks for the fresh portion
    failure_replay_passes_to_graduate: int = 5  # consecutive passes needed to evict from replay
    failure_replay_failures_to_evict: int = 3  # consecutive failures (post-reflection) to evict stuck tasks

    # Single-pass / pretraining settings
    single_pass: bool = False  # Sequential 1-pass through all tasks (no replay)
    reflect_all_traces: bool = False  # Include passing traces in signal (not just failures)
    playbook_budget: int = 0  # Max entries (0 = unlimited)

    # Structured optimizer state
    use_optimization_state: bool = False
    optimization_state_model: Optional[str] = None
    optimization_state_max_history: int = 10
    # Composable features
    perturbation_set: str = ""  # PP: empty = off, "rich"/"minimal"/"standard"/"full" = on
    group_size: int = 1  # Group: K rollouts per task (1 = off)
    dual_trace: bool = False  # Credit-assignment: run each task twice (baseline + PP) for richer reflection
    batched_reflection: bool = False  # Reflect on all signal traces in a single LLM call (vs per-trace)

    # Data settings
    seed: int = 42

    # Infrastructure
    output_dir: str = "results"
    checkpoint_interval: int = 1
    skip_validation: bool = False
    val_interval: int = 1

    def __post_init__(self):
        if self.reflector_model is None:
            self.reflector_model = self.model
        if self.mutator_model is None:
            self.mutator_model = self.model
        if self.optimization_state_model is None:
            self.optimization_state_model = self.reflector_model

    @classmethod
    def from_yaml(cls, path: str) -> "RCLConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

"""Public, platform-independent RL-UAV core."""

from .config import ModelConfig, PlannerConfig, RewardConfig
from .types import ObservationBatch, TargetContext, TargetKind, TaskId, TransitionBatch

__all__ = [
    "ModelConfig",
    "ObservationBatch",
    "PlannerConfig",
    "RewardConfig",
    "TargetContext",
    "TargetKind",
    "TaskId",
    "TransitionBatch",
]

__version__ = "0.1.0"


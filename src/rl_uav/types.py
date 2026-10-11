"""Shared data contracts used by planning, policy, reward, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np


class TaskId(IntEnum):
    NAVIGATION = 0
    HOVER = 1


class TargetKind(str, Enum):
    WAYPOINT = "waypoint"
    GOAL = "goal"
    SAFETY_GUIDE = "safety_guide"
    RECOVERY = "recovery"


def _point() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TargetContext:
    control_target: np.ndarray = field(default_factory=_point)
    mission_target: np.ndarray = field(default_factory=_point)
    next_mission_target: np.ndarray = field(default_factory=_point)
    kind: TargetKind = TargetKind.GOAL
    mission_kind: TargetKind = TargetKind.GOAL
    waypoint_index: int = -1
    route_version: int = 0
    feasible: bool = True
    diagnostics: str = ""

    def __post_init__(self) -> None:
        for name in ("control_target", "mission_target", "next_mission_target"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape}")
            object.__setattr__(self, name, value.copy())


@dataclass(slots=True)
class ObservationBatch:
    shared_obs: Any
    task_obs: Any
    task_id: Any
    task_obs_mask: Any

    def validate_shapes(self) -> None:
        batch = int(self.shared_obs.shape[0])
        if self.task_obs.shape[0] != batch or self.task_id.shape[0] != batch:
            raise ValueError("shared_obs, task_obs, and task_id must share a batch dimension")
        if self.task_obs_mask.shape != self.task_obs.shape:
            raise ValueError("task_obs_mask must have the same shape as task_obs")


@dataclass(slots=True)
class TransitionBatch:
    observations: ObservationBatch
    actions: Any
    rewards: Any
    values: Any
    log_probs: Any
    dones: Any
    task_ids: Any
    target_contexts: list[TargetContext]
    next_task_ids: Any | None = None

    def validate(self) -> None:
        self.observations.validate_shapes()
        batch = int(self.observations.shared_obs.shape[0])
        if len(self.target_contexts) != batch:
            raise ValueError("one TargetContext is required for every transition")


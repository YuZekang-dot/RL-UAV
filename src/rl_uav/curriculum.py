"""Recoverable min-steps-plus-evaluation-gate curriculum state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


class CurriculumStatus(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageThresholds:
    arrival_rate: float | None = None
    hover_success_rate: float | None = None
    waypoint_score: float | None = None
    collision_rate_max: float | None = None
    hover_error_max_m: float | None = None

    def passes(self, metrics: dict[str, float]) -> bool:
        minimums = {
            "arrival_rate": self.arrival_rate,
            "hover_success_rate": self.hover_success_rate,
            "waypoint_score": self.waypoint_score,
        }
        maximums = {
            "collision_rate": self.collision_rate_max,
            "hover_error_m": self.hover_error_max_m,
        }
        for key, threshold in minimums.items():
            if threshold is not None and (key not in metrics or float(metrics[key]) < threshold):
                return False
        for key, threshold in maximums.items():
            if threshold is not None and (key not in metrics or float(metrics[key]) > threshold):
                return False
        return True


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    min_transitions: int
    max_transitions: int
    num_obstacles: tuple[int, int]
    radius_choices: tuple[float, ...]
    num_waypoints: tuple[int, int]
    max_collisions: int
    learning_rate_scale: float
    thresholds: StageThresholds = field(default_factory=StageThresholds)
    terminal_stage: bool = False


@dataclass(slots=True)
class CurriculumState:
    stage_index: int = 0
    transitions_in_stage: int = 0
    total_transitions: int = 0
    consecutive_passes: int = 0
    status: CurriculumStatus = CurriculumStatus.ACTIVE
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CurriculumState":
        data = dict(payload)
        data["status"] = CurriculumStatus(data.get("status", CurriculumStatus.ACTIVE))
        return cls(**data)


class CurriculumScheduler:
    def __init__(
        self,
        stages: list[CurriculumStage],
        *,
        required_consecutive_passes: int = 2,
        state: CurriculumState | None = None,
    ) -> None:
        if not stages:
            raise ValueError("curriculum requires at least one stage")
        self.stages = stages
        self.required_consecutive_passes = int(required_consecutive_passes)
        self.state = state or CurriculumState()
        self._validate_state()

    @classmethod
    def from_toml(
        cls, path: str | Path, *, state: CurriculumState | None = None
    ) -> "CurriculumScheduler":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        stages = []
        for item in payload["stages"]:
            thresholds = StageThresholds(**item.get("thresholds", {}))
            stages.append(
                CurriculumStage(
                    name=item["name"],
                    min_transitions=int(item["min_transitions"]),
                    max_transitions=int(item["max_transitions"]),
                    num_obstacles=tuple(item["num_obstacles"]),
                    radius_choices=tuple(item["radius_choices"]),
                    num_waypoints=tuple(item["num_waypoints"]),
                    max_collisions=int(item["max_collisions"]),
                    learning_rate_scale=float(item["learning_rate_scale"]),
                    thresholds=thresholds,
                    terminal_stage=bool(item.get("terminal_stage", False)),
                )
            )
        return cls(
            stages,
            required_consecutive_passes=int(payload.get("required_consecutive_passes", 2)),
            state=state,
        )

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.state.stage_index]

    def add_transitions(self, count: int) -> None:
        if self.state.status is not CurriculumStatus.ACTIVE:
            return
        if count < 0:
            raise ValueError("transition count must be non-negative")
        self.state.transitions_in_stage += int(count)
        self.state.total_transitions += int(count)

    def observe_evaluation(self, metrics: dict[str, float]) -> CurriculumStatus:
        if self.state.status is not CurriculumStatus.ACTIVE:
            return self.state.status
        stage = self.current_stage
        passed = stage.thresholds.passes(metrics)
        self.state.consecutive_passes = self.state.consecutive_passes + 1 if passed else 0
        enough_steps = self.state.transitions_in_stage >= stage.min_transitions

        if stage.terminal_stage:
            if self.state.transitions_in_stage >= stage.max_transitions:
                self.state.status = CurriculumStatus.COMPLETE
            return self.state.status

        if enough_steps and self.state.consecutive_passes >= self.required_consecutive_passes:
            self.state.stage_index += 1
            self.state.transitions_in_stage = 0
            self.state.consecutive_passes = 0
            return self.state.status

        if self.state.transitions_in_stage >= stage.max_transitions:
            self.state.status = CurriculumStatus.FAILED
            self.state.failure_reason = (
                f"{stage.name} exceeded {stage.max_transitions} transitions without "
                f"{self.required_consecutive_passes} consecutive passing evaluations"
            )
        return self.state.status

    def _validate_state(self) -> None:
        if not 0 <= self.state.stage_index < len(self.stages):
            raise ValueError("curriculum stage index is outside configured stages")
        if self.state.transitions_in_stage < 0 or self.state.total_transitions < 0:
            raise ValueError("curriculum transition counters must be non-negative")


"""Typed configuration for the platform-independent core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    shared_obs_dim: int = 19
    navigation_obs_dim: int = 31
    hover_obs_dim: int = 5
    max_task_obs_dim: int = 31
    action_dim: int = 4
    encoder_hidden_dim: int = 128
    latent_dim: int = 32
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    init_action_std: float = 0.30
    min_action_std: float = 0.05
    max_action_std: float = 0.90
    tanh_epsilon: float = 1.0e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    drone_radius: float = 0.05
    safety_margin: float = 0.08
    graph_margin: float = 0.05
    circle_samples: int = 16
    average_speed: float = 2.7
    turn_time: float = 0.20
    time_buffer: float = 1.5
    arena_size: tuple[float, float, float] = (5.0, 5.0, 3.0)
    line_of_sight_epsilon: float = 1.0e-5
    guide_reached_radius: float = 0.18
    switch_hysteresis_steps: int = 4


@dataclass(frozen=True, slots=True)
class RewardConfig:
    navigation_progress: float = 6.0
    waypoint_bonus: float = 15.0
    goal_bonus: float = 100.0
    collision_penalty: float = 80.0
    obstacle_penalty: float = 2.0
    boundary_penalty: float = 1.2
    action_smooth_penalty: float = 0.05
    angular_rate_penalty: float = 0.02
    time_penalty: float = 0.002
    hover_position_penalty: float = 6.0
    hover_velocity_penalty: float = 0.30
    hover_angular_rate_penalty: float = 0.10
    hover_attitude_penalty: float = 0.20
    hover_success_bonus: float = 10.0


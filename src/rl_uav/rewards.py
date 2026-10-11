"""Task-isolated reward functions consuming the policy's TargetContext."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_uav.config import PlannerConfig, RewardConfig
from rl_uav.types import TargetContext


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]


def navigation_reward(
    *,
    previous_position: np.ndarray,
    position: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    target: TargetContext,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    waypoint_visited: bool = False,
    goal_reached: bool = False,
    collision: bool = False,
    config: RewardConfig | None = None,
    planner_config: PlannerConfig | None = None,
) -> RewardBreakdown:
    config = config or RewardConfig()
    planner_config = planner_config or PlannerConfig()
    previous_distance = float(np.linalg.norm(target.control_target - previous_position))
    current_distance = float(np.linalg.norm(target.control_target - position))
    components: dict[str, float] = {
        "target_progress": config.navigation_progress * (previous_distance - current_distance),
        "waypoint_bonus": config.waypoint_bonus if waypoint_visited else 0.0,
        "goal_bonus": config.goal_bonus if goal_reached else 0.0,
        "collision": -config.collision_penalty if collision else 0.0,
        "action_smooth": -config.action_smooth_penalty * float(np.linalg.norm(action - previous_action)),
        "angular_rate": -config.angular_rate_penalty * float(np.linalg.norm(angular_velocity)),
        "time": -config.time_penalty,
    }
    obstacles = np.asarray(obstacle_positions, dtype=np.float32).reshape(-1, 3)
    radii = np.asarray(obstacle_radii, dtype=np.float32).reshape(-1)
    active = radii > 0.0
    if active.any():
        distance = np.linalg.norm(obstacles[active, :2] - position[None, :2], axis=1)
        clearance = distance - radii[active] - planner_config.drone_radius
        components["obstacle"] = -config.obstacle_penalty * float(np.square(np.clip(0.30 - clearance, 0.0, None)).sum())
    else:
        components["obstacle"] = 0.0
    x, y, z = position
    length, width, height = planner_config.arena_size
    boundary_clearance = min(x, y, z, length - x, width - y, height - z)
    components["boundary"] = -config.boundary_penalty * max(0.0, 0.10 - float(boundary_clearance))
    _ = linear_velocity  # kept in the stable public interface for future velocity shaping
    return RewardBreakdown(sum(components.values()), components)


def hover_reward(
    *,
    position: np.ndarray,
    goal_position: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    rotation_matrix: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    stable: bool = False,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    config = config or RewardConfig()
    upright_error = 1.0 - float(np.asarray(rotation_matrix).reshape(3, 3)[2, 2])
    components = {
        "position": -config.hover_position_penalty * float(np.linalg.norm(position - goal_position)),
        "linear_velocity": -config.hover_velocity_penalty * float(np.linalg.norm(linear_velocity)),
        "angular_rate": -config.hover_angular_rate_penalty * float(np.linalg.norm(angular_velocity)),
        "attitude": -config.hover_attitude_penalty * abs(upright_error),
        "action_smooth": -config.action_smooth_penalty * float(np.linalg.norm(action - previous_action)),
        "stable_bonus": config.hover_success_bonus if stable else 0.0,
    }
    return RewardBreakdown(sum(components.values()), components)


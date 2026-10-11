"""Time-budgeted obstacle-aware waypoint ordering."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from rl_uav.config import PlannerConfig
from rl_uav.planning.geometry import shortest_safe_path


@dataclass(frozen=True, slots=True)
class RoutePlan:
    waypoint_indices: tuple[int, ...]
    estimated_time: float
    feasible: bool
    visited_count: int


def _pairwise_times(
    key_points: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    config: PlannerConfig,
) -> np.ndarray:
    count = key_points.shape[0]
    result = np.full((count, count), math.inf, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    for left in range(count):
        for right in range(left + 1, count):
            path = shortest_safe_path(
                key_points[left], key_points[right], obstacle_positions, obstacle_radii, config
            )
            if path.feasible:
                travel_time = path.length / max(config.average_speed, 1.0e-6) + config.turn_time
                result[left, right] = travel_time
                result[right, left] = travel_time
    return result


def optimize_waypoint_order(
    current_position: np.ndarray,
    waypoint_positions: np.ndarray,
    pending_mask: np.ndarray,
    goal_position: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    remaining_time: float,
    config: PlannerConfig | None = None,
) -> RoutePlan:
    """Maximize visited waypoints, then minimize travel time, while preserving goal reachability."""
    config = config or PlannerConfig()
    waypoints = np.asarray(waypoint_positions, dtype=np.float64).reshape(-1, 3)
    pending = np.asarray(pending_mask, dtype=bool).reshape(-1)[: waypoints.shape[0]]
    pending_indices = np.flatnonzero(pending)
    pending_points = waypoints[pending_indices]
    keys = np.vstack((np.asarray(current_position).reshape(1, 3), pending_points, np.asarray(goal_position).reshape(1, 3)))
    travel = _pairwise_times(keys, obstacle_positions, obstacle_radii, config)
    waypoint_count = len(pending_indices)
    goal_node = waypoint_count + 1
    budget = max(0.0, float(remaining_time) - config.time_buffer)

    direct_time = float(travel[0, goal_node])
    direct_feasible = math.isfinite(direct_time) and direct_time <= budget
    if waypoint_count == 0:
        return RoutePlan((), direct_time, direct_feasible, 0)

    state_count = 1 << waypoint_count
    dp = np.full((state_count, waypoint_count), math.inf, dtype=np.float64)
    parent = np.full((state_count, waypoint_count), -1, dtype=np.int16)
    for index in range(waypoint_count):
        dp[1 << index, index] = travel[0, index + 1]

    for mask in range(state_count):
        for last in range(waypoint_count):
            current_time = dp[mask, last]
            if not math.isfinite(current_time):
                continue
            for following in range(waypoint_count):
                if mask & (1 << following):
                    continue
                candidate = current_time + travel[last + 1, following + 1]
                next_mask = mask | (1 << following)
                if candidate < dp[next_mask, following]:
                    dp[next_mask, following] = candidate
                    parent[next_mask, following] = last

    best_mask = 0
    best_last = -1
    best_count = 0
    best_time = direct_time
    for mask in range(1, state_count):
        count = int(mask.bit_count())
        for last in range(waypoint_count):
            total = float(dp[mask, last] + travel[last + 1, goal_node])
            if not math.isfinite(total) or total > budget:
                continue
            if count > best_count or (count == best_count and total < best_time):
                best_mask, best_last, best_count, best_time = mask, last, count, total

    if best_last < 0:
        return RoutePlan((), direct_time, direct_feasible, 0)

    local_order: list[int] = []
    mask = best_mask
    last = best_last
    while last >= 0:
        local_order.append(last)
        previous = int(parent[mask, last])
        mask &= ~(1 << last)
        last = previous
    local_order.reverse()
    order = tuple(int(pending_indices[index]) for index in local_order)
    return RoutePlan(order, best_time, True, best_count)


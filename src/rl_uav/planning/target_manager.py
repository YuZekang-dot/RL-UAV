"""Stateful mission-target and safety-guide selection."""

from __future__ import annotations

import numpy as np

from rl_uav.config import PlannerConfig
from rl_uav.planning.geometry import line_of_sight_clear, safest_reachable_point, shortest_safe_path
from rl_uav.planning.route_optimizer import optimize_waypoint_order
from rl_uav.types import TargetContext, TargetKind


class TargetManager:
    """Unify waypoint, goal, safety-guide, and recovery targets.

    A guide is retained until reached or until direct mission visibility has
    remained clear for a configured number of updates. This small hysteresis
    avoids left/right target chatter around obstacle tangencies.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self._route_version = 0
        self._mission_key: tuple[TargetKind, int] | None = None
        self._active_guide: np.ndarray | None = None
        self._clear_steps = 0

    def reset(self) -> None:
        self._route_version += 1
        self._mission_key = None
        self._active_guide = None
        self._clear_steps = 0

    def update(
        self,
        current_position: np.ndarray,
        waypoint_positions: np.ndarray,
        pending_mask: np.ndarray,
        goal_position: np.ndarray,
        obstacle_positions: np.ndarray,
        obstacle_radii: np.ndarray,
        remaining_time: float,
    ) -> TargetContext:
        current = np.asarray(current_position, dtype=np.float32).reshape(3)
        waypoints = np.asarray(waypoint_positions, dtype=np.float32).reshape(-1, 3)
        goal = np.asarray(goal_position, dtype=np.float32).reshape(3)
        route = optimize_waypoint_order(
            current,
            waypoints,
            pending_mask,
            goal,
            obstacle_positions,
            obstacle_radii,
            remaining_time,
            self.config,
        )
        if route.waypoint_indices:
            waypoint_index = route.waypoint_indices[0]
            mission = waypoints[waypoint_index]
            mission_kind = TargetKind.WAYPOINT
            if len(route.waypoint_indices) > 1:
                next_mission = waypoints[route.waypoint_indices[1]]
            else:
                next_mission = goal
        else:
            waypoint_index = -1
            mission = goal
            mission_kind = TargetKind.GOAL
            next_mission = goal

        mission_key = (mission_kind, waypoint_index)
        if mission_key != self._mission_key:
            self._route_version += 1
            self._mission_key = mission_key
            self._active_guide = None
            self._clear_steps = 0

        centers = np.asarray(obstacle_positions, dtype=np.float32).reshape(-1, 3)
        clear = line_of_sight_clear(
            current[:2], mission[:2], centers[:, :2], obstacle_radii, self.config
        )
        if self._active_guide is not None:
            guide_reached = np.linalg.norm(current - self._active_guide) <= self.config.guide_reached_radius
            guide_visible = line_of_sight_clear(
                current[:2], self._active_guide[:2], centers[:, :2], obstacle_radii, self.config
            )
            self._clear_steps = self._clear_steps + 1 if clear else 0
            if not guide_reached and guide_visible and self._clear_steps < self.config.switch_hysteresis_steps:
                return self._context(
                    self._active_guide,
                    mission,
                    next_mission,
                    TargetKind.SAFETY_GUIDE,
                    mission_kind,
                    waypoint_index,
                    route.feasible,
                    "retained safety guide",
                )
            self._active_guide = None

        if clear:
            self._clear_steps = 0
            return self._context(
                mission,
                mission,
                next_mission,
                mission_kind,
                mission_kind,
                waypoint_index,
                route.feasible,
                "direct line of sight",
            )

        path = shortest_safe_path(
            current, mission, obstacle_positions, obstacle_radii, self.config
        )
        if path.feasible and path.points.shape[0] > 2:
            self._active_guide = path.points[1].copy()
            self._route_version += 1
            return self._context(
                self._active_guide,
                mission,
                next_mission,
                TargetKind.SAFETY_GUIDE,
                mission_kind,
                waypoint_index,
                True,
                "visibility-graph guide",
            )

        recovery = safest_reachable_point(
            current, mission, obstacle_positions, obstacle_radii, self.config
        )
        self._active_guide = recovery
        self._route_version += 1
        return self._context(
            recovery,
            mission,
            next_mission,
            TargetKind.RECOVERY,
            mission_kind,
            waypoint_index,
            False,
            path.diagnostics or "mission route infeasible",
        )

    def _context(
        self,
        control: np.ndarray,
        mission: np.ndarray,
        next_mission: np.ndarray,
        kind: TargetKind,
        mission_kind: TargetKind,
        waypoint_index: int,
        feasible: bool,
        diagnostics: str,
    ) -> TargetContext:
        return TargetContext(
            control_target=control,
            mission_target=mission,
            next_mission_target=next_mission,
            kind=kind,
            mission_kind=mission_kind,
            waypoint_index=waypoint_index,
            route_version=self._route_version,
            feasible=feasible,
            diagnostics=diagnostics,
        )


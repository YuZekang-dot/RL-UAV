"""Obstacle-aware route optimization and online target management."""

from .geometry import PathResult, line_of_sight_clear, shortest_safe_path
from .route_optimizer import RoutePlan, optimize_waypoint_order
from .target_manager import TargetManager

__all__ = [
    "PathResult",
    "RoutePlan",
    "TargetManager",
    "line_of_sight_clear",
    "optimize_waypoint_order",
    "shortest_safe_path",
]


"""2-D geometry for floor-to-ceiling cylindrical obstacles."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from rl_uav.config import PlannerConfig


@dataclass(frozen=True, slots=True)
class PathResult:
    points: np.ndarray
    length: float
    feasible: bool
    diagnostics: str = ""


def _as_points(values: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, width), dtype=np.float64)
    return array.reshape(-1, width)


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return Euclidean distance from a 2-D point to a closed segment."""
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1.0e-16:
        return float(np.linalg.norm(point - start))
    projection = float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + projection * delta)))


def line_of_sight_clear(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    obstacle_radii: np.ndarray,
    config: PlannerConfig | None = None,
) -> bool:
    """Check a segment against inflated cylindrical obstacles."""
    config = config or PlannerConfig()
    centers = _as_points(obstacle_xy, 2)
    radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError("obstacle positions and radii must have the same length")
    inflated = radii + config.drone_radius + config.safety_margin
    active = radii > 0.0
    for center, radius in zip(centers[active], inflated[active], strict=True):
        if point_segment_distance(center, start_xy, end_xy) < radius - config.line_of_sight_epsilon:
            return False
    return True


def _inside_arena(point: np.ndarray, config: PlannerConfig) -> bool:
    margin = config.drone_radius
    return bool(
        margin <= point[0] <= config.arena_size[0] - margin
        and margin <= point[1] <= config.arena_size[1] - margin
    )


def _sample_visibility_nodes(
    centers: np.ndarray,
    radii: np.ndarray,
    config: PlannerConfig,
) -> np.ndarray:
    nodes: list[np.ndarray] = []
    graph_radii = radii + config.drone_radius + config.safety_margin + config.graph_margin
    count = max(8, int(config.circle_samples))
    for center, radius in zip(centers, graph_radii, strict=True):
        for sample_index in range(count):
            angle = 2.0 * math.pi * sample_index / count
            node = center + radius * np.array([math.cos(angle), math.sin(angle)])
            if _inside_arena(node, config):
                nodes.append(node)
    if not nodes:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(nodes, dtype=np.float64)


def shortest_safe_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    config: PlannerConfig | None = None,
) -> PathResult:
    """Build a visibility graph and return a collision-free 3-D polyline.

    Obstacles span floor to ceiling, so collision checks are performed in XY.
    Z is interpolated by cumulative horizontal distance to avoid abrupt climbs.
    """
    config = config or PlannerConfig()
    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    centers3 = _as_points(obstacle_positions, 3)
    radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
    if centers3.shape[0] != radii.shape[0]:
        raise ValueError("obstacle positions and radii must have the same length")
    active = radii > 0.0
    centers = centers3[active, :2]
    radii = radii[active]

    if not _inside_arena(start[:2], config) or not _inside_arena(goal[:2], config):
        return PathResult(np.empty((0, 3), dtype=np.float32), math.inf, False, "endpoint outside arena")
    if line_of_sight_clear(start[:2], goal[:2], centers, radii, config):
        return PathResult(np.stack((start, goal)).astype(np.float32), float(np.linalg.norm(goal - start)), True)

    sampled = _sample_visibility_nodes(centers, radii, config)
    nodes = np.vstack((start[:2], goal[:2], sampled))
    node_count = nodes.shape[0]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if line_of_sight_clear(nodes[left], nodes[right], centers, radii, config):
                distance = float(np.linalg.norm(nodes[right] - nodes[left]))
                adjacency[left].append((right, distance))
                adjacency[right].append((left, distance))

    distances = [math.inf] * node_count
    parents = [-1] * node_count
    distances[0] = 0.0
    queue: list[tuple[float, int]] = [(0.0, 0)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distances[node]:
            continue
        if node == 1:
            break
        for neighbor, weight in adjacency[node]:
            candidate = current_distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    if not math.isfinite(distances[1]):
        return PathResult(np.empty((0, 3), dtype=np.float32), math.inf, False, "visibility graph disconnected")

    indices: list[int] = []
    cursor = 1
    while cursor >= 0:
        indices.append(cursor)
        cursor = parents[cursor]
    indices.reverse()
    xy_path = nodes[indices]
    segment_lengths = np.linalg.norm(np.diff(xy_path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_xy = float(cumulative[-1])
    fractions = cumulative / total_xy if total_xy > 1.0e-9 else np.zeros_like(cumulative)
    z = start[2] + fractions * (goal[2] - start[2])
    points = np.column_stack((xy_path, z)).astype(np.float32)
    length_3d = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return PathResult(points, length_3d, True)


def safest_reachable_point(
    current: np.ndarray,
    mission_target: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    config: PlannerConfig | None = None,
) -> np.ndarray:
    """Choose a deterministic high-clearance point for explicit recovery."""
    config = config or PlannerConfig()
    current = np.asarray(current, dtype=np.float64).reshape(3)
    mission_target = np.asarray(mission_target, dtype=np.float64).reshape(3)
    centers3 = _as_points(obstacle_positions, 3)
    radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
    active = radii > 0.0
    centers = centers3[active, :2]
    radii = radii[active]
    x_values = np.linspace(config.drone_radius + 0.1, config.arena_size[0] - config.drone_radius - 0.1, 11)
    y_values = np.linspace(config.drone_radius + 0.1, config.arena_size[1] - config.drone_radius - 0.1, 11)
    candidates = np.asarray([(x, y) for x in x_values for y in y_values], dtype=np.float64)
    best = current[:2]
    best_score = -math.inf
    inflated = radii + config.drone_radius + config.safety_margin
    for candidate in candidates:
        if not line_of_sight_clear(current[:2], candidate, centers, radii, config):
            continue
        clearance = float(np.min(np.linalg.norm(centers - candidate, axis=1) - inflated)) if centers.size else 10.0
        progress = float(np.linalg.norm(current[:2] - mission_target[:2]) - np.linalg.norm(candidate - mission_target[:2]))
        score = clearance + 0.15 * progress
        if score > best_score:
            best_score = score
            best = candidate
    return np.array([best[0], best[1], current[2]], dtype=np.float32)


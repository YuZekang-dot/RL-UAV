"""Decode the 95-D competition observation into shared and task views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_uav.config import ModelConfig, PlannerConfig
from rl_uav.types import ObservationBatch, TargetContext, TargetKind, TaskId


RAW_OBS_DIM = 95
START_POSITION = np.array([0.5, 2.5, 0.15], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class DecodedObservation:
    drone_position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    rotation_matrix: np.ndarray
    obstacle_positions: np.ndarray
    obstacle_radii: np.ndarray
    goal_position: np.ndarray
    phase: np.ndarray
    waypoint_positions: np.ndarray
    waypoint_pending: np.ndarray
    time_fraction: np.ndarray


def decode_raw_observation(raw_observation: np.ndarray) -> DecodedObservation:
    raw = np.asarray(raw_observation, dtype=np.float32)
    if raw.shape[-1] != RAW_OBS_DIM:
        raise ValueError(f"expected raw observation dimension {RAW_OBS_DIM}, got {raw.shape}")
    raw = raw.reshape(-1, RAW_OBS_DIM)
    start_relative = raw[:, 51:54]
    drone = START_POSITION[None, :] - start_relative
    obstacle_relative = raw[:, 3:27].reshape(-1, 8, 3)
    waypoint_relative = raw[:, 59:83].reshape(-1, 8, 3)
    obstacle_radii = raw[:, 43:51]
    waypoint_active = np.linalg.norm(waypoint_relative, axis=-1) > 1.0e-6
    waypoint_pending = waypoint_active & (raw[:, 83:91] < 0.5)
    return DecodedObservation(
        drone_position=drone,
        linear_velocity=raw[:, 27:30],
        angular_velocity=raw[:, 30:33],
        rotation_matrix=raw[:, 33:42].reshape(-1, 3, 3),
        obstacle_positions=drone[:, None, :] + obstacle_relative,
        obstacle_radii=obstacle_radii,
        goal_position=drone + raw[:, 54:57],
        phase=raw[:, 57:59],
        waypoint_positions=drone[:, None, :] + waypoint_relative,
        waypoint_pending=waypoint_pending,
        time_fraction=np.clip(raw[:, 91], 0.0, 1.0),
    )


def _target_features(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1.0e-6)
    return np.concatenate(([min(distance / 5.0, 1.0)], direction)).astype(np.float32)


def _kind_one_hot(kind: TargetKind) -> np.ndarray:
    order = (TargetKind.WAYPOINT, TargetKind.GOAL, TargetKind.SAFETY_GUIDE, TargetKind.RECOVERY)
    result = np.zeros(len(order), dtype=np.float32)
    result[order.index(kind)] = 1.0
    return result


def _obstacle_features(
    position: np.ndarray,
    target: np.ndarray,
    obstacles: np.ndarray,
    radii: np.ndarray,
    planner_config: PlannerConfig,
) -> np.ndarray:
    active = radii > 0.0
    relative = obstacles[:, :2] - position[None, :2]
    distance = np.linalg.norm(relative, axis=1)
    clearance = distance - radii - planner_config.drone_radius - planner_config.safety_margin
    target_delta = target[:2] - position[:2]
    target_direction = target_delta / max(float(np.linalg.norm(target_delta)), 1.0e-6)
    obstacle_direction = relative / np.maximum(distance[:, None], 1.0e-6)
    corridor = np.clip(obstacle_direction @ target_direction, -1.0, 1.0)
    danger = np.clip(1.0 - clearance / 0.70, 0.0, 1.0) * np.clip(0.5 + 0.5 * corridor, 0.0, 1.0)
    danger = np.where(active, danger, 0.0)
    top = np.argsort(-danger, kind="stable")[:4]
    features = np.zeros((4, 3), dtype=np.float32)
    for output_index, obstacle_index in enumerate(top):
        value = float(danger[obstacle_index])
        features[output_index] = (value, value * obstacle_direction[obstacle_index, 0], value * obstacle_direction[obstacle_index, 1])
    return features.reshape(-1)


def _boundary_features(position: np.ndarray, arena_size: tuple[float, float, float]) -> np.ndarray:
    x, y, z = position
    length, width, height = arena_size
    clearances = np.array([x, y, z, length - x, width - y, height - z], dtype=np.float32)
    margins = np.array([0.4, 0.4, 0.1, 0.4, 0.4, 0.4], dtype=np.float32)
    return np.square(np.clip((margins - clearances) / margins, 0.0, 1.0))


def build_observation_batch(
    raw_observation: np.ndarray,
    target_contexts: list[TargetContext],
    previous_actions: np.ndarray | None = None,
    model_config: ModelConfig | None = None,
    planner_config: PlannerConfig | None = None,
) -> ObservationBatch:
    model_config = model_config or ModelConfig()
    planner_config = planner_config or PlannerConfig()
    decoded = decode_raw_observation(raw_observation)
    batch_size = decoded.drone_position.shape[0]
    if len(target_contexts) != batch_size:
        raise ValueError("one TargetContext is required per raw observation")
    if previous_actions is None:
        previous_actions = np.zeros((batch_size, model_config.action_dim), dtype=np.float32)
    previous_actions = np.asarray(previous_actions, dtype=np.float32).reshape(batch_size, model_config.action_dim)

    rotation_6d = decoded.rotation_matrix[:, :, :2].reshape(batch_size, 6)
    shared = np.concatenate(
        (
            decoded.drone_position,
            rotation_6d,
            decoded.linear_velocity,
            decoded.angular_velocity,
            previous_actions,
        ),
        axis=1,
    ).astype(np.float32)
    if shared.shape[1] != model_config.shared_obs_dim:
        raise AssertionError(f"shared observation has {shared.shape[1]} dimensions")

    task_ids = np.where(decoded.phase[:, 1] > decoded.phase[:, 0], TaskId.HOVER, TaskId.NAVIGATION).astype(np.int64)
    task_obs = np.zeros((batch_size, model_config.max_task_obs_dim), dtype=np.float32)
    task_mask = np.zeros_like(task_obs)
    for index, context in enumerate(target_contexts):
        if task_ids[index] == TaskId.NAVIGATION:
            features = np.concatenate(
                (
                    _target_features(decoded.drone_position[index], context.control_target),
                    _target_features(decoded.drone_position[index], context.next_mission_target),
                    _kind_one_hot(context.kind),
                    _obstacle_features(
                        decoded.drone_position[index],
                        context.control_target,
                        decoded.obstacle_positions[index],
                        decoded.obstacle_radii[index],
                        planner_config,
                    ),
                    _boundary_features(decoded.drone_position[index], planner_config.arena_size),
                    [decoded.time_fraction[index]],
                )
            ).astype(np.float32)
            expected = model_config.navigation_obs_dim
        else:
            features = np.concatenate(
                (
                    _target_features(decoded.drone_position[index], decoded.goal_position[index]),
                    [decoded.time_fraction[index]],
                )
            ).astype(np.float32)
            expected = model_config.hover_obs_dim
        if features.shape[0] != expected:
            raise AssertionError(f"task observation has {features.shape[0]} dimensions, expected {expected}")
        task_obs[index, :expected] = features
        task_mask[index, :expected] = 1.0

    result = ObservationBatch(shared, task_obs, task_ids, task_mask)
    result.validate_shapes()
    return result


"""Thin KaiwuDRL-facing wrapper around the public multi-task core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:  # The real base class is available only in the authorized platform image.
    from kaiwudrl.interface.agent import BaseAgent
except ModuleNotFoundError:
    class BaseAgent:  # type: ignore[no-redef]
        """Local contract stub; it does not emulate KaiwuDRL runtime behavior."""

        def __init__(self, agent_type: str, device: str, logger: Any, monitor: Any) -> None:
            self.agent_type = agent_type


from rl_uav.algorithms import MultiTaskPPO
from rl_uav.checkpoint import load_checkpoint, save_checkpoint
from rl_uav.config import ModelConfig, PlannerConfig
from rl_uav.models import MultiTaskActorCritic
from rl_uav.observations import build_observation_batch, decode_raw_observation
from rl_uav.planning import TargetManager
from rl_uav.types import ObservationBatch, TargetContext, TargetKind, TaskId


class Agent(BaseAgent):
    def __init__(
        self,
        agent_type: str = "player",
        device: str | None = None,
        logger: Any = None,
        monitor: Any = None,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("Agent requires the 'model' extra with PyTorch installed") from error
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(agent_type, device, logger, monitor)
        self.torch = torch
        self.device = torch.device(device)
        self.logger = logger
        self.monitor = monitor
        self.model_config = ModelConfig()
        self.planner_config = PlannerConfig()
        self.model = MultiTaskActorCritic(self.model_config).to(self.device)
        self.algorithm = MultiTaskPPO(self.model)
        self._target_managers: list[TargetManager] = []
        self._previous_actions = np.empty((0, self.model_config.action_dim), dtype=np.float32)

    def _ensure_state(self, batch_size: int) -> None:
        if len(self._target_managers) != batch_size:
            self._target_managers = [TargetManager(self.planner_config) for _ in range(batch_size)]
            self._previous_actions = np.zeros(
                (batch_size, self.model_config.action_dim), dtype=np.float32
            )

    def observation_process(self, raw_observation: Any) -> ObservationBatch:
        if hasattr(raw_observation, "detach"):
            raw_numpy = raw_observation.detach().cpu().numpy()
        else:
            raw_numpy = np.asarray(raw_observation, dtype=np.float32)
        decoded = decode_raw_observation(raw_numpy)
        batch_size = decoded.drone_position.shape[0]
        self._ensure_state(batch_size)
        contexts: list[TargetContext] = []
        for index in range(batch_size):
            is_hover = decoded.phase[index, 1] > decoded.phase[index, 0]
            if is_hover:
                goal = decoded.goal_position[index]
                contexts.append(
                    TargetContext(
                        control_target=goal,
                        mission_target=goal,
                        next_mission_target=goal,
                        kind=TargetKind.GOAL,
                        mission_kind=TargetKind.GOAL,
                    )
                )
            else:
                contexts.append(
                    self._target_managers[index].update(
                        decoded.drone_position[index],
                        decoded.waypoint_positions[index],
                        decoded.waypoint_pending[index],
                        decoded.goal_position[index],
                        decoded.obstacle_positions[index],
                        decoded.obstacle_radii[index],
                        remaining_time=max(0.0, (1.0 - float(decoded.time_fraction[index])) * 30.0),
                    )
                )
        return build_observation_batch(
            raw_numpy,
            contexts,
            self._previous_actions,
            self.model_config,
            self.planner_config,
        )

    def _to_torch(self, observations: ObservationBatch) -> ObservationBatch:
        torch = self.torch
        return ObservationBatch(
            shared_obs=torch.as_tensor(observations.shared_obs, device=self.device, dtype=torch.float32),
            task_obs=torch.as_tensor(observations.task_obs, device=self.device, dtype=torch.float32),
            task_id=torch.as_tensor(observations.task_id, device=self.device, dtype=torch.long),
            task_obs_mask=torch.as_tensor(observations.task_obs_mask, device=self.device, dtype=torch.float32),
        )

    def predict(self, raw_observation: Any, return_policy_obs: bool = False):
        observations = self._to_torch(self.observation_process(raw_observation))
        with self.torch.no_grad():
            actions, values, log_probs = self.model.act(observations)
        self._previous_actions = actions.detach().cpu().numpy()
        if return_policy_obs:
            return actions, values, log_probs, observations
        return actions, values, log_probs

    def exploit(self, raw_observation: Any):
        observations = self._to_torch(self.observation_process(raw_observation))
        with self.torch.no_grad():
            actions, _, _ = self.model.act(observations, deterministic=True)
        self._previous_actions = actions.detach().cpu().numpy()
        return actions

    def value(self, raw_observation: Any):
        observations = self._to_torch(self.observation_process(raw_observation))
        with self.torch.no_grad():
            return self.model.values(observations)

    def learn(self, training_data: dict[str, Any]):
        required = {"observations", "actions", "old_log_probs", "returns", "advantages"}
        missing = required - set(training_data)
        if missing:
            raise KeyError(f"missing PPO training fields: {sorted(missing)}")
        return self.algorithm.update(**{key: training_data[key] for key in required})

    @staticmethod
    def action_process(action_data: Any) -> Any:
        return action_data

    def reset_envs(self, done_mask: Any | None = None) -> None:
        if done_mask is None:
            indices = range(len(self._target_managers))
        else:
            mask = np.asarray(done_mask, dtype=bool).reshape(-1)
            indices = np.flatnonzero(mask)
        for index in indices:
            self._target_managers[int(index)].reset()
            self._previous_actions[int(index)] = 0.0

    def save_model(self, path: str | None = None, id: str = "1") -> None:
        destination = Path(path or ".") / f"model.ckpt-{id}.pkl"
        save_checkpoint(
            destination,
            model=self.model,
            model_config=self.model_config,
            optimizer=self.algorithm.optimizer,
        )
        if self.logger:
            self.logger.info("saved v2 checkpoint to %s", destination)

    def load_model(self, path: str | None = None, id: str = "1") -> None:
        source = Path(path or ".") / f"model.ckpt-{id}.pkl"
        load_checkpoint(
            source,
            model=self.model,
            model_config=self.model_config,
            optimizer=self.algorithm.optimizer,
        )
        if self.logger:
            self.logger.info("loaded v2 checkpoint from %s", source)


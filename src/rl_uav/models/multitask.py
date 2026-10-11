"""Shared-actor, multi-critic architecture for navigation and hover."""

from __future__ import annotations

import torch

from rl_uav.config import ModelConfig
from rl_uav.models.common import SquashedGaussianPolicy, mlp
from rl_uav.types import ObservationBatch, TaskId


class MultiTaskActorCritic(SquashedGaussianPolicy):
    """Paper-aligned MTRL model adapted to sequential ObstacleHover phases."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig()
        super().__init__(config)
        hidden = config.encoder_hidden_dim
        latent = config.latent_dim
        self.shared_encoder = mlp([config.shared_obs_dim, hidden, hidden, latent])
        self.task_encoders = torch.nn.ModuleDict(
            {
                str(int(TaskId.NAVIGATION)): mlp(
                    [config.navigation_obs_dim, hidden, hidden, latent]
                ),
                str(int(TaskId.HOVER)): mlp([config.hover_obs_dim, hidden, hidden, latent]),
            }
        )
        self.actor = mlp(
            [2 * latent, config.actor_hidden_dim, config.actor_hidden_dim, config.action_dim],
            final_gain=0.01,
        )
        self.critics = torch.nn.ModuleDict(
            {
                str(int(TaskId.NAVIGATION)): mlp(
                    [
                        config.shared_obs_dim + config.navigation_obs_dim,
                        config.critic_hidden_dim,
                        config.critic_hidden_dim,
                        1,
                    ]
                ),
                str(int(TaskId.HOVER)): mlp(
                    [
                        config.shared_obs_dim + config.hover_obs_dim,
                        config.critic_hidden_dim,
                        config.critic_hidden_dim,
                        1,
                    ]
                ),
            }
        )

    def _policy_mean(self, observations: ObservationBatch) -> torch.Tensor:
        observations.validate_shapes()
        shared_latent = self.shared_encoder(observations.shared_obs)
        task_latent = torch.zeros_like(shared_latent)
        for task, dimension in (
            (TaskId.NAVIGATION, self.config.navigation_obs_dim),
            (TaskId.HOVER, self.config.hover_obs_dim),
        ):
            mask = observations.task_id.long() == int(task)
            if mask.any():
                task_latent[mask] = self.task_encoders[str(int(task))](
                    observations.task_obs[mask, :dimension]
                )
        return self.actor(torch.cat((shared_latent, task_latent), dim=-1))

    def values(self, observations: ObservationBatch) -> torch.Tensor:
        observations.validate_shapes()
        result = torch.zeros(
            observations.shared_obs.shape[0],
            dtype=observations.shared_obs.dtype,
            device=observations.shared_obs.device,
        )
        for task, dimension in (
            (TaskId.NAVIGATION, self.config.navigation_obs_dim),
            (TaskId.HOVER, self.config.hover_obs_dim),
        ):
            mask = observations.task_id.long() == int(task)
            if mask.any():
                critic_input = torch.cat(
                    (observations.shared_obs[mask], observations.task_obs[mask, :dimension]), dim=-1
                )
                result[mask] = self.critics[str(int(task))](critic_input).squeeze(-1)
        return result

    def act(
        self, observations: ObservationBatch, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self._policy_mean(observations)
        actions, log_probs, _ = self._sample(mean, deterministic=deterministic)
        return actions, self.values(observations), log_probs

    def evaluate_actions(
        self, observations: ObservationBatch, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self._policy_mean(observations)
        log_probs, entropy = self._evaluate_distribution(mean, actions)
        return self.values(observations), log_probs, entropy

    def forward(
        self, observations: ObservationBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.act(observations)


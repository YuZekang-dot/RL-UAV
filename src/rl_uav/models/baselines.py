"""Fair flat-policy and fully specialized PPO baselines."""

from __future__ import annotations

import torch
from torch.nn import functional as functional

from rl_uav.config import ModelConfig
from rl_uav.models.common import SquashedGaussianPolicy, mlp
from rl_uav.types import ObservationBatch, TaskId


class FlatActorCritic(SquashedGaussianPolicy):
    """One actor and one critic over padded task features plus phase one-hot."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig()
        super().__init__(config)
        input_dim = config.shared_obs_dim + config.max_task_obs_dim + 2
        self.actor = mlp(
            [input_dim, config.actor_hidden_dim, config.actor_hidden_dim, config.action_dim],
            final_gain=0.01,
        )
        self.critic = mlp(
            [input_dim, config.critic_hidden_dim, config.critic_hidden_dim, 1]
        )

    def _inputs(self, observations: ObservationBatch) -> torch.Tensor:
        observations.validate_shapes()
        task_one_hot = functional.one_hot(observations.task_id.long(), num_classes=2).to(
            observations.shared_obs.dtype
        )
        return torch.cat(
            (
                observations.shared_obs,
                observations.task_obs * observations.task_obs_mask,
                task_one_hot,
            ),
            dim=-1,
        )

    def act(
        self, observations: ObservationBatch, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._inputs(observations)
        actions, log_probs, _ = self._sample(self.actor(inputs), deterministic=deterministic)
        return actions, self.critic(inputs).squeeze(-1), log_probs

    def evaluate_actions(
        self, observations: ObservationBatch, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._inputs(observations)
        log_probs, entropy = self._evaluate_distribution(self.actor(inputs), actions)
        return self.critic(inputs).squeeze(-1), log_probs, entropy

    def forward(
        self, observations: ObservationBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.act(observations)


class SeparateActorCritic(SquashedGaussianPolicy):
    """Independent navigation/hover policies selected by phase."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig()
        super().__init__(config)
        self.actors = torch.nn.ModuleDict()
        self.critics = torch.nn.ModuleDict()
        for task, dimension in (
            (TaskId.NAVIGATION, config.navigation_obs_dim),
            (TaskId.HOVER, config.hover_obs_dim),
        ):
            input_dim = config.shared_obs_dim + dimension
            self.actors[str(int(task))] = mlp(
                [input_dim, config.actor_hidden_dim, config.actor_hidden_dim, config.action_dim],
                final_gain=0.01,
            )
            self.critics[str(int(task))] = mlp(
                [input_dim, config.critic_hidden_dim, config.critic_hidden_dim, 1]
            )

    def _means_and_values(
        self, observations: ObservationBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations.validate_shapes()
        batch = observations.shared_obs.shape[0]
        means = torch.zeros(
            (batch, self.config.action_dim),
            dtype=observations.shared_obs.dtype,
            device=observations.shared_obs.device,
        )
        values = torch.zeros(
            batch, dtype=observations.shared_obs.dtype, device=observations.shared_obs.device
        )
        for task, dimension in (
            (TaskId.NAVIGATION, self.config.navigation_obs_dim),
            (TaskId.HOVER, self.config.hover_obs_dim),
        ):
            mask = observations.task_id.long() == int(task)
            if mask.any():
                inputs = torch.cat(
                    (observations.shared_obs[mask], observations.task_obs[mask, :dimension]), dim=-1
                )
                means[mask] = self.actors[str(int(task))](inputs)
                values[mask] = self.critics[str(int(task))](inputs).squeeze(-1)
        return means, values

    def act(
        self, observations: ObservationBatch, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means, values = self._means_and_values(observations)
        actions, log_probs, _ = self._sample(means, deterministic=deterministic)
        return actions, values, log_probs

    def evaluate_actions(
        self, observations: ObservationBatch, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means, values = self._means_and_values(observations)
        log_probs, entropy = self._evaluate_distribution(means, actions)
        return values, log_probs, entropy

    def forward(
        self, observations: ObservationBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.act(observations)


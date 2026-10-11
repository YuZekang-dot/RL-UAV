"""Shared PyTorch utilities for tanh-squashed continuous policies."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from rl_uav.config import ModelConfig


def mlp(dimensions: list[int], *, final_gain: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        linear = nn.Linear(input_dim, output_dim)
        gain = final_gain if index == len(dimensions) - 2 else math.sqrt(2.0)
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class SquashedGaussianPolicy(nn.Module):
    """Base class exposing a consistent action distribution API."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), math.log(config.init_action_std), dtype=torch.float32)
        )

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = torch.exp(self.log_std).clamp(self.config.min_action_std, self.config.max_action_std)
        return Normal(mean, std.expand_as(mean))

    def _log_prob(
        self, distribution: Normal, raw_actions: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        correction = torch.log(1.0 - actions.square() + self.config.tanh_epsilon).sum(dim=-1)
        return distribution.log_prob(raw_actions).sum(dim=-1) - correction

    def _atanh(self, actions: torch.Tensor) -> torch.Tensor:
        epsilon = self.config.tanh_epsilon
        actions = actions.clamp(-1.0 + epsilon, 1.0 - epsilon)
        return 0.5 * (torch.log1p(actions) - torch.log1p(-actions))

    def _sample(
        self, mean: torch.Tensor, *, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(mean)
        raw_actions = mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probs = self._log_prob(distribution, raw_actions, actions)
        entropy = distribution.entropy().sum(dim=-1)
        return actions, log_probs, entropy

    def _evaluate_distribution(
        self, mean: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(mean)
        raw_actions = self._atanh(actions)
        log_probs = self._log_prob(distribution, raw_actions, actions)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy


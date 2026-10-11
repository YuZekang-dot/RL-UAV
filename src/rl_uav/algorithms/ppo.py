"""Task-balanced PPO with phase-boundary-aware GAE."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from rl_uav.types import ObservationBatch, TaskId


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 5.0e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.10
    value_loss_coefficient: float = 0.50
    entropy_coefficient: float = 0.001
    max_gradient_norm: float = 10.0
    epochs: int = 4
    minibatches: int = 16


def compute_task_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    task_ids: torch.Tensor,
    next_values: torch.Tensor,
    next_task_ids: torch.Tensor,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without bootstrapping across a navigation→hover boundary."""
    if rewards.shape != values.shape or rewards.shape != dones.shape or rewards.shape != task_ids.shape:
        raise ValueError("rewards, values, dones, and task_ids must have identical [T, N] shapes")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_values)
    for step in reversed(range(rewards.shape[0])):
        following_values = next_values if step == rewards.shape[0] - 1 else values[step + 1]
        following_tasks = next_task_ids if step == rewards.shape[0] - 1 else task_ids[step + 1]
        same_task = task_ids[step] == following_tasks
        bootstrap = (~dones[step].bool()) & same_task
        nonterminal = bootstrap.to(rewards.dtype)
        delta = rewards[step] + gamma * following_values * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    return advantages + values, advantages


def _index_observations(observations: ObservationBatch, indices: torch.Tensor) -> ObservationBatch:
    return ObservationBatch(
        shared_obs=observations.shared_obs[indices],
        task_obs=observations.task_obs[indices],
        task_id=observations.task_id[indices],
        task_obs_mask=observations.task_obs_mask[indices],
    )


def _task_balanced_mean(values: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    task_means = [values[task_ids == int(task)].mean() for task in TaskId if (task_ids == int(task)).any()]
    if not task_means:
        raise ValueError("at least one task must be present")
    return torch.stack(task_means).mean()


def _normalize_advantages(advantages: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    normalized = torch.zeros_like(advantages)
    for task in TaskId:
        mask = task_ids == int(task)
        if mask.any():
            task_advantages = advantages[mask]
            normalized[mask] = (task_advantages - task_advantages.mean()) / (
                task_advantages.std(unbiased=False) + 1.0e-8
            )
    return normalized


def _stratified_batches(
    task_ids: torch.Tensor,
    batch_size: int,
    batch_count: int,
) -> Iterable[torch.Tensor]:
    groups = [torch.flatnonzero(task_ids == int(task)) for task in TaskId]
    groups = [group for group in groups if group.numel()]
    per_task = max(1, batch_size // len(groups))
    for _ in range(batch_count):
        selected = []
        for group in groups:
            if group.numel() >= per_task:
                selected.append(group[torch.randperm(group.numel(), device=group.device)[:per_task]])
            else:
                selected.append(group[torch.randint(group.numel(), (per_task,), device=group.device)])
        indices = torch.cat(selected)
        yield indices[torch.randperm(indices.numel(), device=indices.device)]


class MultiTaskPPO:
    """PPO optimizer compatible with multi-task and both baseline model APIs."""

    def __init__(self, model: torch.nn.Module, config: PPOConfig | None = None) -> None:
        self.model = model
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    def update(
        self,
        observations: ObservationBatch,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float | list[int]]:
        observations.validate_shapes()
        task_ids = observations.task_id.long()
        advantages = _normalize_advantages(advantages, task_ids)
        sample_count = actions.shape[0]
        batch_size = max(2, math.ceil(sample_count / self.config.minibatches))
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        updates = 0

        for _ in range(self.config.epochs):
            for indices in _stratified_batches(task_ids, batch_size, self.config.minibatches):
                batch_observations = _index_observations(observations, indices)
                batch_tasks = task_ids[indices]
                values, new_log_probs, entropy = self.model.evaluate_actions(
                    batch_observations, actions[indices]
                )
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[indices]
                clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[indices]
                policy_loss = _task_balanced_mean(-torch.minimum(unclipped, clipped), batch_tasks)
                value_loss = _task_balanced_mean((returns[indices] - values).square(), batch_tasks)
                entropy_mean = _task_balanced_mean(entropy, batch_tasks)
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                for name, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy_mean),
                    ("total_loss", loss),
                    ("approx_kl", approximate_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    sums[name] += float(value.detach().cpu())
                updates += 1

        present = sorted(int(value) for value in torch.unique(task_ids).detach().cpu().tolist())
        return {**{key: value / max(updates, 1) for key, value in sums.items()}, "tasks_present": present}


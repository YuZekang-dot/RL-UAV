"""Versioned checkpoints that reject accidental legacy-model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rl_uav.config import ModelConfig


CHECKPOINT_FORMAT = "rl-uav-checkpoint"
CHECKPOINT_VERSION = 2
OBSERVATION_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer | None = None,
    curriculum_state: dict[str, Any] | None = None,
    normalization_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "model_config": model_config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "curriculum_state": curriculum_state or {},
        "normalization_state": normalization_state or {},
        "metadata": metadata or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer | None = None,
    allow_legacy_baseline: bool = False,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        if allow_legacy_baseline:
            state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            model.load_state_dict(state)
            return {"legacy": True}
        raise ValueError("legacy checkpoint rejected; load it through a baseline adapter explicitly")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    if payload.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("checkpoint observation schema does not match this repository")
    if payload.get("model_config") != model_config.to_dict():
        raise ValueError("checkpoint model configuration does not match the requested model")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


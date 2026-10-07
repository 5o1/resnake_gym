"""Validated loading for V-trace policy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from resnake_gym.gamepad_runtime import build_model
from resnake_gym.gamepad_vtrace_contract import (
    CHECKPOINT_FORMAT as VTRACE_CHECKPOINT_FORMAT,
)
from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.gamepad_vtrace_contract import (
    validate_checkpoint as validate_vtrace_checkpoint,
)


def config_from_checkpoint(payload: dict[str, Any]) -> VTraceConfig:
    """Validate V-trace semantics and construct its config."""
    if payload.get("format") != VTRACE_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not a V-trace policy")
    validate_vtrace_checkpoint(payload)
    return VTraceConfig(**payload["config"])


def load_policy_config(path: Path) -> tuple[dict[str, Any], VTraceConfig]:
    """Read and validate a lightweight inference or full checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload, config_from_checkpoint(payload)


def load_policy_model(
    path: Path,
    device,
) -> tuple[dict[str, Any], VTraceConfig, Any]:
    """Read one V-trace checkpoint and construct its model."""

    payload, config = load_policy_config(path)
    model = build_model(config).to(device).eval()
    model.load_state_dict(payload["model"])
    return payload, config, model


__all__ = [
    "config_from_checkpoint",
    "load_policy_config",
    "load_policy_model",
]

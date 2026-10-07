"""Validated loading for PPO and V-trace policy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from resnake_gym.gamepad_ppo_contract import (
    PPOConfig,
    normalize_config_payload,
    validate_policy_checkpoint,
)
from resnake_gym.gamepad_runtime import build_model
from resnake_gym.gamepad_vtrace_contract import (
    CHECKPOINT_FORMAT as VTRACE_CHECKPOINT_FORMAT,
)
from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.gamepad_vtrace_contract import (
    validate_checkpoint as validate_vtrace_checkpoint,
)

PolicyConfig = PPOConfig | VTraceConfig


def config_from_checkpoint(payload: dict[str, Any]) -> PolicyConfig:
    """Validate checkpoint semantics and construct its matching config type."""

    if payload.get("format") == VTRACE_CHECKPOINT_FORMAT:
        validate_vtrace_checkpoint(payload)
        return VTraceConfig(**payload["config"])
    validate_policy_checkpoint(payload)
    config_payload = normalize_config_payload(payload["config"])
    return PPOConfig(**config_payload)


def load_policy_config(path: Path) -> tuple[dict[str, Any], PolicyConfig]:
    """Read and validate a lightweight inference or full checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload, config_from_checkpoint(payload)


def load_policy_model(
    path: Path,
    device,
    *,
    expected_config_type: type[PolicyConfig] | None = None,
) -> tuple[dict[str, Any], PolicyConfig, Any]:
    """Read one checkpoint and construct the exact model required by its config.

    ``expected_config_type`` lets algorithm-specific adapters retain a narrow
    checkpoint contract while sharing this loading path.  The family check is
    deliberately performed before model construction and weight loading.
    """

    payload, config = load_policy_config(path)
    if expected_config_type is not None and not isinstance(
        config, expected_config_type
    ):
        raise ValueError(
            f"checkpoint config is {type(config).__name__}; "
            f"expected {expected_config_type.__name__}"
        )
    model = build_model(config).to(device).eval()
    model.load_state_dict(payload["model"])
    return payload, config, model


__all__ = [
    "PolicyConfig",
    "config_from_checkpoint",
    "load_policy_config",
    "load_policy_model",
]

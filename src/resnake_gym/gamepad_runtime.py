"""Algorithm-independent construction helpers for gamepad policies and tasks.

The PPO and V-trace trainers intentionally share the same observation batching,
model construction, and environment construction semantics.  Keeping those
operations here prevents either training algorithm from becoming infrastructure
for the other one.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from resnake_gym.envs.reward_gamepad import RewardGamepadEnv
from resnake_gym.evaluation_protocol import DETERMINISTIC_SOLVER_BUDGET
from resnake_gym.models import GamepadPolicy
from resnake_gym.wrappers.gamepad_async import AsyncGamepad


class GamepadModelConfig(Protocol):
    """Configuration fields required to construct a gamepad policy."""

    dim: int
    chunk_length: int
    decoder: str
    spatial_pool: bool
    chunk_rho: float | None
    action_head: str


class GamepadEnvironmentConfig(Protocol):
    """Configuration fields required to construct the shared training task."""

    width: int
    height: int
    initial_length: int
    logic_fps: float
    chunk_length: int
    decision_min: int
    decision_max: int
    observation_delay_max: int
    command_delay_max: int
    drop_probability: float
    max_logic_steps: int
    gamma: float
    perturbation_min: int
    perturbation_max: int
    random_rotation: bool
    solver_ms: float
    shaping_scale: float
    death_cost: float


def build_model(config: GamepadModelConfig) -> GamepadPolicy:
    """Build the policy architecture shared by PPO and V-trace."""
    return GamepadPolicy(
        config.dim,
        config.chunk_length,
        channels=14,
        time_features=True,
        decoder=config.decoder,
        spatial_pool=config.spatial_pool,
        chunk_rho=config.chunk_rho,
        action_head=config.action_head,
    )


def make_env(
    config: GamepadEnvironmentConfig, *, deterministic_solver: bool = False
) -> AsyncGamepad:
    """Build the gamepad task without depending on a training algorithm."""
    solver_budget = (
        DETERMINISTIC_SOLVER_BUDGET
        if deterministic_solver
        else {"time_limit_ms": config.solver_ms}
    )
    return AsyncGamepad(
        RewardGamepadEnv(
            reward_gamma=config.gamma,
            shaping_scale=config.shaping_scale,
            death_cost=config.death_cost,
            width=config.width,
            height=config.height,
            initial_length=config.initial_length,
            logic_fps=config.logic_fps,
            max_logic_steps=config.max_logic_steps,
            perturbation_interval=(config.perturbation_min, config.perturbation_max),
            random_rotation=config.random_rotation,
            solver_budget=solver_budget,
        ),
        chunk_length=config.chunk_length,
        decision_ticks=(config.decision_min, config.decision_max),
        observation_delay=(0, config.observation_delay_max),
        command_delay=(0, config.command_delay_max),
        frame_drop_probability=config.drop_probability,
        gamma=config.gamma,
        time_features=True,
    )


def stack_observations(
    observations: Sequence[Mapping[str, Any]], device: torch.device | str
) -> dict[str, Tensor]:
    """Stack a non-empty sequence of environment observations on ``device``."""
    return {
        key: torch.as_tensor(
            np.stack([observation[key] for observation in observations]),
            device=device,
            dtype=torch.float32,
        )
        for key in observations[0]
    }


__all__ = [
    "GamepadEnvironmentConfig",
    "GamepadModelConfig",
    "build_model",
    "make_env",
    "stack_observations",
]

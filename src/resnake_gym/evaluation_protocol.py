"""Canonical, wall-time-independent protocol fields for policy evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from resnake_gym.envs.perturbed_gamepad import (
    OBSTACLE_SIZE_FILTER,
    SCENE_VERSION,
    SYNC_OBSTACLE_SAMPLER,
)
from resnake_gym.gamepad import LAYOUT_VERSION

DETERMINISTIC_SOLVER_BUDGET = {
    "cycle_nodes": 2000,
    "state_limit": 5000,
    "exact_cell_limit": 16,
    "time_limit_ms": None,
}


def frozen_gamepad_evaluation_request(
    *,
    sizes: Sequence[str],
    episodes_per_size: int,
    base_seed: int,
    stochastic: bool,
) -> dict:
    return {
        "version": "frozen-gamepad-eval-request-v1",
        "sizes": list(sizes),
        "episodes_per_size": episodes_per_size,
        "base_seed": base_seed,
        "stochastic": stochastic,
        "deterministic_solver_budget": dict(DETERMINISTIC_SOLVER_BUDGET),
    }


def gamepad_environment_protocol_fields(config, *, random_rotation: bool) -> dict:
    """Return shared environment semantics for evaluation and live preview."""

    return {
        "initial_snake_length": config.initial_length,
        "logic_fps": config.logic_fps,
        "max_logic_steps": config.max_logic_steps,
        "chunk_length": config.chunk_length,
        "decision_ticks": [config.decision_min, config.decision_max],
        "observation_delay": [0, config.observation_delay_max],
        "command_delay": [0, config.command_delay_max],
        "frame_drop_probability": config.drop_probability,
        "perturbation_interval": [
            config.perturbation_min,
            config.perturbation_max,
        ],
        "random_rotation": random_rotation,
        "obstacle_probability": 0.5,
        "notice_ticks": 6,
        "protected_ticks": 3,
        "max_obstacle_cells": 32,
        "obstacle_batch_size": [1, 6],
        "stick_deadzone": 0.2,
        "reward_gamma": config.gamma,
        "shaping_scale": config.shaping_scale,
        "death_cost": config.death_cost,
        "time_features": True,
        "scene_version": SCENE_VERSION,
        "gamepad_layout": LAYOUT_VERSION,
        "obstacle_sampler": SYNC_OBSTACLE_SAMPLER,
        "obstacle_size_filter": OBSTACLE_SIZE_FILTER,
        "solver_mode": "sync",
    }


def frozen_gamepad_evaluation_protocol(
    config,
    *,
    sizes: Sequence[str],
    episodes_per_size: int,
    base_seed: int,
    stochastic: bool,
) -> dict:
    """Describe every effective environment field used by the evaluator."""
    request = frozen_gamepad_evaluation_request(
        sizes=sizes,
        episodes_per_size=episodes_per_size,
        base_seed=base_seed,
        stochastic=stochastic,
    )
    return {
        **request,
        "version": "frozen-gamepad-eval-v2",
        **gamepad_environment_protocol_fields(
            config,
            random_rotation=config.random_rotation,
        ),
    }


def protocol_sha256(protocol: dict) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "DETERMINISTIC_SOLVER_BUDGET",
    "frozen_gamepad_evaluation_request",
    "frozen_gamepad_evaluation_protocol",
    "gamepad_environment_protocol_fields",
    "protocol_sha256",
]

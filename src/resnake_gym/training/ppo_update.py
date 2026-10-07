"""Internal data flow for one recurrent PPO update.

The policy and environment interactions stay in :mod:`resnake_gym.gamepad_ppo`.
This module owns the phase-boundary records and the algorithm-only preparation
of GAE targets and recurrent windows.  Keeping these boundaries explicit makes
it possible to test the bookkeeping without changing the serialized trainer
state or the order in which random samples are drawn.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """Transitions and audit counters produced by one collection phase."""

    initial_hidden: torch.Tensor
    data: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    food_events: list[dict[str, Any]] = field(default_factory=list)
    no_food_ages: list[int] = field(default_factory=list)
    terminals: int = 0
    truncations: int = 0
    completed_food_episodes: int = 0
    fragments: list[dict[str, Any]] = field(default_factory=list)
    rollout_logic_ticks: int = 0
    expired_reports: int = 0
    perturbation_events: int = 0
    neutral_reports: int = 0
    executed_reports: int = 0
    sampled_direction_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.int64)
    )
    sampled_conflict_neutral: int = 0
    sampled_dpad_active: int = 0
    stop_reason: str | None = None


@dataclass(frozen=True)
class PreparedRollout:
    """GAE tensors and verified recurrent learn windows."""

    advantages: torch.Tensor
    targets: torch.Tensor
    old_logp: torch.Tensor
    resets: torch.Tensor
    windows: list[dict[str, int]]
    total_samples: int
    recurrent_samples: int
    reset_crossings: int
    max_learn_steps: int
    max_burn_in_steps: int
    burn_in_steps: int


@dataclass(frozen=True)
class PPOOptimization:
    """Scalar outputs of the on-policy optimization phase."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    latent_entropy: torch.Tensor
    approx_kl: torch.Tensor
    grad_norm: torch.Tensor
    epochs_done: int
    burn_in_grad_enabled: bool
    learn_grad_enabled: bool


def generalized_advantages(
    rewards, values, next_values, discounts, terminated, truncated, gae_lambda
):
    """Bootstrap time limits, never bootstrap death; never cross a reset.

    Rewards already sum ``gamma**j * r_j`` within each decision and discounts
    are ``gamma**K``. Lambda is a per-decision trace coefficient.
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + discounts[step] * next_values[step] * (~terminated[step])
            - values[step]
        )
        carry = (
            delta
            + discounts[step]
            * gae_lambda
            * (~(terminated[step] | truncated[step]))
            * carry
        )
        advantages[step] = carry
    return advantages


def recurrent_windows(resets, burn_in, unroll):
    """Partition a rollout into episode-local, per-environment learn windows."""
    if resets.ndim != 2:
        raise ValueError("reset tensor must have time and environment dimensions")
    if burn_in < 0 or unroll < 1:
        raise ValueError("invalid recurrent window lengths")
    steps, num_envs = resets.shape
    windows = []
    for env in range(num_envs):
        episode_starts = [0]
        episode_starts.extend(
            step for step in range(1, steps) if bool(resets[step, env].item())
        )
        episode_starts.append(steps)
        for episode, (start, end) in enumerate(
            zip(episode_starts[:-1], episode_starts[1:], strict=True)
        ):
            for learn_start in range(start, end, unroll):
                learn_end = min(learn_start + unroll, end)
                windows.append(
                    {
                        "env": env,
                        "episode": episode,
                        "burn_start": max(start, learn_start - burn_in),
                        "learn_start": learn_start,
                        "learn_end": learn_end,
                    }
                )
    return windows


def stack_rollout(data, key):
    """Stack one tensor field from ordered rollout rows."""
    return torch.stack([row[key] for row in data])


def prepare_rollout(data, config) -> PreparedRollout:
    """Build normalized GAE targets and validate recurrent sample coverage."""
    old_values = stack_rollout(data, "value")
    advantages = generalized_advantages(
        stack_rollout(data, "reward"),
        old_values,
        stack_rollout(data, "next_value"),
        stack_rollout(data, "discount"),
        stack_rollout(data, "terminated"),
        stack_rollout(data, "truncated"),
        config.gae_lambda,
    )
    targets = advantages + old_values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    old_logp = stack_rollout(data, "logp")

    # Window construction is control flow; copy masks once instead of
    # synchronizing an accelerator for every reset check.
    resets = stack_rollout(data, "reset").detach().cpu()
    windows = recurrent_windows(
        resets, config.recurrent_burn_in, config.recurrent_unroll
    )
    total_samples = len(data) * config.num_envs
    recurrent_samples = sum(
        window["learn_end"] - window["learn_start"] for window in windows
    )
    if recurrent_samples != total_samples:
        raise RuntimeError("recurrent windows did not preserve every PPO sample")
    reset_crossings = sum(
        int(
            resets[window["burn_start"] + 1 : window["learn_end"], window["env"]]
            .any()
            .item()
        )
        for window in windows
    )
    if reset_crossings:
        raise RuntimeError("recurrent window crossed an episode reset")
    max_learn_steps = max(
        (window["learn_end"] - window["learn_start"] for window in windows),
        default=0,
    )
    max_burn_in_steps = max(
        (window["learn_start"] - window["burn_start"] for window in windows),
        default=0,
    )
    burn_in_steps = sum(
        window["learn_start"] - window["burn_start"] for window in windows
    )
    return PreparedRollout(
        advantages=advantages,
        targets=targets,
        old_logp=old_logp,
        resets=resets,
        windows=windows,
        total_samples=total_samples,
        recurrent_samples=recurrent_samples,
        reset_crossings=reset_crossings,
        max_learn_steps=max_learn_steps,
        max_burn_in_steps=max_burn_in_steps,
        burn_in_steps=burn_in_steps,
    )


__all__ = [
    "PPOOptimization",
    "PreparedRollout",
    "RolloutBatch",
    "generalized_advantages",
    "prepare_rollout",
    "recurrent_windows",
    "stack_rollout",
]

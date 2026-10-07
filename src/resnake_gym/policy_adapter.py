"""Deterministic runtime adapter for supported gamepad policy checkpoints."""

import os

import torch

from resnake_gym.gamepad_runtime import stack_observations
from resnake_gym.models import (
    distribution_from_observation,
    reports_from_policy_action,
    sample_policy_action,
)
from resnake_gym.policy_checkpoint import load_policy_model


class PolicyAdapter:
    """Expose either PPO/SIL or V-trace weights through the realtime API."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        _, self.config, self.model = load_policy_model(checkpoint, device)
        self.chunk_length = self.config.chunk_length
        self.requires_timing_v2 = True
        self.hidden = None

    def reset(self, *, seed=None):
        self.hidden = None

    @torch.no_grad()
    def act(self, observation):
        distribution, _, self.hidden = distribution_from_observation(
            self.model, stack_observations([observation], self.device), self.hidden
        )
        return (
            reports_from_policy_action(
                distribution,
                sample_policy_action(distribution, stochastic=False),
                continuous_buttons=getattr(self.config, "chunk_rho", None) is not None,
            )[0]
            .cpu()
            .numpy()
        )

    def fork(self):
        copy = object.__new__(type(self))
        copy.device, copy.config, copy.model = self.device, self.config, self.model
        copy.chunk_length = self.chunk_length
        copy.requires_timing_v2 = self.requires_timing_v2
        copy.hidden = None if self.hidden is None else self.hidden.clone()
        return copy


def from_environment():
    return PolicyAdapter(
        os.environ["RESNAKE_CHECKPOINT"],
        os.environ.get("RESNAKE_DEVICE", "cpu"),
    )


__all__ = ["PolicyAdapter", "from_environment"]

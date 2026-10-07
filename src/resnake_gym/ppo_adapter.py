"""Explicit PPO checkpoint adapter; deterministic evaluation, no action repair."""

import os

import torch

from resnake_gym.gamepad_ppo_contract import PPOConfig
from resnake_gym.gamepad_runtime import stack_observations
from resnake_gym.models import (
    distribution_from_observation,
    reports_from_policy_action,
    sample_policy_action,
)
from resnake_gym.policy_checkpoint import load_policy_model


class PPOAdapter:
    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        _, self.config, self.model = load_policy_model(
            checkpoint,
            device,
            expected_config_type=PPOConfig,
        )
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
                continuous_buttons=self.config.chunk_rho is not None,
            )[0]
            .cpu()
            .numpy()
        )

    def fork(self):
        copy = object.__new__(type(self))
        copy.device, copy.config, copy.model = self.device, self.config, self.model
        copy.hidden = None if self.hidden is None else self.hidden.clone()
        return copy


def from_environment():
    return PPOAdapter(
        os.environ["RESNAKE_CHECKPOINT"], os.environ.get("RESNAKE_DEVICE", "cpu")
    )

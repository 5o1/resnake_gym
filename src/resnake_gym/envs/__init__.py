"""Environment implementations shipped by :mod:`resnake_gym`."""

from resnake_gym.envs.gamepad_env import GamepadSnakeEnv
from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv
from resnake_gym.envs.reward_gamepad import RewardGamepadEnv
from resnake_gym.envs.snake_env import SnakeEnv as LegacySnakeEnv
from resnake_gym.envs.snake_env import SnakeState

# Backward-compatible name for v0 scripts.  New experiments should select one
# of the gamepad-only environments above.
SnakeEnv = LegacySnakeEnv

__all__ = [
    "GamepadSnakeEnv",
    "PerturbedGamepadEnv",
    "RewardGamepadEnv",
    "SnakeState",
    "LegacySnakeEnv",
    "SnakeEnv",
]

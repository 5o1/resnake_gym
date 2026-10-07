"""ReSnake Gymnasium environments."""

from gymnasium.envs.registration import register, registry

ENV_ID = "resnake_gym/ReSnake-v1"
PERTURBED_ENV_ID = "resnake_gym/ReSnakePerturbed-v1"

if PERTURBED_ENV_ID not in registry:
    register(
        id=PERTURBED_ENV_ID,
        entry_point="resnake_gym.envs.perturbed_gamepad:PerturbedGamepadEnv",
    )

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="resnake_gym.envs.gamepad_env:GamepadSnakeEnv",
    )


# Frozen for reproducing historical experiments, not for new training.
if "resnake_gym/ReSnake-v0" not in registry:
    register(id="resnake_gym/ReSnake-v0", entry_point="resnake_gym.envs:SnakeEnv")

from resnake_gym.envs import (  # noqa: E402 -- register IDs before public imports
    GamepadSnakeEnv,
    LegacySnakeEnv,
    PerturbedGamepadEnv,
    RewardGamepadEnv,
    SnakeState,
)
from resnake_gym.wrappers import AsyncGamepad  # noqa: E402

__all__ = [
    "AsyncGamepad",
    "ENV_ID",
    "GamepadSnakeEnv",
    "LegacySnakeEnv",
    "PERTURBED_ENV_ID",
    "PerturbedGamepadEnv",
    "RewardGamepadEnv",
    "SnakeState",
]

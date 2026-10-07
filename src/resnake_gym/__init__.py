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

from resnake_gym.envs import (  # noqa: E402 -- register IDs before public imports
    GamepadSnakeEnv,
    PerturbedGamepadEnv,
    RewardGamepadEnv,
    SnakeEnv,
    SnakeState,
)
from resnake_gym.wrappers import AsyncGamepad  # noqa: E402

__all__ = [
    "AsyncGamepad",
    "ENV_ID",
    "GamepadSnakeEnv",
    "PERTURBED_ENV_ID",
    "PerturbedGamepadEnv",
    "RewardGamepadEnv",
    "SnakeEnv",
    "SnakeState",
]

"""Historical optional reward shaping based on distance to food."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from resnake_gym.envs.snake_env import SnakeEnv


class DistanceRewardWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Add scaled Manhattan-distance progress to each transition's reward.

    The shaping term is ``scale * (distance_before - distance_after)``. It is
    omitted when the food changes during the transition, because the two
    distances would refer to different targets, and on terminating transitions,
    where a collision must not look like progress. Time-limit truncations are
    still eligible because the snake completed a valid move.
    """

    def __init__(
        self,
        env: gym.Env[np.ndarray, int],
        *,
        scale: float = 0.0,
    ) -> None:
        super().__init__(env)
        scale = float(scale)
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        if not isinstance(self.unwrapped, SnakeEnv):
            raise TypeError("DistanceRewardWrapper requires a SnakeEnv")
        self.scale = scale

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        snake_env = self.unwrapped
        assert isinstance(snake_env, SnakeEnv)
        head_before = snake_env.head
        food_before = snake_env.food

        observation, reward, terminated, truncated, info = self.env.step(action)

        if not terminated and food_before is not None and snake_env.food == food_before:
            distance_before = self._manhattan_distance(head_before, food_before)
            distance_after = self._manhattan_distance(snake_env.head, food_before)
            reward += self.scale * (distance_before - distance_after)

        return observation, float(reward), terminated, truncated, info

    @staticmethod
    def _manhattan_distance(first: tuple[int, int], second: tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])


__all__ = ["DistanceRewardWrapper"]

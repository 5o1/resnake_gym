"""Tests for optional ReSnake Gymnasium wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from resnake_gym.envs import SnakeEnv
from resnake_gym.wrappers import DistanceRewardWrapper

RIGHT = 1
UP = 0


def test_distance_reward_is_positive_when_closer_and_negative_when_farther() -> None:
    env = DistanceRewardWrapper(
        SnakeEnv(width=8, height=6, food_reward=0.0, death_reward=0.0),
        scale=0.5,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (6, 2),
            },
        )

        _, closer_reward, terminated, truncated, _ = env.step(RIGHT)
        assert closer_reward == pytest.approx(0.5)
        assert not terminated
        assert not truncated

        _, farther_reward, terminated, truncated, _ = env.step(UP)
        assert farther_reward == pytest.approx(-0.5)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_distance_reward_skips_transition_that_eats_and_replaces_food() -> None:
    env = DistanceRewardWrapper(
        SnakeEnv(width=8, height=6, food_reward=2.0),
        scale=10.0,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (4, 2),
            },
        )
        _, reward, terminated, truncated, info = env.step(RIGHT)

        assert reward == pytest.approx(2.0)
        assert info["score"] == 1
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_distance_reward_composes_with_internal_timeout_reward() -> None:
    env = DistanceRewardWrapper(
        SnakeEnv(
            width=8,
            height=6,
            max_logic_steps=1,
            timeout_reward=4.0,
        ),
        scale=0.5,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (6, 2),
            },
        )
        _, reward, terminated, truncated, info = env.step(RIGHT)

        assert reward == pytest.approx(4.5)
        assert not terminated
        assert truncated
        assert info["termination_reason"] == "time_limit"
    finally:
        env.close()


def test_distance_reward_does_not_change_collision_reward() -> None:
    env = DistanceRewardWrapper(
        SnakeEnv(width=6, height=6, death_reward=-2.0),
        scale=10.0,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(0, 2), (1, 2), (2, 2)],
                "direction": "left",
                "food": (5, 5),
            },
        )
        _, reward, terminated, truncated, _ = env.step(3)

        assert reward == pytest.approx(-2.0)
        assert terminated
        assert not truncated
    finally:
        env.close()


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf])
def test_distance_reward_scale_must_be_finite(scale: float) -> None:
    env = SnakeEnv()
    try:
        with pytest.raises(ValueError, match="scale must be finite"):
            DistanceRewardWrapper(env, scale=scale)
    finally:
        env.close()

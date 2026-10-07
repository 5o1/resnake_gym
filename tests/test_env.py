"""Black-box tests for the ReSnake Gymnasium environment."""

from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import resnake_gym  # noqa: F401 - importing the package registers the environment
from resnake_gym.envs import SnakeEnv

ENV_ID = "resnake_gym/ReSnake-v1"

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def _head_xy(observation: np.ndarray) -> tuple[int, int]:
    """Return the single head location from the documented CHW observation."""

    locations = np.argwhere(observation[2] == 1.0)
    assert locations.shape == (1, 2)
    y, x = locations[0]
    return int(x), int(y)


def _rollout(seed: int, actions: Iterable[int]):
    env = SnakeEnv(width=12, height=8, logic_fps=10.0)
    observation, info = env.reset(
        seed=seed,
        options={
            "snake": [(4, 3), (3, 3), (2, 3)],
            "direction": "right",
            # Omit food intentionally: its placement must use the seeded RNG.
        },
    )
    result = [(observation.copy(), 0.0, False, False, info)]
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        result.append((observation.copy(), reward, terminated, truncated, info.copy()))
        if terminated or truncated:
            break
    env.close()
    return result


def test_environment_passes_gymnasium_checker() -> None:
    env = SnakeEnv(width=10, height=8, max_logic_steps=30)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_seed_reproduces_reset_and_trajectory() -> None:
    actions = [RIGHT, DOWN, DOWN, LEFT, UP]
    first = _rollout(2026, actions)
    second = _rollout(2026, actions)

    assert len(first) == len(second)
    for transition_a, transition_b in zip(first, second, strict=True):
        obs_a, reward_a, terminated_a, truncated_a, info_a = transition_a
        obs_b, reward_b, terminated_b, truncated_b, info_b = transition_b
        np.testing.assert_array_equal(obs_a, obs_b)
        assert reward_a == reward_b
        assert terminated_a is terminated_b
        assert truncated_a is truncated_b
        for key in (
            "length",
            "score",
            "logic_fps",
            "frame_skip",
            "logic_steps",
            "elapsed_seconds",
        ):
            assert info_a[key] == info_b[key]


def test_observation_shape_dtype_and_space_membership() -> None:
    env = SnakeEnv(width=8, height=6)
    try:
        observation, _ = env.reset(seed=7)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (9, 6, 8)
        assert env.observation_space.dtype == np.dtype(np.float32)
        assert observation.shape == (9, 6, 8)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert np.all((observation >= 0.0) & (observation <= 1.0))
    finally:
        env.close()


def test_public_kernel_snapshot_and_views_do_not_expose_mutable_state() -> None:
    env = SnakeEnv(width=8, height=6)
    observation, _ = env.reset(seed=7)
    snapshot = env.state

    assert snapshot.snake == env.snake
    assert snapshot.occupied == frozenset(env.snake)
    np.testing.assert_array_equal(env.observation(), observation)
    assert env.contains_cell((7, 5))
    assert not env.contains_cell((8, 5))

    env.step(RIGHT)
    assert snapshot.logic_steps == 0
    assert snapshot.snake != env.state.snake


def test_counterfactual_clone_and_time_limit_mutation_are_isolated() -> None:
    env = SnakeEnv(width=8, height=6, max_logic_steps=10)
    env.reset(seed=7, options={"food": (0, 0)})
    trial = env.clone_for_simulation()
    trial.set_time_limits(max_logic_steps=None, max_episode_seconds=None)
    trial.step(RIGHT)

    assert env.state.logic_steps == 0
    assert trial.state.logic_steps == 1
    assert env.max_logic_steps == 10
    assert trial.max_logic_steps is None


def test_opposite_direction_is_ignored() -> None:
    env = SnakeEnv(width=8, height=6)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (0, 5),
            },
        )
        observation, _, terminated, truncated, _ = env.step(LEFT)
        assert not terminated
        assert not truncated
        assert _head_xy(observation) == (4, 2)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("action", "expected_head"),
    [
        (0, (3, 2)),  # straight
        (1, (4, 3)),  # turn right
        (2, (2, 3)),  # turn left
    ],
)
def test_relative_actions_turn_from_current_direction(
    action: int, expected_head: tuple[int, int]
) -> None:
    env = SnakeEnv(width=7, height=7, action_mode="relative")
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 3), (3, 4), (3, 5)],
                "direction": "up",
                "food": (0, 0),
            },
        )
        observation, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert not truncated
        assert _head_xy(observation) == expected_head
    finally:
        env.close()


def test_eating_food_grows_snake_and_increments_score() -> None:
    env = SnakeEnv(width=8, height=6)
    try:
        _, before = env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": RIGHT,
                "food": (4, 2),
            },
        )
        observation, reward, terminated, truncated, after = env.step(RIGHT)

        assert before["length"] == 3
        assert before["score"] == 0
        assert after["length"] == 4
        assert after["score"] == 1
        assert reward > 0
        assert not terminated
        assert not truncated
        assert _head_xy(observation) == (4, 2)
    finally:
        env.close()


def test_moving_into_tail_cell_is_legal_when_tail_moves_away() -> None:
    env = SnakeEnv(width=6, height=6)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(1, 1), (1, 2), (0, 2), (0, 1)],
                "direction": "up",
                "food": (5, 5),
            },
        )
        observation, _, terminated, truncated, info = env.step(LEFT)
        assert not terminated
        assert not truncated
        assert _head_xy(observation) == (0, 1)
        assert info["length"] == 4
    finally:
        env.close()


def test_wall_collision_terminates_episode() -> None:
    env = SnakeEnv(width=6, height=6)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(0, 2), (1, 2), (2, 2)],
                "direction": "left",
                "food": (5, 5),
            },
        )
        _, reward, terminated, truncated, _ = env.step(LEFT)
        assert terminated
        assert not truncated
        assert reward < 0
    finally:
        env.close()


def test_body_collision_terminates_episode() -> None:
    env = SnakeEnv(width=6, height=6)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [
                    (2, 2),
                    (2, 3),
                    (1, 3),
                    (1, 2),
                    (1, 1),
                    (2, 1),
                ],
                "direction": "up",
                "food": (5, 5),
            },
        )
        _, reward, terminated, truncated, _ = env.step(LEFT)
        assert terminated
        assert not truncated
        assert reward < 0
    finally:
        env.close()


def test_filling_board_terminates_as_a_win() -> None:
    env = SnakeEnv(width=3, height=3)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [
                    (1, 0),
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 2),
                    (2, 2),
                    (2, 1),
                    (1, 1),
                ],
                "direction": "right",
                "food": (2, 0),
            },
        )
        observation, reward, terminated, truncated, info = env.step(RIGHT)
        assert terminated
        assert not truncated
        assert reward > 0
        assert info["won"] is True
        assert info["termination_reason"] == "board_filled"
        assert info["length"] == 9
        assert info["score"] == 1
        assert _head_xy(observation) == (2, 0)
        assert observation[4].sum() == 0.0
    finally:
        env.close()


def test_max_logic_steps_truncates_without_termination() -> None:
    env = SnakeEnv(width=12, height=8, max_logic_steps=3)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 3), (2, 3), (1, 3)],
                "direction": "right",
                "food": (0, 0),
            },
        )
        for expected_step in (1, 2):
            _, _, terminated, truncated, info = env.step(RIGHT)
            assert not terminated
            assert not truncated
            assert info["logic_steps"] == expected_step

        _, _, terminated, truncated, info = env.step(RIGHT)
        assert not terminated
        assert truncated
        assert info["logic_steps"] == 3
    finally:
        env.close()


def test_timeout_reward_is_added_once_at_a_frame_skip_boundary() -> None:
    env = SnakeEnv(
        width=12,
        height=8,
        frame_skip=4,
        max_logic_steps=2,
        living_reward=0.25,
        timeout_reward=3.0,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 3), (2, 3), (1, 3)],
                "direction": "right",
                "food": (0, 0),
            },
        )
        observation, reward, terminated, truncated, info = env.step(RIGHT)

        assert reward == pytest.approx(3.5)
        assert not terminated
        assert truncated
        assert _head_xy(observation) == (5, 3)
        assert info["ticks_advanced"] == 2
        assert info["logic_steps"] == 2
        assert info["termination_reason"] == "time_limit"
    finally:
        env.close()


def test_rule_termination_on_limit_tick_does_not_get_timeout_reward() -> None:
    env = SnakeEnv(
        width=12,
        height=8,
        frame_skip=4,
        max_logic_steps=2,
        living_reward=0.25,
        death_reward=-1.0,
        timeout_reward=7.0,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(10, 3), (9, 3), (8, 3)],
                "direction": "right",
                "food": (0, 0),
            },
        )
        _, reward, terminated, truncated, info = env.step(RIGHT)

        assert reward == pytest.approx(-0.5)
        assert terminated
        assert not truncated
        assert info["ticks_advanced"] == 2
        assert info["logic_steps"] == 2
        assert info["termination_reason"] == "wall_collision"
    finally:
        env.close()


@pytest.mark.parametrize("timeout_reward", [np.nan, np.inf, -np.inf])
def test_timeout_reward_must_be_finite(timeout_reward: float) -> None:
    with pytest.raises(ValueError, match="timeout_reward must be finite"):
        SnakeEnv(timeout_reward=timeout_reward)


def test_max_episode_seconds_uses_logic_time() -> None:
    env = SnakeEnv(
        width=12,
        height=8,
        logic_fps=4.0,
        max_episode_seconds=0.5,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 3), (2, 3), (1, 3)],
                "direction": "right",
                "food": (0, 0),
            },
        )
        _, _, terminated, truncated, info = env.step(RIGHT)
        assert not terminated
        assert not truncated
        assert info["elapsed_seconds"] == pytest.approx(0.25)

        _, _, terminated, truncated, info = env.step(RIGHT)
        assert not terminated
        assert truncated
        assert info["logic_steps"] == 2
        assert info["elapsed_seconds"] == pytest.approx(0.5)
    finally:
        env.close()


def test_frame_skip_repeats_action_for_logic_steps() -> None:
    env = SnakeEnv(width=12, height=8, logic_fps=10.0, frame_skip=3)
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 3), (2, 3), (1, 3)],
                "direction": "right",
                "food": (0, 0),
            },
        )
        observation, _, terminated, truncated, info = env.step(RIGHT)
        assert not terminated
        assert not truncated
        assert _head_xy(observation) == (6, 3)
        assert info["frame_skip"] == 3
        assert info["logic_steps"] == 3
        assert info["elapsed_seconds"] == pytest.approx(0.3)
    finally:
        env.close()


def test_logic_fps_can_be_overridden_on_reset() -> None:
    env = SnakeEnv(width=8, height=6, logic_fps=7.5)
    try:
        _, initial_info = env.reset(seed=0)
        assert initial_info["logic_fps"] == pytest.approx(7.5)

        _, reset_info = env.reset(
            seed=0,
            options={
                "logic_fps": 20.0,
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (0, 5),
            },
        )
        assert reset_info["logic_fps"] == pytest.approx(20.0)
        assert reset_info["logic_steps"] == 0
        assert reset_info["elapsed_seconds"] == pytest.approx(0.0)

        _, _, _, _, step_info = env.step(RIGHT)
        assert step_info["logic_fps"] == pytest.approx(20.0)
        assert step_info["elapsed_seconds"] == pytest.approx(0.05)
    finally:
        env.close()


@pytest.mark.parametrize(
    "options",
    [
        {
            "snake": [(2, 2), (1, 2), (2, 2)],
            "direction": "right",
            "food": (5, 5),
        },
        {
            "snake": [(0, 0), (-1, 0)],
            "direction": "right",
            "food": (5, 5),
        },
        {
            "snake": [(2, 2), (1, 2), (0, 2)],
            "direction": "right",
            "food": (1, 2),
        },
    ],
    ids=["overlapping-snake", "out-of-bounds-snake", "food-on-snake"],
)
def test_reset_rejects_invalid_explicit_state(options: dict[str, object]) -> None:
    env = SnakeEnv(width=6, height=6)
    try:
        with pytest.raises(ValueError):
            env.reset(seed=0, options=options)
    finally:
        env.close()


def test_body_progress_and_direction_channels_have_documented_values() -> None:
    env = SnakeEnv(width=8, height=6)
    snake = [(4, 2), (3, 2), (2, 2), (1, 2)]
    try:
        observation, _ = env.reset(
            seed=0,
            options={
                "snake": snake,
                "direction": "right",
                "food": (0, 5),
            },
        )
        assert observation[1, 2, 4] == pytest.approx(1.0)
        assert observation[1, 2, 3] == pytest.approx(0.75)
        assert observation[1, 2, 2] == pytest.approx(0.5)
        assert observation[1, 2, 1] == pytest.approx(0.25)

        direction_masks = observation[5:9]
        assert direction_masks.sum() == pytest.approx(1.0)
        assert direction_masks[RIGHT, 2, 4] == pytest.approx(1.0)
    finally:
        env.close()


def test_body_progress_stays_consistent_after_move_and_growth() -> None:
    env = SnakeEnv(width=8, height=6)
    try:
        observation, _ = env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (5, 2),
            },
        )
        observation, _, _, _, _ = env.step(RIGHT)
        assert observation[1, 2, 4] == pytest.approx(1.0)
        assert observation[1, 2, 3] == pytest.approx(2 / 3)
        assert observation[1, 2, 2] == pytest.approx(1 / 3)
        assert observation[1, 2, 1] == pytest.approx(0.0)

        observation, _, _, _, info = env.step(RIGHT)
        assert info["length"] == 4
        assert observation[1, 2, 5] == pytest.approx(1.0)
        assert observation[1, 2, 4] == pytest.approx(0.75)
        assert observation[1, 2, 3] == pytest.approx(0.5)
        assert observation[1, 2, 2] == pytest.approx(0.25)
    finally:
        env.close()


def test_reset_rejects_full_board_and_inconsistent_direction() -> None:
    env = SnakeEnv(width=3, height=3)
    try:
        with pytest.raises(ValueError, match="free food cell"):
            env.reset(
                options={
                    "snake": [
                        (2, 2),
                        (1, 2),
                        (0, 2),
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (2, 0),
                        (1, 0),
                        (0, 0),
                    ],
                    "direction": "right",
                    "food": None,
                }
            )

        with pytest.raises(ValueError, match="neck"):
            env.reset(
                options={
                    "snake": [(2, 2), (1, 2), (0, 2)],
                    "direction": "left",
                    "food": (2, 1),
                }
            )
    finally:
        env.close()


def test_rgb_array_render_returns_uint8_image() -> None:
    env = SnakeEnv(width=8, height=6, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0
    finally:
        env.close()


def test_ansi_render_returns_board_string() -> None:
    env = SnakeEnv(width=8, height=6, render_mode="ansi")
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(3, 2), (2, 2), (1, 2)],
                "direction": "right",
                "food": (0, 5),
            },
        )
        frame = env.render()
        assert isinstance(frame, str)
        rows = frame.splitlines()
        assert len(rows) == 6
        assert all(len(row) == 8 for row in rows)
        assert sum(row.count("H") for row in rows) == 1
        assert sum(row.count("*") for row in rows) == 1
    finally:
        env.close()


def test_registered_environment_can_be_created_with_gym_make() -> None:
    env = gym.make(
        ENV_ID,
        width=9,
        height=7,
        logic_fps=12.0,
        frame_skip=2,
        max_episode_steps=40,
    )
    try:
        observation, info = env.reset(seed=123)
        assert env.spec is not None
        assert env.spec.id == ENV_ID
        assert observation.shape == (9, 7, 9)
        assert info["logic_fps"] == pytest.approx(12.0)
        assert info["frame_skip"] == 2
    finally:
        env.close()

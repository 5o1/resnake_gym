"""Tests for delayed observations and overlapping action chunks."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.error import ResetNeeded

from resnake_gym.envs import SnakeEnv
from resnake_gym.wrappers import ChunkedControlWrapper, DistanceRewardWrapper


def _make_env(
    *,
    width: int = 20,
    height: int = 20,
    control_interval_ticks: int = 2,
    chunk_length: int = 4,
    observation_age_ticks: int | tuple[int, int] = 0,
    command_delay_ticks: int | tuple[int, int] = 0,
    reward_discount_per_tick: float = 1.0,
    max_logic_steps: int | None = None,
    living_reward: float = 0.0,
) -> ChunkedControlWrapper:
    return ChunkedControlWrapper(
        SnakeEnv(
            width=width,
            height=height,
            frame_skip=1,
            action_mode="relative",
            max_logic_steps=max_logic_steps,
            living_reward=living_reward,
        ),
        control_interval_ticks=control_interval_ticks,
        chunk_length=chunk_length,
        observation_age_ticks=observation_age_ticks,
        command_delay_ticks=command_delay_ticks,
        reward_discount_per_tick=reward_discount_per_tick,
    )


def _reset_options() -> dict[str, object]:
    return {
        "snake": [(7, 7), (6, 7), (5, 7)],
        "direction": "right",
        "food": (18, 18),
    }


def test_wrapper_requires_direct_one_tick_relative_snake_env() -> None:
    with pytest.raises(ValueError, match="frame_skip=1"):
        ChunkedControlWrapper(
            SnakeEnv(frame_skip=2, action_mode="relative"),
            control_interval_ticks=1,
            chunk_length=1,
        )

    with pytest.raises(ValueError, match="action_mode='relative'"):
        ChunkedControlWrapper(
            SnakeEnv(frame_skip=1, action_mode="absolute"),
            control_interval_ticks=1,
            chunk_length=1,
        )

    wrapped = DistanceRewardWrapper(SnakeEnv(frame_skip=1, action_mode="relative"))
    with pytest.raises(TypeError, match="directly wrap SnakeEnv"):
        ChunkedControlWrapper(
            wrapped,  # type: ignore[arg-type]
            control_interval_ticks=1,
            chunk_length=1,
        )
    wrapped.close()


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("control_interval_ticks", 0, ValueError),
        ("chunk_length", -1, ValueError),
        ("observation_age_ticks", (-1, 2), ValueError),
        ("observation_age_ticks", (3, 2), ValueError),
        ("command_delay_ticks", [0, 1], TypeError),
        ("reward_discount_per_tick", -0.01, ValueError),
        ("reward_discount_per_tick", 1.01, ValueError),
        ("reward_discount_per_tick", float("nan"), ValueError),
    ],
)
def test_timing_configuration_validation(
    keyword: str, value: object, error: type[Exception]
) -> None:
    kwargs: dict[str, object] = {
        "control_interval_ticks": 2,
        "chunk_length": 4,
        "observation_age_ticks": 0,
        "command_delay_ticks": 0,
        "reward_discount_per_tick": 1.0,
    }
    kwargs[keyword] = value
    with pytest.raises(error):
        ChunkedControlWrapper(
            SnakeEnv(action_mode="relative"),
            **kwargs,  # type: ignore[arg-type]
        )


def test_reset_and_step_observations_belong_to_declared_space() -> None:
    env = _make_env(
        control_interval_ticks=3,
        chunk_length=5,
        observation_age_ticks=(0, 4),
        command_delay_ticks=(0, 2),
    )
    try:
        observation, info = env.reset(seed=91, options=_reset_options())
        assert env.observation_space.contains(observation)
        assert info["observation_tick"] == 0
        assert observation["executed_actions"].tolist() == [-1, -1, -1]

        observation, _, terminated, truncated, _ = env.step([0, 1, 0, 2, 0])
        assert not terminated
        assert not truncated
        assert env.observation_space.contains(observation)
        assert observation["previous_chunk"].tolist() == [0, 1, 0, 2, 0]
    finally:
        env.close()


def test_zero_delay_new_chunk_replaces_unexecuted_old_suffix() -> None:
    env = _make_env(control_interval_ticks=2, chunk_length=4)
    try:
        env.reset(seed=0, options=_reset_options())

        _, _, terminated, truncated, first_info = env.step([1, 0, 2, 2])
        assert not terminated
        assert not truncated
        assert first_info["executed_actions"].tolist() == [1, 0]
        assert first_info["executed_action_sources"].tolist() == [0, 0]
        assert first_info["executed_action_target_ticks"].tolist() == [0, 1]
        assert first_info["executed_action_chunk_indices"].tolist() == [0, 1]
        assert first_info["submitted_chunk_origin_tick"] == 0
        assert first_info["submitted_target_start_tick"] == 0
        assert first_info["submitted_target_end_tick"] == 3
        assert first_info["queue_target_ticks"] == (2, 3)
        assert first_info["queue_remaining"] == 2

        observation, _, terminated, truncated, second_info = env.step([2, 0, 0, 0])
        assert not terminated
        assert not truncated
        # The [2, 2] suffix from sequence 0 is discarded.  Sequence 1 starts
        # immediately at the next fixed control boundary.
        assert second_info["executed_actions"].tolist() == [2, 0]
        assert second_info["executed_action_sources"].tolist() == [1, 1]
        assert second_info["executed_action_target_ticks"].tolist() == [2, 3]
        assert second_info["executed_action_chunk_indices"].tolist() == [0, 1]
        assert second_info["overwritten_action_count"] == 2
        assert second_info["queue_target_ticks"] == (4, 5)
        assert second_info["queue_action_sources"] == (1, 1)
        assert env.unwrapped.head == (9, 9)
        assert observation["executed_actions"].tolist() == [2, 0]
    finally:
        env.close()


def test_delayed_command_discards_expired_prefix_without_shifting_chunk() -> None:
    env = _make_env(
        control_interval_ticks=4,
        chunk_length=4,
        command_delay_ticks=2,
    )
    try:
        env.reset(seed=0, options=_reset_options())
        _, _, terminated, truncated, info = env.step([1, 2, 1, 2])

        assert not terminated
        assert not truncated
        assert info["sampled_command_delay_ticks"] == 2
        assert info["command_arrival_tick"] == 2
        # Elements 0 and 1 targeted ticks 0 and 1.  They are expired when the
        # packet arrives at tick 2, so execution resumes at chunk index 2.
        assert info["executed_actions"].tolist() == [0, 0, 1, 2]
        assert info["executed_action_sources"].tolist() == [-1, -1, 0, 0]
        assert info["executed_action_target_ticks"].tolist() == [0, 1, 2, 3]
        assert info["executed_action_chunk_indices"].tolist() == [-1, -1, 2, 3]
        assert info["arrived_command_target_ranges"] == ((0, 0, 3),)
        assert info["expired_action_count"] == 2
        assert info["expired_command_prefixes"] == ((0, 2),)
        assert info["installed_action_count"] == 2
        assert info["queue_exhausted_tick_offsets"] == (0, 1)
        assert info["queue_remaining"] == 0
    finally:
        env.close()


def test_command_that_arrives_after_its_target_range_fully_expires() -> None:
    env = _make_env(
        control_interval_ticks=5,
        chunk_length=2,
        command_delay_ticks=3,
    )
    try:
        env.reset(seed=0, options=_reset_options())
        _, _, terminated, truncated, info = env.step([1, 2])

        assert not terminated
        assert not truncated
        assert info["executed_actions"].tolist() == [0, 0, 0, 0, 0]
        assert info["executed_action_sources"].tolist() == [-1] * 5
        assert info["expired_action_count"] == 2
        assert info["expired_command_prefixes"] == ((0, 2),)
        assert info["fully_expired_command_sequences"] == (0,)
        assert info["installed_command_sequences"] == ()
        assert info["installed_action_count"] == 0
        assert info["queue_remaining"] == 0
    finally:
        env.close()


def test_older_late_packet_cannot_overwrite_newer_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env(
        control_interval_ticks=2,
        chunk_length=6,
        command_delay_ticks=(0, 4),
    )
    command_delays = iter((4, 0, 4))

    def sample_timing(bounds: tuple[int, int]) -> int:
        if bounds == env.command_delay_range:
            return next(command_delays)
        return bounds[0]

    monkeypatch.setattr(env, "_sample_range", sample_timing)
    try:
        env.reset(seed=0, options=_reset_options())

        _, _, terminated, truncated, first_info = env.step([0] * 6)
        assert not terminated
        assert not truncated
        assert first_info["executed_action_sources"].tolist() == [-1, -1]

        _, _, terminated, truncated, second_info = env.step([0] * 6)
        assert not terminated
        assert not truncated
        assert second_info["executed_action_sources"].tolist() == [1, 1]
        assert second_info["queue_target_ticks"] == (4, 5, 6, 7)

        _, _, terminated, truncated, third_info = env.step([0] * 6)
        assert not terminated
        assert not truncated
        # Sequence 0 finally arrives at tick 4.  Its live target ticks 4 and 5
        # are already owned by sequence 1, so the stale packet installs none.
        assert third_info["arrived_command_sequences"] == (0,)
        assert third_info["expired_command_prefixes"] == ((0, 4),)
        assert third_info["discarded_stale_command_sequences"] == (0,)
        assert third_info["installed_action_count"] == 0
        assert third_info["shadowed_action_count"] == 2
        assert third_info["executed_action_sources"].tolist() == [1, 1]
        assert third_info["executed_action_chunk_indices"].tolist() == [2, 3]
        assert third_info["queue_target_ticks"] == (6, 7)
        assert third_info["queue_action_sources"] == (1, 1)
    finally:
        env.close()


def test_stale_observation_returns_past_state_and_monotonic_timestamp() -> None:
    env = _make_env(
        control_interval_ticks=1,
        chunk_length=1,
        observation_age_ticks=2,
    )
    try:
        initial, _ = env.reset(seed=3, options=_reset_options())
        initial_state = initial["delayed_state"].copy()

        first, _, _, _, first_info = env.step([0])
        second, _, _, _, second_info = env.step([0])
        third, _, _, _, third_info = env.step([0])

        assert first_info["observation_tick"] == 0
        assert first_info["observation_age"] == 1
        assert not first_info["new_observation"]
        assert np.array_equal(first["delayed_state"], initial_state)

        assert second_info["observation_tick"] == 0
        assert second_info["observation_age"] == 2
        assert not second_info["new_observation"]
        assert np.array_equal(second["delayed_state"], initial_state)

        assert third_info["observation_tick"] == 1
        assert third_info["observation_age"] == 2
        assert third_info["new_observation"]
        assert np.argwhere(third["delayed_state"][2] == 1).tolist() == [[7, 8]]
    finally:
        env.close()


def test_random_observation_timestamps_never_go_backwards() -> None:
    env = _make_env(
        width=30,
        height=30,
        control_interval_ticks=1,
        chunk_length=1,
        observation_age_ticks=(0, 6),
    )
    try:
        env.reset(
            seed=702,
            options={
                "snake": [(15, 15), (14, 15), (13, 15)],
                "direction": "right",
                "food": (29, 29),
            },
        )
        timestamps: list[int] = []
        flags: list[bool] = []
        for action in ([1], [1], [1], [1]) * 3:
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated
            assert not truncated
            timestamps.append(info["observation_tick"])
            flags.append(info["new_observation"])

        assert timestamps == sorted(timestamps)
        assert flags == [
            current > previous
            for current, previous in zip(timestamps, [0, *timestamps[:-1]], strict=True)
        ]
    finally:
        env.close()


def test_timing_randomness_is_reproducible_without_changing_game_rng() -> None:
    first = _make_env(
        control_interval_ticks=2,
        chunk_length=3,
        observation_age_ticks=(0, 5),
        command_delay_ticks=(0, 3),
        max_logic_steps=12,
    )
    second = _make_env(
        control_interval_ticks=2,
        chunk_length=3,
        observation_age_ticks=(0, 5),
        command_delay_ticks=(0, 3),
        max_logic_steps=12,
    )
    chunks = ([1, 0, 0], [1, 2, 0], [2, 0, 1], [0, 2, 0], [1, 0, 2], [0, 0, 0])
    try:
        first_observation, first_reset_info = first.reset(seed=2026)
        second_observation, second_reset_info = second.reset(seed=2026)
        for key in first_observation:
            assert np.array_equal(first_observation[key], second_observation[key])
        assert (
            first_reset_info["observation_tick"]
            == second_reset_info["observation_tick"]
        )

        for chunk in chunks:
            first_result = first.step(chunk)
            second_result = second.step(chunk)
            first_obs, first_reward, first_terminated, first_truncated, first_info = (
                first_result
            )
            (
                second_obs,
                second_reward,
                second_terminated,
                second_truncated,
                second_info,
            ) = second_result
            assert first_reward == second_reward
            assert first_terminated == second_terminated
            assert first_truncated == second_truncated
            for key in first_obs:
                assert np.array_equal(first_obs[key], second_obs[key])
            for key in (
                "sampled_command_delay_ticks",
                "command_arrival_tick",
                "executed_actions",
                "executed_action_sources",
                "requested_observation_age",
                "observation_tick",
                "observation_age",
                "new_observation",
            ):
                if isinstance(first_info[key], np.ndarray):
                    assert np.array_equal(first_info[key], second_info[key])
                else:
                    assert first_info[key] == second_info[key]
            if first_terminated or first_truncated:
                break
    finally:
        first.close()
        second.close()


def test_terminal_tick_stops_control_period_and_pads_executed_observation() -> None:
    env = _make_env(
        width=8,
        height=8,
        control_interval_ticks=4,
        chunk_length=3,
    )
    try:
        env.reset(
            seed=0,
            options={
                "snake": [(7, 3), (6, 3), (5, 3)],
                "direction": "right",
                "food": (0, 0),
            },
        )
        observation, _, terminated, truncated, info = env.step([0, 1, 2])

        assert terminated
        assert not truncated
        assert info["termination_reason"] == "wall_collision"
        assert info["control_ticks_advanced"] == 1
        assert info["period_end_tick"] == 1
        assert info["terminated_early"]
        assert info["executed_actions"].tolist() == [0]
        assert info["tick_rewards"].shape == (1,)
        assert observation["executed_actions"].tolist() == [0, -1, -1, -1]
        assert env.observation_space.contains(observation)

        with pytest.raises(ResetNeeded):
            env.step([0, 0, 0])
    finally:
        env.close()


def test_period_reward_uses_per_tick_discount_and_keeps_raw_sum_in_info() -> None:
    env = _make_env(
        control_interval_ticks=3,
        chunk_length=3,
        reward_discount_per_tick=0.5,
        living_reward=1.0,
    )
    try:
        _, reset_info = env.reset(seed=0, options=_reset_options())
        assert reset_info["reward_discount_per_tick"] == pytest.approx(0.5)
        assert reset_info["undiscounted_period_reward"] == pytest.approx(0.0)
        assert reset_info["discounted_period_reward"] == pytest.approx(0.0)

        _, reward, terminated, truncated, info = env.step([0, 0, 0])

        assert not terminated
        assert not truncated
        assert info["tick_rewards"].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert info["reward_discount_per_tick"] == pytest.approx(0.5)
        assert info["undiscounted_period_reward"] == pytest.approx(3.0)
        assert info["discounted_period_reward"] == pytest.approx(1.0 + 0.5 + 0.25)
        assert reward == pytest.approx(info["discounted_period_reward"])
    finally:
        env.close()

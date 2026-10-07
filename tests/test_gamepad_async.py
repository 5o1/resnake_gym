import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from resnake_gym.envs.gamepad_env import GamepadSnakeEnv
from resnake_gym.wrappers import AsyncGamepad as ExportedAsyncGamepad
from resnake_gym.wrappers.gamepad_async import AsyncGamepad


def make(**kwargs):
    return AsyncGamepad(GamepadSnakeEnv(width=31, height=20), **kwargs)


def reset(env, seed=42):
    return env.reset(
        seed=seed, options={"snake": [(10, 10), (9, 10), (8, 10)], "food": (0, 0)}
    )


def chunk(length=8, button=None):
    reports = np.zeros((length, 20), dtype=np.float32)
    if button is not None:
        reports[:, button] = 1
    return reports


def test_checker():
    check_env(make(), skip_render_check=True)


def test_current_async_wrapper_is_a_public_package_export():
    assert ExportedAsyncGamepad is AsyncGamepad


def test_game_advances_while_command_and_observation_are_in_flight():
    env = make(
        decision_ticks=2, command_delay=2, observation_delay=3, frame_drop_probability=0
    )
    original, _ = reset(env)
    obs, _, _, _, info = env.step(chunk(button=0))
    np.testing.assert_array_equal(obs["board"], original["board"])
    assert info["controller_tick"] == 2
    assert info["capture_tick"] == 0
    assert list(info["executed_sources"]) == [-1, -1]
    assert np.isclose(obs["timing"][0], 0.2)
    obs, _, _, _, info = env.step(chunk(button=3))
    assert info["expired_reports"] == 2
    # First chunk indices 2,3 execute; it was not shifted to start at index 0.
    assert list(info["executed_sources"]) == [1, 1]
    assert obs["history"][:, 0].sum() == 2
    assert info["capture_tick"] == 1


def test_expired_prefix_not_shifted():
    env = make(
        decision_ticks=3, command_delay=2, observation_delay=0, frame_drop_probability=0
    )
    reset(env)
    reports = chunk()
    reports[0, 0] = 1
    reports[2, 1] = 1
    obs, _, _, _, info = env.step(reports)
    assert info["expired_reports"] == 2
    assert obs["history"][2, 1] == 1
    assert obs["board"][2, 11, 12] == 1


def test_reordered_old_commands_cannot_replace_newer_schedule():
    env = make(decision_ticks=1, command_delay=0)
    reset(env)
    env._commands = [(0, 2, 0, chunk(button=0)), (1, 1, 0, chunk(button=1))]
    env._deliver_commands()
    env.tick = 1
    env._deliver_commands()
    assert env._schedule[2][0] == 2
    assert env._schedule[2][1][0] == 1


def test_old_frames_do_not_replace_new_frames():
    env = make()
    original, _ = reset(env)
    env.tick = 4
    env._frames = [(3, 3, original["board"] + 1), (4, 1, original["board"] + 2)]
    env._deliver_frames()
    assert env._frame_tick == 3
    np.testing.assert_array_equal(env._board, original["board"] + 1)


def test_dropouts_do_not_reveal_current_board():
    env = make(decision_ticks=3, frame_drop_probability=1, command_delay=0)
    original, _ = reset(env)
    obs, _, _, _, info = env.step(chunk())
    np.testing.assert_array_equal(obs["board"], original["board"])
    assert obs["timing"][2] == 0
    assert info["controller_tick"] == 3
    assert info["capture_tick"] == 0


def test_discount_uses_actual_ticks_and_truncation():
    env = AsyncGamepad(
        GamepadSnakeEnv(living_reward=1, max_logic_steps=2), decision_ticks=5, gamma=0.5
    )
    env.reset(seed=2)
    _, reward, terminated, truncated, info = env.step(chunk())
    assert truncated and not terminated
    assert reward == 1.5
    assert info["bootstrap_discount"] == 0.25
    assert info["ticks_advanced"] == 2
    with pytest.raises(ResetNeeded):
        env.step(chunk())


def test_seeding_and_irregular_decision_intervals():
    first, second = make(), make()
    reset(first)
    reset(second)
    intervals = []
    for _ in range(3):
        a, ra, ta, ua, ia = first.step(chunk())
        b, rb, tb, ub, ib = second.step(chunk())
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert (ra, ta, ua) == (rb, tb, ub)
        assert ia["capture_tick"] == ib["capture_tick"]
        intervals.append(ia["ticks_advanced"])
    assert len(set(intervals)) > 1


def test_scalar_actions_and_nonfinite_chunks_rejected():
    env = make()
    reset(env)
    for invalid in (1, np.full((8, 20), np.nan), chunk(7)):
        with pytest.raises(ValueError):
            env.step(invalid)


def test_reset_fps_changes_time_units_not_tick_schedule():
    env = make(decision_ticks=2, frame_drop_probability=1)
    env.reset(seed=1, options={"logic_fps": 20})
    obs, _, _, _, info = env.step(chunk())
    assert info["controller_tick"] == 2
    assert np.isclose(obs["timing"][0], 0.1)
    assert np.isclose(obs["timing"][1], 0.1)

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import resnake_gym
from resnake_gym import gamepad
from resnake_gym.envs import GamepadSnakeEnv as ExportedGamepadSnakeEnv
from resnake_gym.envs import PerturbedGamepadEnv, RewardGamepadEnv
from resnake_gym.envs.gamepad_env import GamepadSnakeEnv


def test_registered_current_interface():
    env = gym.make(resnake_gym.ENV_ID)
    assert env.action_space.shape == (20,)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(1)
    env.close()


def test_checker():
    check_env(GamepadSnakeEnv(), skip_render_check=True)


def test_current_environments_are_public_package_exports():
    assert ExportedGamepadSnakeEnv is GamepadSnakeEnv
    assert issubclass(RewardGamepadEnv, PerturbedGamepadEnv)


def test_no_relative_action_option():
    with pytest.raises(ValueError):
        GamepadSnakeEnv(action_mode="relative")


@pytest.mark.parametrize("index,direction", [(0, 0), (1, 2), (2, 3), (3, 1)])
def test_dpad_mapping(index, direction):
    action = gamepad.neutral()
    action[index] = 1
    assert gamepad.direction_request(action) == direction


def test_conflict_neutral_deadzone_and_axis_convention():
    report = gamepad.neutral()
    assert gamepad.direction_request(report) is None
    report[15] = 0.1
    assert gamepad.direction_request(report) is None
    report[15] = 0.9
    assert gamepad.direction_request(report) == 0
    report[:2] = 1
    assert gamepad.direction_request(report) is None
    report[:4] = [1, 0, 0, 1]
    assert gamepad.direction_request(report) is None


def test_release_keeps_motion_and_held_button_is_not_repeated_turn():
    env = GamepadSnakeEnv(width=12, height=10)
    env.reset(seed=1, options={"snake": [(5, 6), (4, 6), (3, 6)], "food": (0, 0)})
    report = gamepad.neutral()
    report[0] = 1
    for expected_y in (5, 4):
        obs, _, _, _, info = env.step(report)
        assert obs[2, expected_y, 5] == 1
        assert info["gamepad_report"][0] == 1
    obs, _, _, _, info = env.step(gamepad.neutral())
    assert obs[2, 3, 5] == 1
    assert not info["gamepad_report"].any()


@pytest.mark.parametrize("bad", [np.full(20, np.nan), np.ones(19), np.full(20, 2.0)])
def test_invalid_reports(bad):
    with pytest.raises(ValueError):
        gamepad.validate(bad)

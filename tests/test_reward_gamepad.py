import numpy as np
import pytest

from resnake_gym.envs.reward_gamepad import RewardGamepadEnv
from resnake_gym.wrappers.gamepad_async import AsyncGamepad


def env_at_food(**kwargs):
    env = RewardGamepadEnv(perturbation_interval=None, **kwargs)
    env.reset(seed=3, options={"snake": [(4, 5), (3, 5), (2, 5)], "food": (6, 5)})
    return env


def test_food_reward_and_new_target_potential_not_clipped():
    env = env_at_food()
    phi = env._potential()
    _, r0, *_ = env.step(np.zeros(20, np.float32))
    _, r1, done, _, info = env.step(np.zeros(20, np.float32))
    expected = 1 + 0.5 * 3 / 620 + 0.5
    assert not done
    assert info["food_event"]["ticks"] == 2
    assert info["food_event"]["distance"] == 2
    assert info["reward_base"] == pytest.approx(expected)
    assert r0 + 0.99 * r1 == pytest.approx(
        0.99 * expected + 0.99**2 * env._potential() - phi
    )
    assert env.food_start_tick == 2
    env.close()


def test_terminal_potential_zero_and_timeout_nonzero():
    env = env_at_food()
    env.reset(seed=1, options={"snake": [(30, 5), (29, 5), (28, 5)], "food": (1, 1)})
    phi = env._potential()
    _, reward, term, trunc, info = env.step(np.zeros(20, np.float32))
    assert term and not trunc
    assert reward == pytest.approx(-3 - phi)
    env.close()
    env = env_at_food(max_logic_steps=1)
    phi = env._potential()
    _, reward, term, trunc, info = env.step(np.zeros(20, np.float32))
    assert trunc and not term
    assert reward == pytest.approx(0.99 * env._potential() - phi)
    env.close()


def test_chunk_retains_food_events():
    env = AsyncGamepad(
        RewardGamepadEnv(perturbation_interval=None),
        decision_ticks=(3, 3),
        command_delay=(0, 0),
    )
    env.reset(seed=3, options={"snake": [(4, 5), (3, 5), (2, 5)], "food": (6, 5)})
    *_, info = env.step(np.zeros((8, 20), np.float32))
    assert len(info["food_events"]) == 1
    assert info["food_events"][0]["tick"] == 2
    env.close()

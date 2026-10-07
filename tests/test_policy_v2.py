import numpy as np
import pytest

from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv
from resnake_gym.wrappers.gamepad_async import AsyncGamepad


def test_simulation_time_fields():
    env = AsyncGamepad(
        PerturbedGamepadEnv(perturbation_interval=None),
        time_features=True,
        decision_ticks=(3, 3),
        chunk_length=8,
    )
    try:
        obs, _ = env.reset(seed=8)
        assert env.observation_space.contains(obs)
        np.testing.assert_allclose(obs["target_dt"], np.arange(1, 9) / 10)
        obs, *_ = env.step(np.zeros((8, 20), np.float32))
        assert env.observation_space.contains(obs)
        np.testing.assert_allclose(obs["history_age"][:3], [0.2, 0.1, 0])
        assert obs["time_context"][1] == pytest.approx(0.3)
    finally:
        env.close()


@pytest.mark.parametrize("spatial_pool", [False, True])
def test_encoder_ignores_padding_extent_and_values(spatial_pool):
    torch = pytest.importorskip("torch")
    from resnake_gym.models import HeadLocalEncoder

    torch.set_num_threads(1)
    encoder = HeadLocalEncoder(channels=14, dim=16, spatial_pool=spatial_pool)
    board = torch.zeros(1, 14, 20, 31)
    board[:, 10] = 1
    board[:, 2, 0, 0] = 1
    padded = torch.full((1, 14, 40, 40), float("nan"))
    padded[:, 10] = 0
    padded[:, :, :20, :31] = board
    padded.requires_grad_()
    output = encoder(padded)
    torch.testing.assert_close(output, encoder(board), atol=1e-6, rtol=1e-5)
    output.sum().backward()
    assert (padded.grad[:, :, 20:] == 0).all()
    assert (padded.grad[:, :, :, 31:] == 0).all()


def test_realtime_time_fields_use_nanoseconds_not_tick_age():
    from resnake_gym.realtime import policy_observation

    snapshot = dict(
        controller_tick=1,
        capture_tick=0,
        period_ns=100_000_000,
        epoch_ns=1_000_000_000,
        observation_captured_ns=1_000_000_000,
        snapshot_published_ns=1_110_000_000,
        actual_report=np.zeros(20, np.float32),
        previous_command=None,
        board=np.zeros((14, 31, 31), np.float32),
        feedback=[
            dict(tick=0, applied_ns=1_100_000_000, report=np.zeros(20, np.float32))
        ],
    )
    obs = policy_observation(
        snapshot,
        time_features=True,
        decision_ns=1_250_000_000,
        previous_decision_ns=1_020_000_000,
    )
    np.testing.assert_allclose(obs["time_context"], [0.25, 0.23, 0.1, 0.14, 0.15])
    assert obs["target_dt"][0] == pytest.approx(-0.05)
    assert obs["history_age"][0] == pytest.approx(0.15)


@pytest.mark.parametrize("decoder", ["parallel", "gru"])
def test_timing_conditioned_vtrace_policy(decoder):
    torch = pytest.importorskip("torch")
    from resnake_gym.gamepad_runtime import build_model, make_env, stack_observations
    from resnake_gym.gamepad_vtrace_contract import VTraceConfig
    from resnake_gym.models import distribution_from_observation

    torch.set_num_threads(1)
    config = VTraceConfig(dim=16, decoder=decoder)
    model = build_model(config)
    env = make_env(config)
    try:
        observation, _ = env.reset(seed=7)
        obs = stack_observations([observation], "cpu")
        first, _, _ = distribution_from_observation(model, obs)
        obs["time_context"][:, 0] += 0.2
        second, _, _ = distribution_from_observation(model, obs)
        assert not torch.equal(first.logits, second.logits)
        assert second.logits.shape == (1, 8, 5)
    finally:
        env.close()

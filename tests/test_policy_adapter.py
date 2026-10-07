"""One runtime adapter serves both supported training algorithms."""

from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo_contract import (  # noqa: E402
    COLLECTION_SEMANTICS_VERSION,
    RECURRENT_UPDATE_VERSION,
    REPLAY_OBJECTIVE_VERSION,
    REWARD_VERSION,
    SCENE_ENCODING_VERSION,
    TIME_ENCODING_VERSION,
    PPOConfig,
    checkpoint_semantics,
)
from resnake_gym.gamepad_runtime import build_model, make_env  # noqa: E402
from resnake_gym.gamepad_vtrace_contract import (  # noqa: E402
    VTraceConfig,
    checkpoint_metadata,
)
from resnake_gym.policy_adapter import PolicyAdapter  # noqa: E402


def _payload(config):
    if isinstance(config, PPOConfig):
        metadata = {
            **checkpoint_semantics(config.action_head),
            "collection_semantics": COLLECTION_SEMANTICS_VERSION,
            "recurrent_update": RECURRENT_UPDATE_VERSION,
            "replay_objective": REPLAY_OBJECTIVE_VERSION,
            "scene_encoding": SCENE_ENCODING_VERSION,
            "time_encoding": TIME_ENCODING_VERSION,
            "reward_version": REWARD_VERSION,
            "config": asdict(config),
        }
    else:
        metadata = checkpoint_metadata(config)
    return {**metadata, "model": build_model(config).state_dict()}


@pytest.mark.parametrize(
    "config",
    [
        PPOConfig(width=7, height=6, dim=8, chunk_length=2),
        VTraceConfig(width=7, height=6, dim=8, chunk_length=2),
    ],
)
def test_adapter_loads_and_emits_a_complete_gamepad_chunk(tmp_path, config):
    path = tmp_path / f"{type(config).__name__}.pt"
    torch.save(_payload(config), path)
    adapter = PolicyAdapter(path)
    env = make_env(config)
    try:
        observation, _ = env.reset(seed=17)
        report = adapter.act(observation)

        assert report.shape == (config.chunk_length, 20)
        assert adapter.chunk_length == config.chunk_length
        assert adapter.requires_timing_v2 is True
        assert adapter.hidden.shape == (1, config.dim)
        fork = adapter.fork()
        assert torch.equal(fork.hidden, adapter.hidden)
        assert fork.chunk_length == config.chunk_length
        assert fork.requires_timing_v2 is True
        fork.reset()
        assert fork.hidden is None
    finally:
        env.close()

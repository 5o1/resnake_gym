"""PPO adapter tests for the shared checkpoint-loading boundary."""

from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

import resnake_gym.policy_checkpoint as checkpoint_module  # noqa: E402
import resnake_gym.ppo_adapter as adapter_module  # noqa: E402
from resnake_gym.gamepad_ppo_contract import (  # noqa: E402
    COLLECTION_SEMANTICS_VERSION,
    RECURRENT_UPDATE_VERSION,
    REPLAY_OBJECTIVE_VERSION,
    PPOConfig,
    checkpoint_semantics,
)
from resnake_gym.gamepad_runtime import build_model  # noqa: E402
from resnake_gym.gamepad_vtrace_contract import (  # noqa: E402
    VTraceConfig,
    checkpoint_metadata,
)


class _LoadedModel:
    pass


def _ppo_checkpoint(config):
    return {
        **checkpoint_semantics(config.action_head),
        "collection_semantics": COLLECTION_SEMANTICS_VERSION,
        "recurrent_update": RECURRENT_UPDATE_VERSION,
        "replay_objective": REPLAY_OBJECTIVE_VERSION,
        "config": asdict(config),
        "model": build_model(config).state_dict(),
    }


def test_ppo_adapter_delegates_to_shared_loader_with_ppo_contract(monkeypatch):
    config = PPOConfig(dim=16)
    model = _LoadedModel()
    calls = []

    def load(path, device, *, expected_config_type):
        calls.append((path, device, expected_config_type))
        return {"source": "shared-loader"}, config, model

    monkeypatch.setattr(adapter_module, "load_policy_model", load)

    adapter = adapter_module.PPOAdapter("policy.pt", device="cpu")

    assert calls == [("policy.pt", "cpu", PPOConfig)]
    assert adapter.config is config
    assert adapter.model is model
    assert adapter.hidden is None


def test_ppo_adapter_loads_valid_ppo_checkpoint(tmp_path):
    config = PPOConfig(dim=16, chunk_length=2)
    path = tmp_path / "ppo.pt"
    torch.save(_ppo_checkpoint(config), path)

    adapter = adapter_module.PPOAdapter(path)

    assert adapter.config == config
    assert not adapter.model.training


def test_expected_checkpoint_family_is_checked_before_model_construction(
    tmp_path, monkeypatch
):
    config = VTraceConfig(dim=16)
    payload = {
        **checkpoint_metadata(config),
        "model": {},
    }
    path = tmp_path / "vtrace.pt"
    torch.save(payload, path)

    constructed = False

    def unexpected_build(_config):
        nonlocal constructed
        constructed = True
        raise AssertionError("model construction must follow the family check")

    monkeypatch.setattr(checkpoint_module, "build_model", unexpected_build)

    with pytest.raises(
        ValueError,
        match=r"checkpoint config is VTraceConfig; expected PPOConfig",
    ):
        checkpoint_module.load_policy_model(
            path,
            "cpu",
            expected_config_type=PPOConfig,
        )

    assert not constructed


def test_ppo_adapter_rejects_vtrace_checkpoint(tmp_path):
    config = VTraceConfig(dim=16)
    path = tmp_path / "vtrace.pt"
    torch.save(
        {
            **checkpoint_metadata(config),
            "model": {},
        },
        path,
    )

    with pytest.raises(
        ValueError,
        match=r"checkpoint config is VTraceConfig; expected PPOConfig",
    ):
        adapter_module.PPOAdapter(path)

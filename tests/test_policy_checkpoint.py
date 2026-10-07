"""Shared checkpoint loading and V-trace compatibility boundaries."""

import copy
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
from resnake_gym.gamepad_runtime import build_model  # noqa: E402
from resnake_gym.gamepad_vtrace_contract import (  # noqa: E402
    VTraceConfig,
    checkpoint_metadata,
)
from resnake_gym.policy_checkpoint import (  # noqa: E402
    config_from_checkpoint,
    load_policy_model,
)


def _ppo_payload(config):
    return {
        **checkpoint_semantics(config.action_head),
        "collection_semantics": COLLECTION_SEMANTICS_VERSION,
        "recurrent_update": RECURRENT_UPDATE_VERSION,
        "replay_objective": REPLAY_OBJECTIVE_VERSION,
        "scene_encoding": SCENE_ENCODING_VERSION,
        "time_encoding": TIME_ENCODING_VERSION,
        "reward_version": REWARD_VERSION,
        "config": asdict(config),
        "model": build_model(config).state_dict(),
    }


def _vtrace_payload(config):
    return {
        **checkpoint_metadata(config),
        "model": build_model(config).state_dict(),
    }


@pytest.mark.parametrize(
    ("name", "config", "payload_factory"),
    [
        ("ppo", PPOConfig(dim=8), _ppo_payload),
        ("vtrace", VTraceConfig(dim=8), _vtrace_payload),
    ],
)
def test_shared_loader_reconstructs_both_policy_families(
    tmp_path, name, config, payload_factory
):
    path = tmp_path / f"{name}.pt"
    torch.save(payload_factory(config), path)

    payload, restored_config, model = load_policy_model(path, "cpu")

    assert restored_config == config
    assert payload["format"].startswith(f"gamepad-{name}")
    assert model.training is False


def test_vtrace_loader_accepts_both_serialized_v3_config_shapes():
    config = VTraceConfig(dim=8)
    pre_cleanup = checkpoint_metadata(config)
    assert config_from_checkpoint(pre_cleanup) == config

    cleanup_shape = copy.deepcopy(pre_cleanup)
    cleanup_shape["config"].pop("action_head")
    cleanup_shape["config"].pop("chunk_rho")
    restored = config_from_checkpoint(cleanup_shape)
    assert restored.action_head == "dpad5"
    assert restored.chunk_rho is None


def test_vtrace_loader_explicitly_rejects_the_retired_h1_action_head():
    payload = checkpoint_metadata(VTraceConfig(dim=8))
    payload["config"]["action_head"] = "held_dpad5"

    with pytest.raises(ValueError, match="invalid"):
        config_from_checkpoint(payload)


def test_legacy_raw_ppo_config_omission_does_not_pick_new_dpad5_default():
    config = PPOConfig(dim=8, action_head="raw")
    payload = _ppo_payload(config)
    payload["config"].pop("action_head")

    restored = config_from_checkpoint(payload)

    assert restored.action_head == "raw"


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("action_encoding", "wrong-action"),
        ("scene_encoding", "wrong-scene"),
        ("time_encoding", "wrong-time"),
        ("reward_version", "wrong-reward"),
    ],
)
def test_ppo_loader_rejects_semantic_encoding_mismatches(field, wrong):
    payload = _ppo_payload(PPOConfig(dim=8, action_head="raw"))
    payload[field] = wrong

    with pytest.raises(ValueError, match=field):
        config_from_checkpoint(payload)

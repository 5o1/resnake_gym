"""Artifact and resume contracts for the restored PPO/SIL entry point."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo_contract import PPOConfig  # noqa: E402
from resnake_gym.training import ppo_run  # noqa: E402


def _options(tmp_path: Path, **overrides):
    values = {
        "output": tmp_path / "output",
        "resume": None,
        "updates": 1,
        "save_every": 1,
        "device": "cpu",
        "threads": 1,
        "source_root": tmp_path,
    }
    values.update(overrides)
    return ppo_run.PPORunOptions(**values)


def test_config_mapping_and_metadata_keep_ppo_sil_semantics(tmp_path):
    defaults = PPOConfig()
    arguments = {
        field: getattr(defaults, field) for field in ppo_run.PPO_CONFIG_ARGUMENT_FIELDS
    }
    config = ppo_run.build_ppo_config(arguments)
    metadata = ppo_run.build_run_metadata(config, _options(tmp_path))

    assert config.collection_mode == "events"
    assert (config.minimum_steps, config.event_target, config.rollout_steps) == (
        1,
        1,
        512,
    )
    assert config.sil_updates == 2
    assert metadata["experiment"] == "autonomous_ppo_sil_learning"
    assert metadata["replay_objective"] == "success-snapshot-sil-v2"
    assert metadata["unbounded_episode_archive"] is False
    assert metadata["scene_encoding"] == "scene-grid-v2"


@pytest.mark.parametrize("field", ["updates", "save_every", "threads"])
def test_run_options_reject_nonpositive_process_controls(tmp_path, field):
    with pytest.raises(ValueError, match=field.replace("_", "[-_]?")):
        _options(tmp_path, **{field: 0})


def test_resume_normalizes_only_the_versioned_legacy_raw_default():
    raw = PPOConfig(action_head="raw")
    payload = {"config": asdict(raw)}
    payload["config"].pop("action_head")

    ppo_run.validate_resume_config(payload, raw)
    with pytest.raises(ValueError, match="action_head"):
        ppo_run.validate_resume_config(payload, PPOConfig(action_head="dpad5"))


@pytest.mark.parametrize(
    "field", ["width", "height", "num_envs", "decision_max", "replay_capacity"]
)
def test_resume_rejects_replay_incompatible_environment_shapes(field):
    config = PPOConfig()
    payload = {"config": asdict(config)}
    payload["config"][field] += 1

    with pytest.raises(ValueError, match=field):
        ppo_run.validate_resume_config(payload, config)


def test_restore_rejects_inference_policy_before_trainer_mutation(tmp_path):
    config = PPOConfig()
    payload = {
        "artifact_kind": "inference-policy",
        "config": asdict(config),
    }
    path = tmp_path / "policy.pt"
    torch.save(payload, path)

    class Trainer:
        restored = False

        def restore(self, _payload):
            self.restored = True

    trainer = Trainer()
    with pytest.raises(ValueError, match="full training-state"):
        ppo_run.restore_trainer(trainer, config, path)
    assert trainer.restored is False


class _FakeTrainer:
    def __init__(self):
        self.updates = 0
        self.policy_version = 40

    def update(self):
        self.updates += 1
        self.policy_version += 1
        return {"loss": 1.0, "episodes": [], "food_events": [], "fragments": []}

    def checkpoint(self):
        return {
            "artifact_kind": "training-state",
            "format": "fixture",
            "optimizer": {"state": 1},
            "replay_episodes": [[{"reward": 1.0}]],
            "replay_success_snapshots": {0: [{"reward": 1.0}]},
        }


def test_update_loop_keeps_replay_out_of_inference_policy(tmp_path):
    options = _options(tmp_path)
    options.output.mkdir()
    trainer = _FakeTrainer()

    ppo_run.run_ppo_updates(trainer, options)

    metrics = json.loads((options.output / "metrics.jsonl").read_text())
    policy = torch.load(options.output / "policy-000041.pt", weights_only=True)
    assert metrics["update"] == 41
    assert trainer.updates == 1
    assert "optimizer" not in policy
    assert "replay_episodes" not in policy
    assert "replay_success_snapshots" not in policy
    assert policy["artifact_kind"] == "inference-policy"

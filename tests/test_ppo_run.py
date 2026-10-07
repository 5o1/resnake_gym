import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo import PPOConfig  # noqa: E402
from resnake_gym.training import ppo_run  # noqa: E402


def _config_arguments(**overrides):
    defaults = PPOConfig()
    arguments = {
        name: getattr(defaults, name) for name in ppo_run.PPO_CONFIG_ARGUMENT_FIELDS
    }
    arguments.update(overrides)
    return arguments


def _options(tmp_path: Path, **overrides):
    values = {
        "output": tmp_path / "output",
        "resume": None,
        "updates": 2,
        "save_every": 1,
        "device": "cpu",
        "threads": 1,
        "source_root": tmp_path,
    }
    values.update(overrides)
    return ppo_run.PPORunOptions(**values)


def test_cli_config_mapping_includes_initial_length():
    arguments = _config_arguments(
        width=17,
        height=11,
        initial_length=5,
        action_head="dpad5",
    )

    config = ppo_run.build_ppo_config(arguments)

    assert config.width == 17
    assert config.height == 11
    assert config.initial_length == 5
    assert config.action_head == "dpad5"
    assert {
        name: getattr(config, name) for name in ppo_run.PPO_CONFIG_ARGUMENT_FIELDS
    } == {name: arguments[name] for name in ppo_run.PPO_CONFIG_ARGUMENT_FIELDS}


def test_resume_validation_includes_initial_length_and_legacy_default():
    config = PPOConfig(initial_length=3)
    legacy = {"config": asdict(config)}
    del legacy["config"]["initial_length"]

    ppo_run.validate_resume_config(legacy, config)

    mismatched = {"config": {**asdict(config), "initial_length": 4}}
    with pytest.raises(ValueError, match="resume config mismatch: initial_length"):
        ppo_run.validate_resume_config(mismatched, config)


def test_restore_validates_then_restores_and_applies_invocation_learning_rate(
    tmp_path,
):
    config = PPOConfig(learning_rate=0.004, initial_length=4)
    payload = {"config": asdict(config), "marker": "restored"}
    resume = tmp_path / "checkpoint.pt"
    torch.save(payload, resume)
    trainer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": -1.0}]),
        restored=None,
    )
    trainer.restore = lambda restored: setattr(trainer, "restored", restored)

    ppo_run.restore_trainer(trainer, config, resume)

    assert trainer.restored == payload
    assert trainer.optimizer.param_groups[0]["lr"] == config.learning_rate


class _FakeTrainer:
    def __init__(self):
        self.updates = 0
        self.checkpoints = 0

    def update(self):
        self.updates += 1
        return {
            "loss": float(self.updates),
            "episodes": [{"ignored": True}],
            "food_events": [{"ignored": True}],
            "fragments": [{"ignored": True}],
        }

    def checkpoint(self):
        self.checkpoints += 1
        return {
            "format": "fake",
            "checkpoint_call": self.checkpoints,
            "optimizer": {"state": self.checkpoints},
            "replay_episodes": [[{"reward": 1.0}]],
            "replay_success_snapshots": {0: [{"reward": 1.0}]},
        }


def test_training_loop_preserves_metrics_and_checkpoint_artifact_schema(tmp_path):
    options = _options(tmp_path)
    options.output.mkdir()
    trainer = _FakeTrainer()

    ppo_run.run_ppo_updates(trainer, options)

    metrics = [
        json.loads(line)
        for line in (options.output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["update"] for row in metrics] == [1, 2]
    assert all("episodes" in row and "food_events" in row for row in metrics)
    assert trainer.updates == 2
    assert trainer.checkpoints == 3
    assert (
        torch.load(
            options.output / "checkpoint.pt", map_location="cpu", weights_only=True
        )["checkpoint_call"]
        == 3
    )

    policy = torch.load(
        options.output / "policy-000002.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert policy["checkpoint_call"] == 2
    assert "optimizer" not in policy
    assert "replay_episodes" not in policy
    assert "replay_success_snapshots" not in policy
    assert policy["note"] == (
        "inference-only model checkpoint; no optimizer or replay state"
    )


def test_metadata_keeps_existing_schema_and_hashes_sources(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/example.py").write_text("VALUE = 1\n")
    options = _options(tmp_path, resume=tmp_path / "resume.pt")
    config = PPOConfig(action_head="dpad5", initial_length=4)

    metadata = ppo_run.build_run_metadata(config, options)

    assert metadata["config"] == asdict(config)
    assert metadata["resume"] == str(options.resume)
    assert metadata["checkpoint_format"] == "gamepad-ppo-v4"
    assert metadata["action_encoding"] == "xinput-dpad5-categorical-v1"
    assert metadata["training_objective"] == "snake-dpad5-categorical-v1"
    assert set(metadata["source_sha256"]) == {"src/example.py"}

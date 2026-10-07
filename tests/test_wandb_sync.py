"""Pure log conversion tests; no account/network access."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "sync_wandb", Path(__file__).parents[1] / "scripts/sync_wandb.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_scalar_conversion_and_late_evaluation(tmp_path):
    rows = [
        {
            "update": 20,
            "logic_ticks": 200,
            "value_loss": 0.1,
            "episodes": [
                {"score": 2, "ticks": 100, "won": False},
                {"score": 0, "ticks": 20, "won": False},
            ],
            "fragments": [{"private": "not uploaded"}],
        }
    ]
    (tmp_path / "metrics.jsonl").write_text(json.dumps(rows[0]) + '\n{"unfinished":')
    evaluation = {
        "checkpoint": "/gpu/policy-000010.pt",
        "deterministic_policy": True,
        "episodes": [
            {"size": "31x20", "seed": 9, "score": 1, "ticks": 50, "won": False}
        ],
    }
    (tmp_path / "validation-000010.json").write_text(json.dumps(evaluation))
    events = module().events(tmp_path)
    assert [e[0] for e in events] == [10, 20]
    assert events[-1][2]["train/mean_score"] == 1
    assert events[-1][2]["train/mean_lifetime_ticks"] == 60
    assert events[-1][2]["train/score_p50"] == 1
    assert events[-1][2]["train/score_p90"] == pytest.approx(1.8)
    assert events[-1][2]["train/lifetime_ticks_p99"] == pytest.approx(99.2)
    assert "train/fragments" not in events[-1][2]
    assert "train/episodes" not in events[-1][2]
    assert events[0][2]["eval_deterministic_31x20/update"] == 10
    legacy_signature = hashlib.sha256(json.dumps([9]).encode()).hexdigest()[:12]
    assert events[0][1] == f"eval_deterministic_31x20:10:{legacy_signature}"
    # Copies of the same evaluation have the same id for upload deduplication.
    (tmp_path / "eval-000010.json").write_text(json.dumps(evaluation))
    events = module().events(tmp_path)
    assert events[0][1] == events[1][1]


def test_incomplete_evaluation_not_uploaded(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("")
    (tmp_path / "validation-000010.json").write_text('{"unfinished":')
    assert module().events(tmp_path) == []

    (tmp_path / "validation-000010.json").write_text("[]")
    assert module().events(tmp_path) == []

    (tmp_path / "validation-000010.json").write_text('{"episodes": []}')
    assert module().events(tmp_path) == []


def test_metrics_file_may_not_exist_during_trainer_startup(tmp_path):
    assert module().events(tmp_path) == []


def test_vtrace_training_iteration_and_controller_tick_are_converted(tmp_path):
    row = {
        "training_iteration": 3,
        "fresh_logic_ticks": 400,
        "episodes": [
            {"score": 4, "controller_tick": 90, "won": False},
            {"score": 2, "controller_tick": 30, "won": False},
        ],
    }
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row) + "\n")

    converted = module().events(tmp_path)

    assert len(converted) == 1
    assert converted[0][0] == 3
    assert converted[0][1] == "train:3"
    assert converted[0][2]["train/update"] == 3
    assert converted[0][2]["train/mean_score"] == 3
    assert converted[0][2]["train/mean_lifetime_ticks"] == 60


def test_received_training_scalars_and_episode_failure_breakdown_are_uploaded(
    tmp_path,
):
    row = {
        "training_iteration": 4,
        "cumulative_received_logic_ticks": 250_123,
        "food_per_10k_logic_ticks": 12.5,
        "received_logic_ticks_per_second": 900.0,
        "trained_logic_ticks_per_second": 700.0,
        "entropy_joint_nats": 6.0,
        "entropy_per_action_variable_nats": 0.75,
        "entropy_normalized_fraction": 0.5,
        "entropy_beta_h": 0.006,
        "episodes": [
            {
                "score": 0,
                "ticks": 10,
                "won": False,
                "termination": "wall_collision",
            },
            {
                "score": 3,
                "ticks": 40,
                "won": False,
                "termination": "time_limit",
            },
        ],
        "trained_episodes": [
            {
                "score": 99,
                "ticks": 999,
                "won": True,
                "termination": "board_filled",
            }
        ],
    }
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row) + "\n")

    payload = module().events(tmp_path)[0][2]

    assert payload["train/food_per_10k_logic_ticks"] == 12.5
    assert payload["train/received_logic_ticks"] == 250_123
    assert payload["train/received_logic_ticks_per_second"] == 900.0
    assert payload["train/trained_logic_ticks_per_second"] == 700.0
    assert payload["train/entropy_joint_nats"] == 6.0
    assert payload["train/score_p50"] == 1.5
    assert payload["train/score_p90"] == pytest.approx(2.7)
    assert payload["train/termination_wall_collision_count"] == 1
    assert payload["train/termination_wall_collision_rate"] == 0.5
    assert payload["train/termination_time_limit_count"] == 1
    assert payload["train/death_count"] == 1
    assert payload["train/death_rate"] == 0.5
    assert "train/trained_episodes" not in payload


def test_full_checkpoint_evaluation_uses_recorded_training_iteration(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("")
    evaluation = {
        "checkpoint": "/gpu/checkpoint.pt",
        "source_training_iteration": 37,
        "source_received_logic_ticks": 98765,
        "deterministic_policy": True,
        "episodes": [
            {"size": "31x20", "seed": 9, "score": 1, "ticks": 50, "won": False}
        ],
    }
    (tmp_path / "validation-full.json").write_text(json.dumps(evaluation))

    converted = module().events(tmp_path)

    assert len(converted) == 1
    assert converted[0][0] == 37
    assert converted[0][2]["eval_deterministic_31x20/update"] == 37
    assert converted[0][2]["eval_deterministic_31x20/received_logic_ticks"] == 98765


def test_legacy_full_checkpoint_evaluation_without_update_is_skipped(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("")
    evaluation = {
        "checkpoint": "/gpu/checkpoint.pt",
        "deterministic_policy": True,
        "episodes": [
            {"size": "31x20", "seed": 9, "score": 1, "ticks": 50, "won": False}
        ],
    }
    (tmp_path / "validation-full.json").write_text(json.dumps(evaluation))

    assert module().events(tmp_path) == []


def test_metadata_waits_through_partial_config_while_trainer_is_alive(
    tmp_path, monkeypatch
):
    script = module()
    valid = {"config": {"gamma": 0.9999}}
    reads = iter((None, None, valid))
    monkeypatch.setattr(script, "read_json", lambda _path: next(reads))
    monkeypatch.setattr(script, "alive", lambda _pid, _directory: True)
    monkeypatch.setattr(script.time, "sleep", lambda _seconds: None)

    assert script.wait_for_metadata(tmp_path, 123) == valid


def test_metadata_failure_is_explicit_after_trainer_exits(tmp_path, monkeypatch):
    script = module()
    monkeypatch.setattr(script, "read_json", lambda _path: None)
    monkeypatch.setattr(script, "alive", lambda _pid, _directory: False)

    with pytest.raises(RuntimeError, match="exited before publishing"):
        script.wait_for_metadata(tmp_path, 123)

    with pytest.raises(RuntimeError, match="no valid training config"):
        script.wait_for_metadata(tmp_path, None)


def test_extracted_sync_runner_backfills_once_with_explicit_proxy(
    tmp_path, monkeypatch
):
    from resnake_gym.analysis.wandb_sync import WandbSyncOptions, sync_wandb

    (tmp_path / "config.json").write_text(
        json.dumps({"config": {"gamma": 0.99}, "format": "test-format"})
    )
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"update": 3, "loss": 0.5}) + "\n"
    )
    monkeypatch.setenv("ALL_PROXY", "http://wrong-proxy.invalid")

    class FakeRun:
        resumed = False
        url = "https://wandb.invalid/fake"

        def __init__(self):
            self.summary = {}
            self.defined = []
            self.logged = []
            self.finished = False

        def define_metric(self, name, **kwargs):
            self.defined.append((name, kwargs))

        def log(self, payload):
            self.logged.append(payload)

        def finish(self):
            self.finished = True

    run = FakeRun()
    init_calls = []

    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    fake_wandb = SimpleNamespace(
        init=init,
        Settings=lambda **kwargs: kwargs,
        Api=lambda: pytest.fail("non-resumed run must not query remote history"),
    )
    proxy = "http://127.0.0.1:17891"

    sync_wandb(
        WandbSyncOptions(tmp_path, "entity", "project", proxy=proxy),
        wandb_module=fake_wandb,
    )

    assert os.environ.get("ALL_PROXY") is None
    assert os.environ["HTTP_PROXY"] == proxy
    assert os.environ["HTTPS_PROXY"] == proxy
    assert init_calls[0]["config"]["checkpoint_semantics"] == {"format": "test-format"}
    assert run.defined == [
        ("train/update", {}),
        ("train/*", {"step_metric": "train/update"}),
    ]
    assert run.logged == [
        {
            "train/update": 3,
            "train/loss": 0.5,
            "source_event": "train:3",
        }
    ]
    assert run.summary == {
        "synced_source_events": 1,
        "training_process_alive": False,
    }
    assert run.finished is True

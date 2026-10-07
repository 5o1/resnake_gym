import json
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_vtrace_contract import VTraceConfig  # noqa: E402
from resnake_gym.training import vtrace_run  # noqa: E402
from resnake_gym.training.vtrace_metrics import (  # noqa: E402
    METRICS_STATE_FORMAT,
    RunMetricsState,
)


def _config_arguments(**overrides):
    defaults = VTraceConfig()
    arguments = {
        name: getattr(defaults, name)
        for name in vtrace_run.VTRACE_CONFIG_ARGUMENT_FIELDS
    }
    arguments["no_spatial_pool"] = False
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
        "max_fresh_logic_ticks": None,
        "source_root": tmp_path,
    }
    values.update(overrides)
    return vtrace_run.VTraceRunOptions(**values)


def test_cli_config_mapping_is_explicit_and_preserves_spatial_pool_inversion():
    arguments = _config_arguments(
        width=17,
        no_spatial_pool=True,
    )

    config = vtrace_run.build_vtrace_config(arguments)

    assert config.width == 17
    assert config.spatial_pool is False
    assert {
        name: getattr(config, name) for name in vtrace_run.VTRACE_CONFIG_ARGUMENT_FIELDS
    } == {name: arguments[name] for name in vtrace_run.VTRACE_CONFIG_ARGUMENT_FIELDS}


class FakeOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": -1.0}]


class FakeLearner:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.optimizer = FakeOptimizer()
        self.model = object()
        self.update_count = 0
        self.logic_ticks = 0
        self.transitions = 0
        self.restored = None

    def restore(self, payload):
        self.restored = payload
        self.update_count = int(payload["update"])
        self.logic_ticks = int(payload["logic_ticks"])
        self.transitions = int(payload["transitions"])


class FakeAssembler:
    def __init__(self, config):
        self.config = config
        self.cumulative_received_logic_ticks = 0
        self.cumulative_received_transitions = 0
        self.pending_logic_ticks = 0
        self.pending_transitions = 0
        self.ready_logic_ticks = 0
        self.ready_transitions = 0
        self.loaded = None
        self.flushed = False

    def load_state_dict(self, payload):
        self.loaded = payload
        self.cumulative_received_logic_ticks = int(payload["received_ticks"])
        self.cumulative_received_transitions = int(payload["received_transitions"])

    def flush_for_new_generation(self):
        self.flushed = True


def test_resume_initialization_restores_matching_runtime_generation_and_metadata(
    tmp_path,
    monkeypatch,
):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    resume = resume_dir / "checkpoint.pt"
    payload = {
        "training_iteration": 3,
        "run_generation": 4,
        "runtime_checkpoint_id": "matching-runtime",
        "update": 3,
        "logic_ticks": 7,
        "transitions": 5,
        "script_state": {
            "metrics_state_format": METRICS_STATE_FORMAT,
            "best_score": -1,
        },
    }
    torch.save(payload, resume)
    matching_runtime = {
        "credit_assembler": {
            "received_ticks": 7,
            "received_transitions": 5,
        }
    }
    loaded_runtime = []

    def fake_load_runtime(path, checkpoint_id):
        loaded_runtime.append((path, checkpoint_id))
        return matching_runtime

    monkeypatch.setattr(vtrace_run, "GamepadVTraceLearner", FakeLearner)
    monkeypatch.setattr(vtrace_run, "CreditTraceAssembler", FakeAssembler)
    monkeypatch.setattr(vtrace_run, "load_matching_runtime", fake_load_runtime)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    config = vtrace_run.build_vtrace_config(_config_arguments(learning_rate=0.004))
    options = _options(tmp_path, resume=resume, updates=4)

    state = vtrace_run.initialize_vtrace_run(config, options)

    assert state.start_iteration == 3
    assert state.run_generation == 5
    assert state.learner.restored == payload
    assert state.learner.optimizer.param_groups[0]["lr"] == 0.004
    assert state.assembler.loaded == matching_runtime["credit_assembler"]
    assert state.assembler.flushed is True
    assert loaded_runtime == [(resume, "matching-runtime")]
    metadata = json.loads((options.output / "config.json").read_text())
    assert metadata["resume"] == str(resume)
    assert metadata["run_generation"] == 5
    assert metadata["training_budget"] == {
        "primary_axis": "cumulative_received_logic_ticks",
        "max_fresh_logic_ticks": None,
        "updates_safety_cap": 4,
        "overshoot": "at_most_one_complete_fresh_collection_batch",
    }


@dataclass
class IterationAssembler:
    cumulative_received_logic_ticks: int = 0
    cumulative_received_transitions: int = 0
    pending_logic_ticks: int = 0
    pending_transitions: int = 0
    ready_logic_ticks: int = 0
    ready_transitions: int = 0


class IterationLearner:
    update_count = 0
    logic_ticks = 0
    transitions = 0

    def update(self, traces):
        assert traces == [{"score_end": 2, "trace_marker": "best"}]
        self.update_count = 1
        self.logic_ticks = 3
        self.transitions = 1
        return {"loss": 0.25}


class IterationActorPool:
    def __init__(self, assembler):
        self.assembler = assembler
        self.published = []

    def collect(self, assembler):
        assert assembler is self.assembler
        assembler.cumulative_received_logic_ticks = 3
        assembler.cumulative_received_transitions = 1
        episode = {
            "score": 2,
            "ticks": 3,
            "won": False,
            "termination": "time_limit",
        }
        return (
            [{"score_end": 2, "trace_marker": "best"}],
            {
                "fresh_episodes": [episode],
                "fresh_trained_episodes": [episode],
                "fresh_received_episodes": [episode],
                "fresh_logic_ticks": 3,
                "fresh_received_logic_ticks": 3,
                "fresh_trained_food_count": 2,
                "fresh_received_food_count": 2,
            },
        )

    def publish(self, *, policy_version):
        self.published.append(policy_version)
        return policy_version


def test_one_iteration_preserves_event_populations_accounting_and_best_trace(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    output.mkdir()
    assembler = IterationAssembler()
    learner = IterationLearner()
    state = vtrace_run.VTraceRunState(
        learner=learner,
        assembler=assembler,
        metrics=RunMetricsState(),
        start_iteration=0,
        run_generation=0,
    )
    pool = IterationActorPool(assembler)
    clock = iter((11.0, 13.0, 15.0))
    monkeypatch.setattr(vtrace_run.time, "monotonic", lambda: next(clock))

    metrics = vtrace_run.run_vtrace_iteration(
        state,
        pool,
        _options(tmp_path, output=output),
        iteration=1,
        started=10.0,
    )

    assert pool.published == [1]
    assert metrics["training_iteration"] == 1
    assert metrics["loss"] == 0.25
    assert metrics["wall_seconds"] == 2.0
    assert metrics["total_wall_seconds"] == 5.0
    assert metrics["received_completed_episodes"] == 1
    assert metrics["trained_completed_episodes"] == 1
    assert metrics["cumulative_received_food_count"] == 2
    assert metrics["cumulative_trained_food_count"] == 2
    assert metrics["received_trained_pending_ready_conserved"] is True
    assert state.metrics.best_score == 2
    assert torch.load(
        output / "best-progress-trace.pt",
        weights_only=True,
    ) == {"score_end": 2, "trace_marker": "best"}

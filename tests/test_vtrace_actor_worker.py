import copy
import threading
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import resnake_gym.gamepad_vtrace_actor as actor_module  # noqa: E402
from resnake_gym.gamepad_runtime import build_model, make_env  # noqa: E402
from resnake_gym.gamepad_vtrace_contract import VTraceConfig  # noqa: E402
from resnake_gym.gamepad_vtrace_fragments import validate_fragment  # noqa: E402


def _config(**updates):
    values = {
        "width": 8,
        "height": 8,
        "dim": 8,
        "decision_min": 1,
        "decision_max": 1,
        "observation_delay_max": 0,
        "command_delay_max": 0,
        "drop_probability": 0.0,
        "max_logic_steps": 100,
        "perturbation_min": 100,
        "perturbation_max": 100,
        "spatial_pool": False,
        "actor_processes": 1,
        "envs_per_actor": 1,
        "actor_sync_steps": 1,
        "unroll_length": 4,
        "recurrent_burn_in": 2,
        "bptt_window": 2,
        "credit_trace_max_transitions": 8,
        "batch_min_transitions": 1,
        "batch_max_transitions": 4,
        "batch_food_target": 0,
        "queue_capacity": 4,
    }
    values.update(updates)
    return VTraceConfig(**values)


def _shared_parameters(config):
    return (
        torch.nn.utils.parameters_to_vector(build_model(config).parameters())
        .detach()
        .cpu()
        .clone()
        .share_memory_()
    )


def test_actor_sync_seals_old_fragment_before_new_policy_version(monkeypatch):
    config = _config()
    probe = make_env(config)
    observation = probe.reset(seed=91)[0]
    probe.close()

    class Version:
        value = 0

    version = Version()

    class FakeEnvironment:
        def __init__(self):
            self.steps = 0
            self.reset_seeds = []
            self.closed = False

        def reset(self, *, seed=None):
            self.reset_seeds.append(seed)
            return copy.deepcopy(observation), {}

        def step(self, report):
            self.steps += 1
            if self.steps == 1:
                version.value = 1
            terminated = self.steps == 2
            executed = np.asarray(report, dtype=np.float32)[:1].copy()
            info = {
                "bootstrap_discount": config.gamma,
                "ticks_advanced": 1,
                "controller_tick": self.steps,
                "capture_tick": self.steps,
                "command_origin_tick": self.steps - 1,
                "command_arrival_tick": self.steps - 1,
                "execution_ticks": [self.steps - 1],
                "tick_rewards": [0.0],
                "executed_reports": executed,
                "executed_sources": np.array([self.steps], dtype=np.int64),
                "submitted_sequence": self.steps,
                "food_events": [],
                "score": 0,
                "perturbation_events": [],
                "expired_reports": 0,
                "won": False,
                "termination_reason": "wall_collision" if terminated else None,
            }
            return copy.deepcopy(observation), 0.0, terminated, False, info

        def close(self):
            self.closed = True

    environment = FakeEnvironment()
    monkeypatch.setattr(actor_module, "make_env", lambda unused: environment)
    stop_event = threading.Event()

    class Sink:
        def __init__(self):
            self.items = []

        def put(self, payload, timeout):
            self.items.append(payload)
            if len(self.items) == 2:
                stop_event.set()

    sink = Sink()
    run_generation = 3
    actor_module.actor_worker(
        0,
        run_generation,
        asdict(config),
        _shared_parameters(config),
        version,
        threading.Lock(),
        sink,
        stop_event,
    )

    assert [item["kind"] for item in sink.items] == ["fragment", "fragment"]
    old_fragment, new_fragment = sink.items
    validate_fragment(old_fragment, config)
    validate_fragment(new_fragment, config)
    assert old_fragment["policy_versions"].tolist() == [0]
    assert new_fragment["policy_versions"].tolist() == [1]
    assert [item["fragment_sequence"] for item in sink.items] == [0, 1]
    assert [(item["decision_start"], item["decision_end"]) for item in sink.items] == [
        (0, 1),
        (1, 2),
    ]
    assert old_fragment["termination_reason"] is None
    assert new_fragment["termination_reason"] == "wall_collision"
    assert old_fragment["episode_id"] == new_fragment["episode_id"] == 0
    expected_seed = config.seed + run_generation * 10_000_019
    assert environment.reset_seeds == [expected_seed, None]
    assert environment.closed

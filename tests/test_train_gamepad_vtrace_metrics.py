import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_vtrace import (  # noqa: E402
    CreditTraceAssembler,
    GamepadVTraceLearner,
    VTraceConfig,
)


def module():
    spec = importlib.util.spec_from_file_location(
        "train_gamepad_vtrace_metrics",
        Path(__file__).parents[1] / "scripts/train_gamepad_vtrace.py",
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_episode_metrics_report_quantiles_lifetime_and_death_reasons():
    metrics = module()._episode_metrics(
        [
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
        prefix="received_",
    )

    assert metrics["received_completed_episodes"] == 2
    assert metrics["received_episode_score_p50"] == 1.5
    assert metrics["received_episode_score_p90"] == pytest.approx(2.7)
    assert metrics["received_episode_score_p99"] == pytest.approx(2.97)
    assert metrics["received_episode_lifetime_ticks_mean"] == 25
    assert metrics["received_episode_lifetime_ticks_p90"] == pytest.approx(37)
    assert metrics["received_termination_wall_collision_count"] == 1
    assert metrics["received_termination_time_limit_count"] == 1
    assert metrics["received_death_count"] == 1
    assert metrics["received_death_rate"] == 0.5


def test_empty_episode_population_does_not_publish_fake_zero_quantiles():
    metrics = module()._episode_metrics([], prefix="received_")

    assert metrics == {
        "received_completed_episodes": 0,
        "received_strict_wins": 0,
    }


def test_inference_checkpoint_carries_received_and_trained_axes(tmp_path):
    script = module()
    config = VTraceConfig(
        dim=8,
        actor_processes=1,
        envs_per_actor=1,
        unroll_length=1,
        recurrent_burn_in=1,
        bptt_window=1,
        credit_trace_max_transitions=1,
        batch_min_transitions=1,
        batch_max_transitions=1,
        batch_food_target=0,
    )
    learner = GamepadVTraceLearner(config)
    assembler = CreditTraceAssembler(config)

    script._save_checkpoint(
        learner,
        assembler,
        tmp_path,
        iteration=1,
        script_state={"metrics_state_format": script._METRICS_STATE_FORMAT},
        run_generation=0,
    )

    policy = torch.load(tmp_path / "policy-000001.pt", weights_only=True)
    assert policy["logic_ticks"] == 0
    assert policy["trained_logic_ticks"] == 0
    assert policy["trained_transitions"] == 0
    assert policy["received_logic_ticks"] == 0
    assert policy["received_transitions"] == 0
    runtime = torch.load(tmp_path / "runtime.pt", weights_only=False)
    assert runtime["checkpoint_id"] == policy["runtime_checkpoint_id"]
    assert "replay" not in runtime

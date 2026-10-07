"""Tests for the package-owned checkpoint evaluation workflow."""

import json
import runpy
from pathlib import Path

import pytest

from resnake_gym.evaluation_protocol import protocol_sha256
from resnake_gym.evaluation_run import (
    PolicyEvaluationOptions,
    atomic_json,
    build_evaluation_result,
    run_policy_evaluation,
)
from resnake_gym.gamepad_ppo_contract import PPOConfig
from resnake_gym.gamepad_vtrace_contract import VTraceConfig


def test_result_records_source_and_complete_environment_protocol(tmp_path):
    options = PolicyEvaluationOptions(
        checkpoint=tmp_path / "policy-000007.pt",
        output=tmp_path / "evaluation.json",
        sizes=("12x8",),
        episodes=1,
        seed=41,
        device="cpu",
        stochastic=False,
    )
    config = VTraceConfig(width=12, height=8, initial_length=5)
    result = build_evaluation_result(
        {"logic_ticks": 11, "transitions": 3, "policy_version": 2},
        config,
        options,
        checkpoint_sha256="a" * 64,
        results=[{"size": "12x8", "seed": 41, "score": 1, "won": False}],
        timings=[0.01, 0.02],
    )

    assert result["source_training_iteration"] == 7
    assert result["source_policy_version"] == 2
    assert result["source_trained_logic_ticks"] == 11
    assert result["evaluation_protocol"]["initial_snake_length"] == 5
    assert result["evaluation_protocol_sha256"] == protocol_sha256(
        result["evaluation_protocol"]
    )


def test_atomic_json_refuses_to_replace_an_existing_result(tmp_path):
    path = tmp_path / "evaluation.json"
    atomic_json(path, {"complete": True})

    assert json.loads(path.read_text()) == {"complete": True}
    with pytest.raises(FileExistsError):
        atomic_json(path, {"complete": False})


def test_ppo_result_preserves_recurrent_and_replay_semantics(tmp_path):
    options = PolicyEvaluationOptions(
        checkpoint=tmp_path / "checkpoint.pt",
        output=tmp_path / "evaluation.json",
        sizes=("31x20",),
        episodes=1,
        seed=41,
        device="cpu",
        stochastic=False,
    )
    result = build_evaluation_result(
        {
            "training_iteration": 12,
            "recurrent_update": "stored-state-burnin-v1",
            "replay_objective": "success-snapshot-sil-v2",
        },
        PPOConfig(),
        options,
        checkpoint_sha256="b" * 64,
        results=[{"size": "31x20", "seed": 41, "score": 0, "won": False}],
        timings=[0.01],
    )

    assert result["source_training_iteration"] == 12
    assert result["source_recurrent_update"] == "stored-state-burnin-v1"
    assert result["source_replay_objective"] == "success-snapshot-sil-v2"


def test_general_policy_evaluation_script_uses_the_package_runner():
    namespace = runpy.run_path(
        Path(__file__).parents[1] / "scripts" / "evaluate_gamepad_policy.py"
    )

    assert namespace["run_policy_evaluation"] is run_policy_evaluation

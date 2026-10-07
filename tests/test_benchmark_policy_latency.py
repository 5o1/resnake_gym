"""Tests for the single-environment policy latency benchmark."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from resnake_gym.envs import SnakeEnv

_SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "scripts/benchmark_policy_latency.py"
)
_deadline_summary = _SCRIPT["_deadline_summary"]
_summarize_ms = _SCRIPT["_summarize_ms"]
run_unpaced = _SCRIPT["run_unpaced"]


class _StraightPolicy:
    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[np.integer, None]:
        assert deterministic
        assert observation.shape == (9, 4, 4)
        return np.int64(0), None


def test_latency_summary_reports_distribution() -> None:
    summary = _summarize_ms([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["max_ms"] == pytest.approx(4.0)


def test_deadline_summary_counts_strict_misses() -> None:
    summary = _deadline_summary([4.0, 5.0, 6.0], target_fps=200.0)

    assert summary["period_ms"] == pytest.approx(5.0)
    assert summary["deadline_misses"] == 1
    assert summary["deadline_miss_rate"] == pytest.approx(1 / 3)


def test_unpaced_benchmark_is_batch_one_and_resets_terminal_episodes() -> None:
    env = SnakeEnv(
        width=4,
        height=4,
        action_mode="relative",
        max_logic_steps=10,
    )
    try:
        result = run_unpaced(
            _StraightPolicy(),
            env,
            seed=7,
            warmup_steps=2,
            measure_steps=20,
        )
    finally:
        env.close()

    assert result["measured_decisions"] == 20
    assert result["episode_resets"] >= 1
    assert result["policy_predict"]["count"] == 20
    assert result["environment_step"]["count"] == 20
    assert result["predict_plus_environment_step"]["count"] == 20
    assert result["reset_time_in_latency_samples"] is False

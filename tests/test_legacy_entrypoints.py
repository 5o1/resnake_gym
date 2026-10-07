"""Compatibility checks for isolated historical command implementations."""

from __future__ import annotations

import importlib
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script", "implementation"),
    [
        ("train_ppo.py", "resnake_gym_legacy.v0.train_ppo"),
        ("evaluate_policy.py", "resnake_gym_legacy.v0.evaluate_policy"),
        ("train_bc.py", "resnake_gym_legacy.v0.train_bc"),
        ("benchmark_env.py", "resnake_gym_legacy.v0.benchmark_env"),
        ("evaluate_oracle.py", "resnake_gym_legacy.v0.evaluate_oracle"),
        ("record_video.py", "resnake_gym_legacy.v0.record_video"),
        (
            "benchmark_policy_latency.py",
            "resnake_gym_legacy.v0.benchmark_policy_latency",
        ),
        (
            "train_chunked_ppo.py",
            "resnake_gym_legacy.relative_chunked.train_ppo",
        ),
        (
            "evaluate_chunked_policy.py",
            "resnake_gym_legacy.relative_chunked.evaluate_policy",
        ),
    ],
)
def test_historical_script_reexports_implementation_contract(
    script: str,
    implementation: str,
) -> None:
    namespace = runpy.run_path(Path(__file__).parents[1] / "scripts" / script)
    module = importlib.import_module(implementation)

    exported = namespace["__all__"]
    assert exported
    for name in exported:
        assert namespace[name] is getattr(module, name)

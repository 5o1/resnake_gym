"""Tests for the PPO training command-line interface."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_SCRIPT = runpy.run_path(Path(__file__).parents[1] / "scripts/train_ppo.py")
parse_args = _SCRIPT["parse_args"]


def test_reward_shaping_cli_defaults_are_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ppo.py", "--output-dir", str(tmp_path)],
    )

    args = parse_args()

    assert args.timeout_reward == 0.0
    assert args.distance_reward_scale == 0.0
    assert args.logic_fps == pytest.approx(10.0)
    assert args.frame_skip == 1


def test_reward_shaping_cli_accepts_explicit_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--output-dir",
            str(tmp_path),
            "--timeout-reward",
            "-0.75",
            "--distance-reward-scale",
            "0.05",
            "--logic-fps",
            "30",
            "--frame-skip",
            "2",
        ],
    )

    args = parse_args()

    assert args.timeout_reward == pytest.approx(-0.75)
    assert args.distance_reward_scale == pytest.approx(0.05)
    assert args.logic_fps == pytest.approx(30.0)
    assert args.frame_skip == 2

"""Tests for the online chunked-PPO command-line contract."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_SCRIPT = runpy.run_path(Path(__file__).parents[1] / "scripts/train_chunked_ppo.py")
parse_args = _SCRIPT["parse_args"]
_discount_per_control_period = _SCRIPT["_discount_per_control_period"]
_timing_kwargs_from_args = _SCRIPT["_timing_kwargs_from_args"]
_validate_args = _SCRIPT["_validate_args"]


def test_chunked_training_defaults_keep_full_board_and_expose_timing(
    tmp_path: Path,
) -> None:
    args = parse_args(["--output-dir", str(tmp_path)])

    assert (args.width, args.height) == (31, 20)
    assert args.logic_fps == pytest.approx(10.0)
    assert args.control_interval_ticks == 3
    assert args.chunk_length == 6
    assert (
        args.observation_age_min_ticks,
        args.observation_age_max_ticks,
    ) == (0, 2)
    assert (
        args.command_delay_min_ticks,
        args.command_delay_max_ticks,
    ) == (0, 2)
    assert args.food_reward == pytest.approx(0.1)
    assert args.death_reward == pytest.approx(-1.0)
    assert args.living_reward == pytest.approx(-1e-4)
    assert args.win_reward == pytest.approx(10.0)
    assert args.timeout_reward == pytest.approx(0.0)
    _validate_args(args)


def test_chunked_training_cli_accepts_explicit_timing_and_rewards(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--control-interval-ticks",
            "4",
            "--chunk-length",
            "9",
            "--observation-age-min-ticks",
            "1",
            "--observation-age-max-ticks",
            "5",
            "--command-delay-min-ticks",
            "2",
            "--command-delay-max-ticks",
            "3",
            "--food-reward",
            "2",
            "--death-reward",
            "-3",
            "--living-reward",
            "-0.02",
            "--win-reward",
            "20",
            "--timeout-reward",
            "-4",
        ]
    )

    _validate_args(args)
    assert args.control_interval_ticks == 4
    assert args.chunk_length == 9
    assert (args.observation_age_min_ticks, args.observation_age_max_ticks) == (
        1,
        5,
    )
    assert (args.command_delay_min_ticks, args.command_delay_max_ticks) == (2, 3)
    assert args.food_reward == pytest.approx(2.0)
    assert args.timeout_reward == pytest.approx(-4.0)


@pytest.mark.parametrize(
    ("per_tick", "ticks", "expected"),
    [
        (0.995, 3, 0.995**3),
        (0.95, 4, 0.95**4),
        (0.0, 7, 0.0),
        (1.0, 7, 1.0),
    ],
)
def test_discount_is_converted_from_tick_to_control_period(
    per_tick: float, ticks: int, expected: float
) -> None:
    assert _discount_per_control_period(per_tick, ticks) == pytest.approx(expected)


def test_training_passes_per_tick_gamma_to_reward_aggregation(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--gamma-per-tick",
            "0.97",
            "--control-interval-ticks",
            "4",
        ]
    )

    timing_kwargs = _timing_kwargs_from_args(args)

    assert timing_kwargs["reward_discount_per_tick"] == pytest.approx(0.97)
    assert timing_kwargs["control_interval_ticks"] == 4
    assert _discount_per_control_period(0.97, 4) == pytest.approx(0.97**4)


@pytest.mark.parametrize(
    ("per_tick", "ticks"),
    [(-0.1, 2), (1.1, 2), (float("nan"), 2), (0.9, 0)],
)
def test_discount_conversion_rejects_invalid_values(
    per_tick: float, ticks: int
) -> None:
    with pytest.raises(ValueError):
        _discount_per_control_period(per_tick, ticks)


def test_chunked_training_rejects_reversed_timing_range(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--observation-age-min-ticks",
            "4",
            "--observation-age-max-ticks",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="observation-age range"):
        _validate_args(args)

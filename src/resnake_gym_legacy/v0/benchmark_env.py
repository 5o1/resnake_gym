"""Benchmark full ``SnakeEnv.step`` throughput at controlled snake lengths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from resnake_gym.baselines import (
    HamiltonianOracle,
    absolute_action,
    cycle_aligned_snake,
)
from resnake_gym.envs import SnakeEnv


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _benchmark_lengths(capacity: int) -> tuple[tuple[str, int, float | None], ...]:
    return (
        ("length_3", 3, None),
        ("25_percent", max(1, math.floor(capacity * 0.25)), 0.25),
        ("50_percent", max(1, math.floor(capacity * 0.50)), 0.50),
        ("90_percent", max(1, math.floor(capacity * 0.90)), 0.90),
    )


def _window_setup(
    oracle: HamiltonianOracle,
    length: int,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Return a reset state and steps that cannot reach the fixed food."""

    cycle = oracle.cycle
    capacity = len(cycle)
    if not 1 <= length < capacity:
        raise ValueError("benchmark lengths must be between 1 and capacity - 1")

    head_index = 0
    snake = cycle_aligned_snake(cycle, length=length, head_index=head_index)
    if len(snake) == 1:
        direction = absolute_action(cycle[-1], cycle[0])
    else:
        direction = absolute_action(snake[1], snake[0])

    # Occupied cycle indices are 0, -1, ..., -(length - 1).  The final free
    # cell in the forward direction is therefore capacity - length steps away.
    # Stop one step earlier so no timed step can eat food and change length.
    food_distance = capacity - length
    food = cycle[(head_index + food_distance) % capacity]
    window_steps = food_distance - 1
    if window_steps < 1:
        raise ValueError(
            f"length {length} leaves no non-growing step window on a "
            f"{oracle.width}x{oracle.height} board"
        )

    actions = tuple(
        absolute_action(
            cycle[(head_index + offset) % capacity],
            cycle[(head_index + offset + 1) % capacity],
        )
        for offset in range(window_steps)
    )
    options = {"snake": snake, "direction": direction, "food": food}
    return options, actions


def _run_window(
    env: SnakeEnv,
    *,
    seed: int,
    options: dict[str, Any],
    actions: Sequence[int],
    expected_length: int,
) -> float:
    """Time one step-only window, validating state outside the timed section."""

    observation, reset_info = env.reset(seed=seed, options=options)
    if int(reset_info["length"]) != expected_length:
        raise RuntimeError("reset did not preserve the requested snake length")
    terminated = False
    truncated = False
    info = reset_info
    started_at = time.perf_counter()
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
    elapsed = time.perf_counter() - started_at

    if terminated or truncated:
        raise RuntimeError("a benchmark window ended unexpectedly")
    if int(info["length"]) != expected_length:
        raise RuntimeError("a benchmark window changed the requested snake length")
    if int(info["logic_steps"]) != len(actions):
        raise RuntimeError("a benchmark window advanced an unexpected tick count")
    return elapsed


def _benchmark_case(
    *,
    width: int,
    height: int,
    label: str,
    length: int,
    requested_fraction: float | None,
    duration: float | None,
    repeats: int | None,
    seed: int,
) -> tuple[dict[str, Any], int]:
    oracle = HamiltonianOracle(width, height)
    options, actions = _window_setup(oracle, length)
    env = SnakeEnv(
        width=width,
        height=height,
        action_mode="absolute",
        frame_skip=1,
        render_mode=None,
    )
    observation_bytes = 0
    measured_seconds = 0.0
    completed_windows = 0

    try:
        observation, _ = env.reset(seed=seed, options=options)
        observation_bytes = int(observation.nbytes)

        # One unmeasured window removes most first-use effects.  Its reset and
        # every measured reset are outside the reported step-only duration.
        _run_window(
            env,
            seed=seed,
            options=options,
            actions=actions,
            expected_length=length,
        )

        while (repeats is not None and completed_windows < repeats) or (
            repeats is None and measured_seconds < float(duration)
        ):
            measured_seconds += _run_window(
                env,
                seed=seed,
                options=options,
                actions=actions,
                expected_length=length,
            )
            completed_windows += 1
    finally:
        env.close()

    env_steps = completed_windows * len(actions)
    result = {
        "label": label,
        "requested_fraction": requested_fraction,
        "length": length,
        "actual_fraction": length / (width * height),
        "safe_steps_per_window": len(actions),
        "measured_windows": completed_windows,
        "environment_steps": env_steps,
        "logic_steps": env_steps,
        "measured_seconds": measured_seconds,
        "steps_per_second": env_steps / measured_seconds,
        "logic_steps_per_second": env_steps / measured_seconds,
        "length_preserved": True,
    }
    return result, observation_bytes


def benchmark_environment(
    *,
    width: int,
    height: int,
    duration: float | None,
    repeats: int | None,
    seed: int = 0,
) -> dict[str, Any]:
    """Benchmark step throughput and return a JSON-serializable report."""

    if duration is None and repeats is None:
        duration = 1.0
    if duration is not None and repeats is not None:
        raise ValueError("choose either duration or repeats, not both")
    if duration is not None and (not math.isfinite(duration) or duration <= 0.0):
        raise ValueError("duration must be a finite positive number")
    if repeats is not None and repeats < 1:
        raise ValueError("repeats must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    oracle = HamiltonianOracle(width, height)
    capacity = len(oracle.cycle)
    cases = _benchmark_lengths(capacity)
    results: list[dict[str, Any]] = []
    observation_bytes: int | None = None

    for case_index, (label, length, fraction) in enumerate(cases):
        result, case_observation_bytes = _benchmark_case(
            width=width,
            height=height,
            label=label,
            length=length,
            requested_fraction=fraction,
            duration=duration,
            repeats=repeats,
            seed=seed + case_index,
        )
        if observation_bytes is None:
            observation_bytes = case_observation_bytes
        elif observation_bytes != case_observation_bytes:
            raise RuntimeError("observation size changed between benchmark cases")
        results.append(result)

    return {
        "system": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
            "resnake_gym": importlib.metadata.version("resnake-gym"),
        },
        "config": {
            "width": width,
            "height": height,
            "board_capacity": capacity,
            "action_mode": "absolute",
            "frame_skip": 1,
            "render_mode": None,
            "duration_seconds_per_length": duration,
            "repeats_per_length": repeats,
            "seed": seed,
            "observation_shape": [9, height, width],
            "observation_dtype": "float32",
            "observation_bytes": observation_bytes,
            "percentage_length_rounding": "floor(board_capacity * fraction)",
        },
        "methodology": {
            "metric": "Python-loop throughput of complete SnakeEnv.step calls",
            "included": "action application, logic tick, observation, reward, and info",
            "excluded": "policy computation, reset time, and one warm-up window",
            "length_control": (
                "Each window resets to the exact labeled cycle-aligned length; "
                "food is placed one step beyond the timed path, and final length "
                "is asserted unchanged."
            ),
        },
        "results": results,
    }


def _serialize_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json_atomic(path: Path, contents: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=_positive_int, default=31)
    parser.add_argument("--height", type=_positive_int, default=20)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--duration",
        type=_positive_float,
        help="minimum measured step time per length (default: 1 second)",
    )
    mode.add_argument(
        "--repeats",
        type=_positive_int,
        help="fixed number of measured reset/step windows per length",
    )
    parser.add_argument("--seed", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="also atomically write the complete JSON report to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = benchmark_environment(
            width=args.width,
            height=args.height,
            duration=args.duration,
            repeats=args.repeats,
            seed=args.seed,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    contents = _serialize_json(report)
    if args.output_json is not None:
        _write_json_atomic(args.output_json, contents)
    sys.stdout.write(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

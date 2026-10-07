"""Evaluate the deterministic Hamiltonian oracle over a range of seeds."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _percentile(values: Sequence[int], percentile: float) -> float:
    """Return a linearly interpolated percentile (the NumPy default method)."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return float(lower + (upper - lower) * fraction)


def _initial_state(
    oracle: HamiltonianOracle,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Build an oracle-aligned length-three state for every supported board."""

    snake = cycle_aligned_snake(oracle.cycle, length=3, head_index=2)
    direction = absolute_action(snake[1], snake[0])
    return snake, direction


def evaluate_oracle(
    *,
    width: int,
    height: int,
    episodes: int,
    start_seed: int,
    max_logic_steps: int,
    action_mode: str = "relative",
) -> dict[str, Any]:
    """Run seeded oracle episodes and return JSON-serializable metrics."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if start_seed < 0:
        raise ValueError("start_seed must be non-negative")
    if max_logic_steps < 1:
        raise ValueError("max_logic_steps must be positive")
    if action_mode not in {"absolute", "relative"}:
        raise ValueError("action_mode must be 'absolute' or 'relative'")

    oracle = HamiltonianOracle(width, height)
    initial_snake, initial_direction = _initial_state(oracle)
    env = SnakeEnv(
        width=width,
        height=height,
        action_mode=action_mode,
        frame_skip=1,
        max_logic_steps=max_logic_steps,
    )
    episode_metrics: list[dict[str, Any]] = []
    total_logic_steps = 0
    started_at = time.perf_counter()

    try:
        for episode_index in range(episodes):
            seed = start_seed + episode_index
            _, info = env.reset(
                seed=seed,
                options={
                    "snake": initial_snake,
                    "direction": initial_direction,
                },
            )
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = oracle.act_head(
                    env.head,
                    env.direction,
                    action_mode=action_mode,
                )
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            strict_win = bool(
                terminated
                and info["won"] is True
                and info["termination_reason"] == "board_filled"
            )
            logic_steps = int(info["logic_steps"])
            total_logic_steps += logic_steps
            episode_metrics.append(
                {
                    "episode": episode_index + 1,
                    "seed": seed,
                    "strict_win": strict_win,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "won": bool(info["won"]),
                    "termination_reason": info["termination_reason"],
                    "logic_steps": logic_steps,
                    "score": int(info["score"]),
                    "length": int(info["length"]),
                    "total_reward": float(total_reward),
                }
            )
    finally:
        env.close()

    wall_seconds = time.perf_counter() - started_at
    steps = [int(result["logic_steps"]) for result in episode_metrics]
    wins = sum(bool(result["strict_win"]) for result in episode_metrics)
    summary = {
        "episodes": episodes,
        "wins": wins,
        "failures": episodes - wins,
        "win_rate": wins / episodes,
        "total_logic_steps": total_logic_steps,
        "mean_logic_steps": float(statistics.fmean(steps)),
        "median_logic_steps": float(statistics.median(steps)),
        "p95_logic_steps": _percentile(steps, 0.95),
        "min_logic_steps": min(steps),
        "max_logic_steps": max(steps),
        "wall_seconds": wall_seconds,
        "logic_steps_per_second": total_logic_steps / wall_seconds,
    }
    return {
        "config": {
            "width": width,
            "height": height,
            "episodes": episodes,
            "start_seed": start_seed,
            "max_logic_steps": max_logic_steps,
            "action_mode": action_mode,
            "frame_skip": 1,
            "initial_length": len(initial_snake),
        },
        "episodes": episode_metrics,
        "summary": summary,
    }


def _serialize_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json_atomic(path: Path, contents: str) -> None:
    """Replace a JSON output from a same-directory temporary file."""

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
    parser.add_argument("--episodes", type=_positive_int, default=10)
    parser.add_argument("--start-seed", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--max-logic-steps",
        type=_positive_int,
        default=200_000,
        help="truncate each episode after this many environment logic ticks",
    )
    parser.add_argument(
        "--action-mode",
        choices=("relative", "absolute"),
        default="relative",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="also atomically write the complete JSON report to this path",
    )
    parser.add_argument(
        "--require-perfect",
        action="store_true",
        help="exit with status 1 if any episode is not a strict board-fill win",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_oracle(
            width=args.width,
            height=args.height,
            episodes=args.episodes,
            start_seed=args.start_seed,
            max_logic_steps=args.max_logic_steps,
            action_mode=args.action_mode,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    contents = _serialize_json(report)
    if args.output_json is not None:
        _write_json_atomic(args.output_json, contents)
    sys.stdout.write(contents)
    if args.require_perfect and report["summary"]["failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

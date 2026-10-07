"""Strictly evaluate one PPO action-chunk policy without action correction.

Evaluation uses the same one-tick Snake kernel and asynchronous
``ChunkedControlWrapper`` as training.  Every submitted chunk is the raw
deterministic model prediction: there is no oracle, legality filter, retry, or
ensemble.  Batched inference improves measurement throughput but does not
change any seeded episode.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from resnake_gym.envs import SnakeEnv
from resnake_gym_legacy.relative_chunked.chunked_control import ChunkedControlWrapper


@dataclass
class _ActiveEpisode:
    env: ChunkedControlWrapper
    index: int
    observation: dict[str, np.ndarray]
    control_decisions: int = 0
    raw_action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.int64)
    )
    observation_age_sum: int = 0
    observation_age_max: int = 0
    new_observation_count: int = 0
    command_delay_sum: int = 0
    command_delay_max: int = 0
    control_ticks_advanced: int = 0

    def restart(self, *, index: int, observation: dict[str, np.ndarray]) -> None:
        self.index = index
        self.observation = observation
        self.control_decisions = 0
        self.raw_action_counts.fill(0)
        self.observation_age_sum = 0
        self.observation_age_max = 0
        self.new_observation_count = 0
        self.command_delay_sum = 0
        self.command_delay_max = 0
        self.control_ticks_advanced = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument(
        "--logic-fps",
        type=float,
        default=10.0,
        help="simulated game ticks per second; evaluation is not paced",
    )
    parser.add_argument("--control-interval-ticks", type=int, default=3)
    parser.add_argument("--chunk-length", type=int, default=6)
    parser.add_argument("--observation-age-min-ticks", type=int, default=0)
    parser.add_argument("--observation-age-max-ticks", type=int, default=2)
    parser.add_argument("--command-delay-min-ticks", type=int, default=0)
    parser.add_argument("--command-delay-max-ticks", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=100_000)
    parser.add_argument("--max-logic-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-envs",
        type=int,
        default=256,
        help="maximum independent episodes sent through each predict call",
    )
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-win-rate", type=float, default=0.0)
    return parser.parse_args(argv)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _validate_timing_range(low: int, high: int, *, name: str) -> None:
    if low < 0 or high < low:
        raise ValueError(f"{name} range must satisfy 0 <= min <= max")


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "width",
        "height",
        "control_interval_ticks",
        "chunk_length",
        "episodes",
        "batch_envs",
        "torch_threads",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.width < 3 or args.height < 3:
        raise ValueError("width and height must both be at least 3")
    if args.max_logic_steps is not None and args.max_logic_steps < 1:
        raise ValueError("max-logic-steps must be positive")
    if not math.isfinite(args.logic_fps) or args.logic_fps <= 0:
        raise ValueError("logic-fps must be finite and positive")
    _validate_timing_range(
        args.observation_age_min_ticks,
        args.observation_age_max_ticks,
        name="observation-age",
    )
    _validate_timing_range(
        args.command_delay_min_ticks,
        args.command_delay_max_ticks,
        name="command-delay",
    )
    if not 0.0 <= args.require_win_rate <= 1.0:
        raise ValueError("require-win-rate must be in [0, 1]")


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    z = 1.959963984540054
    estimate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (estimate + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _make_env(
    *,
    width: int,
    height: int,
    logic_fps: float,
    max_logic_steps: int,
    control_interval_ticks: int,
    chunk_length: int,
    observation_age_range: tuple[int, int],
    command_delay_range: tuple[int, int],
) -> ChunkedControlWrapper:
    return ChunkedControlWrapper(
        SnakeEnv(
            width=width,
            height=height,
            logic_fps=logic_fps,
            frame_skip=1,
            action_mode="relative",
            max_logic_steps=max_logic_steps,
        ),
        control_interval_ticks=control_interval_ticks,
        chunk_length=chunk_length,
        observation_age_ticks=observation_age_range,
        command_delay_ticks=command_delay_range,
    )


def _stack_observations(
    active_episodes: Sequence[_ActiveEpisode],
) -> dict[str, np.ndarray]:
    keys = active_episodes[0].observation.keys()
    return {
        key: np.stack([active.observation[key] for active in active_episodes], axis=0)
        for key in keys
    }


def _predict_raw_chunks(
    model: Any,
    active_episodes: Sequence[_ActiveEpisode],
    *,
    chunk_length: int,
) -> np.ndarray:
    model_input = _stack_observations(active_episodes)
    predicted, _ = model.predict(model_input, deterministic=True)
    raw = np.asarray(predicted)
    expected_shape = (len(active_episodes), chunk_length)
    if raw.size != math.prod(expected_shape):
        raise RuntimeError(
            f"policy returned shape {raw.shape}; expected {expected_shape}"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise RuntimeError(
            f"policy returned non-integer actions with dtype {raw.dtype}"
        )
    chunks = raw.reshape(expected_shape).astype(np.int64, copy=False)
    if np.any((chunks < 0) | (chunks > 2)):
        raise RuntimeError("policy returned a relative action outside [0, 2]")
    return chunks


def _episode_result(
    *,
    active: _ActiveEpisode,
    seed: int,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    logic_fps: float,
) -> dict[str, Any]:
    won = bool(
        terminated
        and not truncated
        and info.get("won") is True
        and info.get("termination_reason") == "board_filled"
    )
    decisions = active.control_decisions
    logic_steps = int(info["logic_steps"])
    if active.control_ticks_advanced != logic_steps:
        raise RuntimeError(
            "wrapper tick accounting differs from SnakeEnv logic_steps: "
            f"{active.control_ticks_advanced} != {logic_steps}"
        )
    return {
        "seed": seed,
        "won": won,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": info.get("termination_reason"),
        "logic_steps": logic_steps,
        "nominal_game_seconds": logic_steps / logic_fps,
        "control_decisions": decisions,
        "length": int(info["length"]),
        "score": int(info["score"]),
        "raw_submitted_action_count": int(active.raw_action_counts.sum()),
        "raw_submitted_action_counts": {
            "straight": int(active.raw_action_counts[0]),
            "right": int(active.raw_action_counts[1]),
            "left": int(active.raw_action_counts[2]),
        },
        "mean_observation_age_ticks": (
            active.observation_age_sum / decisions if decisions else 0.0
        ),
        "max_observation_age_ticks": active.observation_age_max,
        "new_observation_fraction": (
            active.new_observation_count / decisions if decisions else 0.0
        ),
        "mean_sampled_command_delay_ticks": (
            active.command_delay_sum / decisions if decisions else 0.0
        ),
        "max_sampled_command_delay_ticks": active.command_delay_max,
    }


def _evaluate_episodes(
    model: Any,
    *,
    width: int,
    height: int,
    episode_count: int,
    start_seed: int,
    max_logic_steps: int,
    batch_envs: int,
    logic_fps: float,
    control_interval_ticks: int,
    chunk_length: int,
    observation_age_range: tuple[int, int],
    command_delay_range: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate seeded episodes using only batched raw model predictions."""

    if episode_count < 1:
        raise ValueError("episode-count must be positive")
    if batch_envs < 1:
        raise ValueError("batch-envs must be positive")

    envs = [
        _make_env(
            width=width,
            height=height,
            logic_fps=logic_fps,
            max_logic_steps=max_logic_steps,
            control_interval_ticks=control_interval_ticks,
            chunk_length=chunk_length,
            observation_age_range=observation_age_range,
            command_delay_range=command_delay_range,
        )
        for _ in range(min(batch_envs, episode_count))
    ]
    episode_results: list[dict[str, Any] | None] = [None] * episode_count
    active_episodes: list[_ActiveEpisode] = []
    next_index = 0
    inference_seconds: list[float] = []
    inference_batch_sizes: list[int] = []
    started = time.perf_counter()
    try:
        for env in envs:
            observation, _ = env.reset(seed=start_seed + next_index)
            active_episodes.append(
                _ActiveEpisode(env=env, index=next_index, observation=observation)
            )
            next_index += 1

        while active_episodes:
            inference_started = time.perf_counter()
            chunks = _predict_raw_chunks(
                model,
                active_episodes,
                chunk_length=chunk_length,
            )
            inference_seconds.append(time.perf_counter() - inference_started)
            inference_batch_sizes.append(len(active_episodes))

            next_active: list[_ActiveEpisode] = []
            for active, chunk in zip(active_episodes, chunks, strict=True):
                # ``chunk`` is passed through exactly as predicted.  The only
                # validation is membership in the declared MultiDiscrete space.
                if not active.env.action_space.contains(chunk):
                    raise RuntimeError(f"model chunk is outside action space: {chunk}")
                active.control_decisions += 1
                active.raw_action_counts += np.bincount(chunk, minlength=3)
                observation, _, terminated, truncated, info = active.env.step(chunk)
                observation_age = int(info["observation_age"])
                command_delay = int(info["sampled_command_delay_ticks"])
                active.observation_age_sum += observation_age
                active.observation_age_max = max(
                    active.observation_age_max, observation_age
                )
                active.new_observation_count += int(bool(info["new_observation"]))
                active.command_delay_sum += command_delay
                active.command_delay_max = max(active.command_delay_max, command_delay)
                active.control_ticks_advanced += int(info["control_ticks_advanced"])

                if terminated or truncated:
                    episode_index = active.index
                    episode_results[episode_index] = _episode_result(
                        active=active,
                        seed=start_seed + episode_index,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        logic_fps=logic_fps,
                    )
                    if next_index < episode_count:
                        observation, _ = active.env.reset(seed=start_seed + next_index)
                        active.restart(index=next_index, observation=observation)
                        next_index += 1
                        next_active.append(active)
                else:
                    active.observation = observation
                    next_active.append(active)
            active_episodes = next_active
    finally:
        for env in envs:
            env.close()

    wall_seconds = time.perf_counter() - started
    if any(result is None for result in episode_results):
        raise RuntimeError("evaluation ended before every episode completed")
    completed = [result for result in episode_results if result is not None]
    policy_decisions = sum(int(result["control_decisions"]) for result in completed)
    inference_deadline = control_interval_ticks / logic_fps
    deadline_miss_calls = sum(
        duration > inference_deadline for duration in inference_seconds
    )
    timing = {
        "wall_seconds": wall_seconds,
        "inference_wall_seconds": sum(inference_seconds),
        "inference_calls": len(inference_seconds),
        "policy_decisions": policy_decisions,
        "mean_inference_batch_size": statistics.fmean(inference_batch_sizes),
        "max_inference_batch_size": max(inference_batch_sizes),
        "mean_inference_seconds_per_call": statistics.fmean(inference_seconds),
        "p50_inference_seconds_per_call": _percentile(inference_seconds, 50),
        "p95_inference_seconds_per_call": _percentile(inference_seconds, 95),
        "max_inference_seconds_per_call": max(inference_seconds),
        "inference_deadline_seconds": inference_deadline,
        "inference_deadline_miss_calls": deadline_miss_calls,
        "inference_deadline_miss_fraction": (
            deadline_miss_calls / len(inference_seconds)
        ),
        "amortized_inference_seconds_per_policy_decision": (
            sum(inference_seconds) / policy_decisions
        ),
    }
    return completed, timing


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)

    try:
        import gymnasium
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'RL dependencies are missing; install with `pip install -e ".[rl]"`'
        ) from exc

    torch.set_num_threads(args.torch_threads)
    model = PPO.load(args.model, device=args.device)

    max_logic_steps = args.max_logic_steps
    if max_logic_steps is None:
        capacity = args.width * args.height
        max_logic_steps = capacity * (capacity + 1) // 2
    observation_age_range = (
        args.observation_age_min_ticks,
        args.observation_age_max_ticks,
    )
    command_delay_range = (
        args.command_delay_min_ticks,
        args.command_delay_max_ticks,
    )
    episodes, timing = _evaluate_episodes(
        model,
        width=args.width,
        height=args.height,
        episode_count=args.episodes,
        start_seed=args.start_seed,
        max_logic_steps=max_logic_steps,
        batch_envs=args.batch_envs,
        logic_fps=args.logic_fps,
        control_interval_ticks=args.control_interval_ticks,
        chunk_length=args.chunk_length,
        observation_age_range=observation_age_range,
        command_delay_range=command_delay_range,
    )

    wins = [episode for episode in episodes if episode["won"]]
    successes = len(wins)
    win_rate = successes / args.episodes
    win_rate_low, win_rate_high = _wilson_interval(successes, args.episodes)
    all_ticks = [int(episode["logic_steps"]) for episode in episodes]
    win_ticks = [int(episode["logic_steps"]) for episode in wins]
    control_period_seconds = args.control_interval_ticks / args.logic_fps
    result = {
        "schema_version": 1,
        "algorithm": "PPO",
        "model": str(args.model.resolve()),
        "policy": "MultiInputPolicy",
        "action_selection": "single_model_raw_deterministic_prediction",
        "uses_ensemble": False,
        "uses_action_filter": False,
        "uses_oracle": False,
        "strict_success_definition": (
            "terminated and not truncated and won is true and "
            "termination_reason equals board_filled"
        ),
        "width": args.width,
        "height": args.height,
        "board_cells": args.width * args.height,
        "logic_fps": args.logic_fps,
        "frame_skip": 1,
        "control_interval_ticks": args.control_interval_ticks,
        "chunk_length": args.chunk_length,
        "observation_age_range_ticks": observation_age_range,
        "command_delay_range_ticks": command_delay_range,
        "decision_fps": args.logic_fps / args.control_interval_ticks,
        "control_period_seconds": control_period_seconds,
        "action_horizon_seconds": args.chunk_length / args.logic_fps,
        "chunk_replan_overlap_ticks": max(
            args.chunk_length - args.control_interval_ticks, 0
        ),
        "observation_age_range_seconds": [
            value / args.logic_fps for value in observation_age_range
        ],
        "command_delay_range_seconds": [
            value / args.logic_fps for value in command_delay_range
        ],
        "headless_not_wall_clock_paced": True,
        "realtime_capability_claimed": False,
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "max_logic_steps": max_logic_steps,
        "batch_envs": args.batch_envs,
        "torch_threads": args.torch_threads,
        "wins": successes,
        "win_rate": win_rate,
        "win_rate_wilson95_low": win_rate_low,
        "win_rate_wilson95_high": win_rate_high,
        "mean_final_length": statistics.fmean(
            int(episode["length"]) for episode in episodes
        ),
        "mean_logic_ticks": statistics.fmean(all_ticks),
        "median_win_ticks": statistics.median(win_ticks) if win_ticks else None,
        "p95_win_ticks": _percentile(win_ticks, 95) if win_ticks else None,
        "total_logic_ticks": sum(all_ticks),
        "simulated_game_seconds": sum(all_ticks) / args.logic_fps,
        "logic_ticks_per_wall_second": sum(all_ticks) / timing["wall_seconds"],
        "timing_measurement": timing,
        "inference_deadline_audit": {
            "deadline_seconds": control_period_seconds,
            "p95_batch_call_meets_deadline": (
                timing["p95_inference_seconds_per_call"] <= control_period_seconds
            ),
            "max_batch_call_meets_deadline": (
                timing["max_inference_seconds_per_call"] <= control_period_seconds
            ),
            "deadline_miss_calls": timing["inference_deadline_miss_calls"],
            "deadline_miss_fraction": timing["inference_deadline_miss_fraction"],
            "interpretation": (
                "batched headless latency is diagnostic only; deployment must "
                "measure batch-one end-to-end sensing and actuation latency"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "resnake_gym": _package_version("resnake-gym"),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "episode_results": episodes,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if win_rate < args.require_win_rate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Measure batch-1 policy-control latency against explicit tick deadlines.

This benchmark is intentionally separate from batched evaluation. It runs one
environment, asks the policy for one action per decision, and reports latency
distributions. The unpaced section measures available compute headroom; the
optional paced section additionally checks a wall-clock release schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from resnake_gym.envs import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--algorithm", choices=("ppo", "qrdqn"), default="ppo")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--logic-fps", type=float, default=10.0)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-logic-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=800_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--measure-steps", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--deadline-fps",
        action="append",
        type=float,
        default=None,
        help="deadline to compare with unpaced batch-1 latency; repeatable",
    )
    parser.add_argument(
        "--paced-fps",
        action="append",
        type=float,
        default=[],
        help="run a wall-clock-paced loop at this rate; repeatable",
    )
    parser.add_argument(
        "--paced-seconds",
        type=float,
        default=5.0,
        help="duration of each optional paced run",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency sample cannot be empty")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": max(values),
    }


def _deadline_summary(
    loop_ms: list[float], target_fps: float
) -> dict[str, float | int]:
    _positive_finite(target_fps, "target-fps")
    period_ms = 1000.0 / target_fps
    misses = sum(latency > period_ms for latency in loop_ms)
    return {
        "target_fps": target_fps,
        "period_ms": period_ms,
        "deadline_misses": misses,
        "deadline_miss_rate": misses / len(loop_ms),
        "p99_headroom_ms": period_ms - _percentile(loop_ms, 99),
    }


def _action(model: Any, observation: np.ndarray) -> int:
    predicted, _ = model.predict(observation, deterministic=True)
    values = np.asarray(predicted).reshape(-1)
    if values.size != 1:
        raise RuntimeError(f"batch-1 policy returned {values.size} actions")
    return int(values[0])


def _advance(
    model: Any,
    env: SnakeEnv,
    observation: np.ndarray,
    synchronize: Callable[[], None],
) -> tuple[np.ndarray, bool, float, float, float]:
    synchronize()
    start_ns = time.perf_counter_ns()
    action = _action(model, observation)
    synchronize()
    predicted_ns = time.perf_counter_ns()
    next_observation, _, terminated, truncated, _ = env.step(action)
    end_ns = time.perf_counter_ns()
    scale = 1e-6
    return (
        next_observation,
        terminated or truncated,
        (predicted_ns - start_ns) * scale,
        (end_ns - predicted_ns) * scale,
        (end_ns - start_ns) * scale,
    )


def _reset(env: SnakeEnv, seed: int) -> np.ndarray:
    observation, _ = env.reset(seed=seed)
    return observation


def run_unpaced(
    model: Any,
    env: SnakeEnv,
    *,
    seed: int,
    warmup_steps: int,
    measure_steps: int,
    synchronize: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if measure_steps < 1:
        raise ValueError("measure-steps must be positive")

    observation = _reset(env, seed)
    next_seed = seed + 1
    resets = 0
    for _ in range(warmup_steps):
        observation, done, _, _, _ = _advance(model, env, observation, synchronize)
        if done:
            observation = _reset(env, next_seed)
            next_seed += 1

    predict_ms: list[float] = []
    env_step_ms: list[float] = []
    loop_ms: list[float] = []
    synchronize()
    wall_start = time.perf_counter()
    for _ in range(measure_steps):
        observation, done, predict, env_step, loop = _advance(
            model, env, observation, synchronize
        )
        predict_ms.append(predict)
        env_step_ms.append(env_step)
        loop_ms.append(loop)
        if done:
            observation = _reset(env, next_seed)
            next_seed += 1
            resets += 1
    synchronize()
    wall_seconds = time.perf_counter() - wall_start
    return {
        "warmup_steps": warmup_steps,
        "measured_decisions": measure_steps,
        "episode_resets": resets,
        "reset_time_in_latency_samples": False,
        "wall_seconds_including_resets": wall_seconds,
        "decisions_per_wall_second_including_resets": measure_steps / wall_seconds,
        "policy_predict": _summarize_ms(predict_ms),
        "environment_step": _summarize_ms(env_step_ms),
        "predict_plus_environment_step": _summarize_ms(loop_ms),
        "_loop_ms": loop_ms,
    }


def run_paced(
    model: Any,
    env: SnakeEnv,
    *,
    seed: int,
    target_fps: float,
    duration_seconds: float,
    synchronize: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    _positive_finite(target_fps, "target-fps")
    _positive_finite(duration_seconds, "duration-seconds")
    ticks = max(1, math.ceil(target_fps * duration_seconds))
    period = 1.0 / target_fps
    observation = _reset(env, seed)
    next_seed = seed + 1
    releases_late_ms: list[float] = []
    completion_ms: list[float] = []
    compute_ms: list[float] = []
    deadline_misses = 0
    resets = 0

    synchronize()
    first_release = time.perf_counter()
    for index in range(ticks):
        release = first_release + index * period
        remaining = release - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        started = time.perf_counter()
        releases_late_ms.append(max(0.0, started - release) * 1000.0)
        observation, done, _, _, compute = _advance(
            model, env, observation, synchronize
        )
        finished = time.perf_counter()
        completion_ms.append((finished - release) * 1000.0)
        compute_ms.append(compute)
        if finished > release + period:
            deadline_misses += 1
        if done:
            observation = _reset(env, next_seed)
            next_seed += 1
            resets += 1

    wall_seconds = time.perf_counter() - first_release
    return {
        "target_fps": target_fps,
        "period_ms": period * 1000.0,
        "requested_seconds": duration_seconds,
        "scheduled_decisions": ticks,
        "episode_resets": resets,
        "wall_seconds": wall_seconds,
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_misses / ticks,
        "release_lateness": _summarize_ms(releases_late_ms),
        "compute": _summarize_ms(compute_ms),
        "release_to_completion": _summarize_ms(completion_ms),
    }


def main() -> None:
    args = parse_args()
    for name in ("width", "height", "frame_skip", "measure_steps", "torch_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    _positive_finite(args.logic_fps, "logic-fps")
    _positive_finite(args.paced_seconds, "paced-seconds")
    deadline_fps = args.deadline_fps or [args.logic_fps / args.frame_skip]
    for rate in (*deadline_fps, *args.paced_fps):
        _positive_finite(rate, "deadline-fps")

    try:
        import stable_baselines3
        import torch

        torch.set_num_threads(args.torch_threads)
        if args.algorithm == "ppo":
            from stable_baselines3 import PPO

            model = PPO.load(args.model, device=args.device)
        else:
            import sb3_contrib
            from sb3_contrib import QRDQN

            model = QRDQN.load(args.model, device=args.device)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'RL dependencies are missing; install with `pip install -e ".[rl]"`'
        ) from exc

    device = getattr(model, "device", None)
    uses_cuda = getattr(device, "type", None) == "cuda"
    synchronize: Callable[[], None]
    synchronize = torch.cuda.synchronize if uses_cuda else lambda: None

    max_logic_steps = args.max_logic_steps
    if max_logic_steps is None:
        capacity = args.width * args.height
        max_logic_steps = capacity * (capacity + 1) // 2
    env = SnakeEnv(
        width=args.width,
        height=args.height,
        logic_fps=args.logic_fps,
        frame_skip=args.frame_skip,
        action_mode="relative",
        max_logic_steps=max_logic_steps,
    )
    try:
        unpaced = run_unpaced(
            model,
            env,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            synchronize=synchronize,
        )
        loop_ms = unpaced.pop("_loop_ms")
        unpaced["deadline_comparisons"] = [
            _deadline_summary(loop_ms, rate) for rate in deadline_fps
        ]
        paced = [
            run_paced(
                model,
                env,
                seed=args.seed + 1_000_000 + index * 100_000,
                target_fps=rate,
                duration_seconds=args.paced_seconds,
                synchronize=synchronize,
            )
            for index, rate in enumerate(args.paced_fps)
        ]
    finally:
        env.close()

    result = {
        "benchmark_scope": (
            "single environment, batch=1, one deterministic model decision per "
            "environment step; excludes rendering, HID, network, and robot latency"
        ),
        "algorithm": args.algorithm,
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "width": args.width,
        "height": args.height,
        "logic_fps": args.logic_fps,
        "frame_skip": args.frame_skip,
        "nominal_decision_fps": args.logic_fps / args.frame_skip,
        "max_logic_steps": max_logic_steps,
        "device_requested": args.device,
        "device_resolved": str(device),
        "torch_threads": args.torch_threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": (sb3_contrib.__version__ if args.algorithm == "qrdqn" else None),
        "unpaced": unpaced,
        "paced": paced,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

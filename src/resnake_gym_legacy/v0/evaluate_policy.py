"""Evaluate an SB3 policy with strict board-filled Snake success."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from resnake_gym.envs import SnakeEnv


@dataclass
class _ActiveEpisode:
    env: SnakeEnv
    index: int
    observation: np.ndarray


class _MajorityVotePolicy:
    """Combine deterministic discrete-action policies by majority vote.

    The first policy breaks ties. With three policies this matters only when
    all three actions differ, and keeps ensemble evaluation reproducible.
    """

    def __init__(self, policies: list[Any]) -> None:
        if len(policies) < 2:
            raise ValueError("majority vote requires at least two policies")
        self.policies = policies

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray | np.integer, None]:
        scalar = observation.ndim == 3
        expected = 1 if scalar else observation.shape[0]
        predictions = []
        for policy in self.policies:
            predicted, _ = policy.predict(observation, deterministic=deterministic)
            actions = np.asarray(predicted).reshape(-1)
            if actions.size != expected:
                raise RuntimeError(
                    "ensemble member returned "
                    f"{actions.size} actions for {expected} observations"
                )
            predictions.append(actions.astype(np.int64, copy=False))

        stacked = np.stack(predictions, axis=0)
        voted = np.empty(expected, dtype=np.int64)
        for index in range(expected):
            values, counts = np.unique(stacked[:, index], return_counts=True)
            candidates = values[counts == counts.max()]
            primary = stacked[0, index]
            voted[index] = primary if primary in candidates else candidates[0]
        return (voted[0] if scalar else voted), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--ensemble-model",
        action="append",
        type=Path,
        default=[],
        help="additional checkpoint; deterministic actions are majority-voted",
    )
    parser.add_argument("--algorithm", choices=("ppo", "qrdqn"), default="ppo")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument(
        "--logic-fps",
        type=float,
        default=10.0,
        help="simulated game ticks per second; evaluation is not wall-clock paced",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="logic ticks advanced per policy decision",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=100_000)
    parser.add_argument("--max-logic-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-envs",
        type=int,
        default=256,
        help="maximum number of independent episodes inferred as one batch",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=8,
        help="PyTorch intra-op CPU threads",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-win-rate", type=float, default=0.0)
    return parser.parse_args()


def _percentile(values: list[int], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


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


def _predict_actions(
    model: Any,
    active_episodes: list[_ActiveEpisode],
    *,
    scalar: bool,
) -> np.ndarray:
    if scalar:
        model_input = active_episodes[0].observation
    else:
        model_input = np.stack(
            [episode.observation for episode in active_episodes],
            axis=0,
        )
    predicted, _ = model.predict(model_input, deterministic=True)
    actions = np.asarray(predicted).reshape(-1)
    if actions.size != len(active_episodes):
        raise RuntimeError(
            "policy returned "
            f"{actions.size} actions for {len(active_episodes)} observations"
        )
    return actions


def _episode_result(
    *,
    seed: int,
    terminated: bool,
    info: dict[str, object],
) -> dict[str, object]:
    won = bool(
        terminated
        and info.get("won") is True
        and info.get("termination_reason") == "board_filled"
    )
    return {
        "seed": seed,
        "won": won,
        "logic_steps": int(info["logic_steps"]),
        "length": int(info["length"]),
        "score": int(info["score"]),
        "termination_reason": info["termination_reason"],
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
    logic_fps: float = 10.0,
    frame_skip: int = 1,
) -> tuple[list[dict[str, object]], float]:
    """Evaluate seeded episodes with batched inference and manual resets."""

    if episode_count < 1:
        raise ValueError("episode-count must be positive")
    if batch_envs < 1:
        raise ValueError("batch-envs must be positive")

    envs = [
        SnakeEnv(
            width=width,
            height=height,
            logic_fps=logic_fps,
            frame_skip=frame_skip,
            action_mode="relative",
            max_logic_steps=max_logic_steps,
        )
        for _ in range(min(batch_envs, episode_count))
    ]
    episode_results: list[dict[str, object] | None] = [None] * episode_count
    active_episodes: list[_ActiveEpisode] = []
    next_index = 0
    started = time.perf_counter()
    try:
        for env in envs:
            observation, _ = env.reset(seed=start_seed + next_index)
            active_episodes.append(
                _ActiveEpisode(
                    env=env,
                    index=next_index,
                    observation=observation,
                )
            )
            next_index += 1

        while active_episodes:
            actions = _predict_actions(
                model,
                active_episodes,
                scalar=batch_envs == 1,
            )
            next_active: list[_ActiveEpisode] = []
            for active, action in zip(active_episodes, actions, strict=True):
                observation, _, terminated, truncated, info = active.env.step(
                    int(action)
                )
                if terminated or truncated:
                    seed = start_seed + active.index
                    episode_results[active.index] = _episode_result(
                        seed=seed,
                        terminated=terminated,
                        info=info,
                    )
                    if next_index < episode_count:
                        observation, _ = active.env.reset(seed=start_seed + next_index)
                        active.index = next_index
                        active.observation = observation
                        next_index += 1
                        next_active.append(active)
                else:
                    active.observation = observation
                    next_active.append(active)
            active_episodes = next_active
    finally:
        for env in envs:
            env.close()

    elapsed = time.perf_counter() - started
    if any(result is None for result in episode_results):
        raise RuntimeError("evaluation ended before every episode completed")
    return [result for result in episode_results if result is not None], elapsed


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.batch_envs < 1:
        raise ValueError("batch-envs must be positive")
    if not math.isfinite(args.logic_fps) or args.logic_fps <= 0:
        raise ValueError("logic-fps must be finite and positive")
    if args.frame_skip < 1:
        raise ValueError("frame-skip must be positive")
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    if not 0.0 <= args.require_win_rate <= 1.0:
        raise ValueError("require-win-rate must be in [0, 1]")

    try:
        import torch

        torch.set_num_threads(args.torch_threads)
        if args.algorithm == "ppo":
            from stable_baselines3 import PPO

            models = [
                PPO.load(path, device=args.device)
                for path in (args.model, *args.ensemble_model)
            ]
        else:
            from sb3_contrib import QRDQN

            models = [
                QRDQN.load(path, device=args.device)
                for path in (args.model, *args.ensemble_model)
            ]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'RL dependencies are missing; install with `pip install -e ".[rl]"`'
        ) from exc

    model = models[0] if len(models) == 1 else _MajorityVotePolicy(models)
    max_logic_steps = args.max_logic_steps
    if max_logic_steps is None:
        capacity = args.width * args.height
        max_logic_steps = capacity * (capacity + 1) // 2
    episodes, elapsed = _evaluate_episodes(
        model,
        width=args.width,
        height=args.height,
        episode_count=args.episodes,
        start_seed=args.start_seed,
        max_logic_steps=max_logic_steps,
        batch_envs=args.batch_envs,
        logic_fps=args.logic_fps,
        frame_skip=args.frame_skip,
    )
    wins = [episode for episode in episodes if episode["won"]]
    win_rate = len(wins) / args.episodes
    win_rate_low, win_rate_high = _wilson_interval(len(wins), args.episodes)
    all_steps = [int(episode["logic_steps"]) for episode in episodes]
    win_steps = [int(episode["logic_steps"]) for episode in wins]
    result = {
        "algorithm": args.algorithm,
        "model": str(args.model.resolve()),
        "models": [str(path.resolve()) for path in (args.model, *args.ensemble_model)],
        "ensemble_size": len(models),
        "width": args.width,
        "height": args.height,
        "logic_fps": args.logic_fps,
        "frame_skip": args.frame_skip,
        "decision_fps": args.logic_fps / args.frame_skip,
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "max_logic_steps": max_logic_steps,
        "batch_envs": args.batch_envs,
        "torch_threads": args.torch_threads,
        "wins": len(wins),
        "win_rate": win_rate,
        "win_rate_wilson95_low": win_rate_low,
        "win_rate_wilson95_high": win_rate_high,
        "mean_final_length": statistics.fmean(
            int(episode["length"]) for episode in episodes
        ),
        "mean_logic_steps": statistics.fmean(all_steps),
        "median_win_steps": statistics.median(win_steps) if win_steps else None,
        "p95_win_steps": _percentile(win_steps, 95) if win_steps else None,
        "wall_seconds": elapsed,
        "logic_steps_per_second": sum(all_steps) / elapsed,
        "episode_results": episodes,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["win_rate"] < args.require_win_rate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

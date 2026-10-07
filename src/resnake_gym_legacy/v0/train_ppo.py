"""Train a reproducible PPO baseline on ReSnake.

Training outputs can grow quickly. On the project GPU node, pass an output
directory below ``/data/lyy/resnake_gym/artifacts/runs``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from resnake_gym.envs import SnakeEnv
from resnake_gym_legacy.v0.distance_reward import DistanceRewardWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument(
        "--logic-fps",
        type=float,
        default=10.0,
        help=(
            "simulated game ticks per second; headless training is not wall-clock paced"
        ),
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="logic ticks advanced per policy decision",
    )
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--max-logic-steps", type=int, default=None)
    parser.add_argument("--food-reward", type=float, default=0.1)
    parser.add_argument("--death-reward", type=float, default=-1.0)
    parser.add_argument("--living-reward", type=float, default=-1e-4)
    parser.add_argument("--win-reward", type=float, default=10.0)
    parser.add_argument("--timeout-reward", type=float, default=0.0)
    parser.add_argument("--distance-reward-scale", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        type=Path,
        help="PPO .zip checkpoint to continue training from",
    )
    return parser.parse_args()


def _positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def main() -> None:
    args = parse_args()
    for name in (
        "width",
        "height",
        "total_timesteps",
        "n_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "torch_threads",
        "checkpoint_every",
        "frame_skip",
    ):
        _positive(getattr(args, name), name.replace("_", "-"))
    for name in ("logic_fps", "timeout_reward", "distance_reward_scale"):
        _finite(getattr(args, name), name.replace("_", "-"))
    if args.logic_fps <= 0:
        raise ValueError("logic-fps must be positive")

    try:
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'RL dependencies are missing; install with `pip install -e ".[rl]"`'
        ) from exc

    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    monitor_dir = args.output_dir / "monitor"
    tensorboard_dir = args.output_dir / "tensorboard"
    checkpoints_dir.mkdir(exist_ok=True)
    monitor_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)
    monitor_run_dir = monitor_dir
    if args.resume is not None and any(monitor_dir.glob("*.monitor.csv")):
        monitor_run_dir = monitor_dir / f"resume-{time.time_ns()}"
        monitor_run_dir.mkdir()

    max_logic_steps = args.max_logic_steps
    if max_logic_steps is None:
        capacity = args.width * args.height
        max_logic_steps = capacity * (capacity + 1) // 2

    env_kwargs: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "logic_fps": args.logic_fps,
        "frame_skip": args.frame_skip,
        "action_mode": "relative",
        "max_logic_steps": max_logic_steps,
        "food_reward": args.food_reward,
        "death_reward": args.death_reward,
        "living_reward": args.living_reward,
        "win_reward": args.win_reward,
        "timeout_reward": args.timeout_reward,
    }
    use_distance_reward = args.distance_reward_scale != 0.0
    vec_env = make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        monitor_dir=str(monitor_run_dir),
        wrapper_class=DistanceRewardWrapper if use_distance_reward else None,
        wrapper_kwargs=(
            {"scale": args.distance_reward_scale} if use_distance_reward else None
        ),
    )

    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.SiLU,
    }
    if args.resume is None:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    else:
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=args.device,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
        )

    save_freq = max(args.checkpoint_every // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoints_dir),
        name_prefix="ppo_resnake",
    )

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir.resolve())
    config["resume"] = None if args.resume is None else str(args.resume.resolve())
    config["monitor_dir"] = str(monitor_run_dir.resolve())
    config["max_logic_steps"] = max_logic_steps
    config["env_kwargs"] = env_kwargs
    config["python"] = platform.python_version()
    config["torch"] = torch.__version__
    config["stable_baselines3"] = stable_baselines3.__version__
    config["cuda_available"] = torch.cuda.is_available()
    config["started_unix"] = time.time()
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=args.resume is None,
            tb_log_name="ppo",
        )
        model.save(args.output_dir / "final_model")
    finally:
        vec_env.close()

    summary = {
        "wall_seconds": time.perf_counter() - started,
        "total_timesteps_requested": args.total_timesteps,
        "model_num_timesteps": model.num_timesteps,
        "model_path": str((args.output_dir / "final_model.zip").resolve()),
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

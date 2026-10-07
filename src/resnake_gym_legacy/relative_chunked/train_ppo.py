"""Train online PPO under delayed observations and chunked Snake control.

The game kernel always advances one logic tick at a time.  The policy is
called once every ``control_interval_ticks`` and directly submits a
``chunk_length`` vector of relative actions.  No demonstrations, oracle,
distance shaping, or action correction are used.

Training outputs can grow quickly.  On the project GPU node, place
``--output-dir`` below ``/data/lyy/resnake_gym/artifacts/runs``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from resnake_gym.envs import SnakeEnv
from resnake_gym_legacy.relative_chunked.chunked_control import ChunkedControlWrapper


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument(
        "--logic-fps",
        type=float,
        default=10.0,
        help="simulated game ticks per second; headless training is not paced",
    )
    parser.add_argument(
        "--control-interval-ticks",
        type=int,
        default=3,
        help="fixed K: logic ticks advanced between policy calls",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=6,
        help="fixed L: relative actions predicted at each policy call",
    )
    parser.add_argument("--observation-age-min-ticks", type=int, default=0)
    parser.add_argument("--observation-age-max-ticks", type=int, default=2)
    parser.add_argument("--command-delay-min-ticks", type=int, default=0)
    parser.add_argument("--command-delay-max-ticks", type=int, default=2)

    parser.add_argument(
        "--total-policy-steps",
        type=int,
        default=2_000_000,
        help="SB3 timesteps; each is one vector-env policy/control decision",
    )
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gamma-per-tick",
        type=float,
        default=0.995,
        help="discount for one logic tick; PPO receives gamma_per_tick ** K",
    )
    parser.add_argument(
        "--gae-lambda-per-tick",
        type=float,
        default=0.95,
        help="GAE lambda for one logic tick; PPO receives lambda_per_tick ** K",
    )
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
    parser.add_argument(
        "--resume",
        type=Path,
        help="PPO .zip checkpoint with matching Dict/MultiDiscrete spaces",
    )
    return parser.parse_args(argv)


def _discount_per_control_period(value_per_tick: float, ticks: int) -> float:
    """Convert a per-logic-tick trace factor to one policy-period factor."""

    value_per_tick = float(value_per_tick)
    if not math.isfinite(value_per_tick) or not 0.0 <= value_per_tick <= 1.0:
        raise ValueError("per-tick discount must be finite and in [0, 1]")
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
        raise ValueError("ticks must be a positive integer")
    return value_per_tick**ticks


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "width",
        "height",
        "control_interval_ticks",
        "chunk_length",
        "total_policy_steps",
        "n_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "torch_threads",
        "checkpoint_every",
    )
    for name in positive_names:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.width < 3 or args.height < 3:
        raise ValueError("width and height must both be at least 3")
    if args.max_logic_steps is not None and args.max_logic_steps < 1:
        raise ValueError("max-logic-steps must be positive")

    for low_name, high_name in (
        ("observation_age_min_ticks", "observation_age_max_ticks"),
        ("command_delay_min_ticks", "command_delay_max_ticks"),
    ):
        low = getattr(args, low_name)
        high = getattr(args, high_name)
        if low < 0 or high < low:
            label = low_name.removesuffix("_min_ticks").replace("_", "-")
            raise ValueError(f"{label} range must satisfy 0 <= min <= max")

    finite_names = (
        "logic_fps",
        "learning_rate",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "food_reward",
        "death_reward",
        "living_reward",
        "win_reward",
        "timeout_reward",
    )
    for name in finite_names:
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name.replace('_', '-')} must be finite")
    if args.logic_fps <= 0:
        raise ValueError("logic-fps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 < args.clip_range <= 1.0:
        raise ValueError("clip-range must be in (0, 1]")
    if args.ent_coef < 0 or args.vf_coef < 0 or args.max_grad_norm <= 0:
        raise ValueError("PPO coefficients must be non-negative and norm positive")
    _discount_per_control_period(args.gamma_per_tick, args.control_interval_ticks)
    _discount_per_control_period(args.gae_lambda_per_tick, args.control_interval_ticks)


def _make_chunked_env(
    *,
    env_kwargs: dict[str, Any],
    timing_kwargs: dict[str, Any],
) -> ChunkedControlWrapper:
    return ChunkedControlWrapper(SnakeEnv(**env_kwargs), **timing_kwargs)


def _timing_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build wrapper timing settings, including the per-tick reward discount."""

    return {
        "control_interval_ticks": args.control_interval_ticks,
        "chunk_length": args.chunk_length,
        "observation_age_ticks": (
            args.observation_age_min_ticks,
            args.observation_age_max_ticks,
        ),
        "command_delay_ticks": (
            args.command_delay_min_ticks,
            args.command_delay_max_ticks,
        ),
        "reward_discount_per_tick": args.gamma_per_tick,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)

    try:
        import gymnasium
        import numpy
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'RL dependencies are missing; install with `pip install -e ".[rl]"`'
        ) from exc

    class _LogicTickCounter(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.logic_ticks = 0
            self.control_decisions = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", ())
            self.control_decisions += len(infos)
            self.logic_ticks += sum(
                int(info.get("control_ticks_advanced", 0)) for info in infos
            )
            return True

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
        "frame_skip": 1,
        "action_mode": "relative",
        "max_logic_steps": max_logic_steps,
        "food_reward": args.food_reward,
        "death_reward": args.death_reward,
        "living_reward": args.living_reward,
        "win_reward": args.win_reward,
        "timeout_reward": args.timeout_reward,
    }
    timing_kwargs = _timing_kwargs_from_args(args)
    vec_env = make_vec_env(
        lambda: _make_chunked_env(
            env_kwargs=env_kwargs,
            timing_kwargs=timing_kwargs,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(monitor_run_dir),
    )

    gamma_control = _discount_per_control_period(
        args.gamma_per_tick, args.control_interval_ticks
    )
    gae_lambda_control = _discount_per_control_period(
        args.gae_lambda_per_tick, args.control_interval_ticks
    )
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.SiLU,
        "optimizer_kwargs": {"eps": 1e-5},
    }
    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": gamma_control,
        "gae_lambda": gae_lambda_control,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }
    if args.resume is None:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            **ppo_kwargs,
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
            **ppo_kwargs,
        )

    save_freq = max(args.checkpoint_every // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoints_dir),
        name_prefix="ppo_chunked_resnake",
    )
    tick_counter = _LogicTickCounter()
    callbacks = CallbackList([checkpoint_callback, tick_counter])

    control_period_seconds = args.control_interval_ticks / args.logic_fps
    action_horizon_seconds = args.chunk_length / args.logic_fps
    config = {
        "schema_version": 1,
        "training_method": "online_ppo_from_environment_interaction",
        "uses_demonstrations": False,
        "uses_oracle": False,
        "uses_action_filter": False,
        "distance_reward_scale": 0.0,
        "policy": "MultiInputPolicy",
        "observation_space_kind": "Dict",
        "action_space_kind": "MultiDiscrete",
        "action_semantics": "raw_relative_action_chunk",
        "reward_aggregation": (
            "sum(gamma_per_tick ** tick_offset * tick_reward[tick_offset])"
        ),
        "sb3_timestep_unit": "one control decision in one vector environment",
        "frame_skip": 1,
        "decision_fps": args.logic_fps / args.control_interval_ticks,
        "control_period_seconds": control_period_seconds,
        "action_horizon_seconds": action_horizon_seconds,
        "chunk_replan_overlap_ticks": max(
            args.chunk_length - args.control_interval_ticks, 0
        ),
        "gamma_control_period": gamma_control,
        "gae_lambda_control_period": gae_lambda_control,
        "discount_conversion": "per_tick_value ** control_interval_ticks",
        "env_kwargs": env_kwargs,
        "timing_kwargs": timing_kwargs,
        "ppo_kwargs": ppo_kwargs,
        "policy_kwargs_applied_to_new_model": (
            {
                "net_arch": policy_kwargs["net_arch"],
                "activation_fn": "torch.nn.SiLU",
                "optimizer": "torch.optim.Adam",
                "optimizer_kwargs": policy_kwargs["optimizer_kwargs"],
                "normalize_images": True,
            }
            if args.resume is None
            else None
        ),
        "policy_architecture_source": (
            "this config" if args.resume is None else "resume checkpoint"
        ),
        "sb3_effective_defaults": {
            "normalize_advantage": True,
            "use_sde": False,
            "sde_sample_freq": -1,
            "rollout_buffer_class": "DictRolloutBuffer",
            "target_kl": None,
        },
        "output_dir": str(args.output_dir.resolve()),
        "resume": None if args.resume is None else str(args.resume.resolve()),
        "monitor_dir": str(monitor_run_dir.resolve()),
        "total_policy_steps_requested": args.total_policy_steps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device_requested": args.device,
        "torch_threads": args.torch_threads,
        "checkpoint_every_policy_steps": args.checkpoint_every,
        "checkpoint_callback_save_freq": save_freq,
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "resnake_gym": _package_version("resnake-gym"),
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "started_unix": time.time(),
        "raw_cli": vars(args),
    }
    # pathlib objects in the raw CLI are made explicit without silently
    # changing any experimental value.
    config["raw_cli"] = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.total_policy_steps,
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
            tb_log_name="chunked_ppo",
        )
        model.save(args.output_dir / "final_model")
    finally:
        vec_env.close()

    wall_seconds = time.perf_counter() - started
    summary = {
        "wall_seconds": wall_seconds,
        "total_policy_steps_requested": args.total_policy_steps,
        "model_num_timesteps": model.num_timesteps,
        "control_decisions_advanced_this_run": tick_counter.control_decisions,
        "logic_ticks_advanced_this_run": tick_counter.logic_ticks,
        "logic_ticks_per_wall_second": tick_counter.logic_ticks / wall_seconds,
        "policy_steps_per_wall_second": (tick_counter.control_decisions / wall_seconds),
        "model_path": str((args.output_dir / "final_model.zip").resolve()),
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

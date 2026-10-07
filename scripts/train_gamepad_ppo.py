"""Pure-RL PPO/SIL training. Save artifacts on the GPU node only."""

import argparse
from pathlib import Path

from resnake_gym.training.ppo_run import (
    PPORunOptions,
    build_ppo_config,
    run_ppo_training,
)
from resnake_gym.training.ppo_run import (
    lightweight_policy_checkpoint as _lightweight_policy_checkpoint,
)


def lightweight_policy_checkpoint(payload):
    """Compatibility export for diagnostics that load this script with runpy."""
    return _lightweight_policy_checkpoint(payload)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sil-updates", type=int, default=2)
    parser.add_argument("--sil-batch-size", type=int, default=1)
    parser.add_argument("--sil-priority-alpha", type=float, default=0.0)
    parser.add_argument("--replay-capacity", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--spatial-pool", action="store_true")
    parser.add_argument("--chunk-rho", type=float, default=None)
    parser.add_argument(
        "--action-head",
        choices=["raw", "dpad5"],
        default="raw",
        help="raw six-factor causal marginal or Cat(5) legal D-pad reports",
    )
    parser.add_argument("--max-logic-steps", type=int, default=200000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--initial-length", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=512,
        help="per-environment decision budget, not game ticks",
    )
    parser.add_argument(
        "--collection-mode", choices=["fixed", "events"], default="events"
    )
    parser.add_argument("--minimum-steps", type=int, default=128)
    parser.add_argument(
        "--event-target",
        type=int,
        default=4,
        help="global food-event quota; deaths and truncations never satisfy it",
    )
    parser.add_argument("--recurrent-burn-in", type=int, default=64)
    parser.add_argument("--recurrent-unroll", type=int, default=128)
    parser.add_argument("--replay-success-fraction", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--shaping-scale", type=float, default=0.25)
    parser.add_argument("--death-cost", type=float, default=3.0)
    parser.add_argument("--decoder", choices=["parallel", "gru"], default="parallel")
    parser.add_argument("--solver-ms", type=float, default=50)
    args = parser.parse_args()
    if args.updates < 1 or args.threads < 1:
        parser.error("updates and threads must be positive")
    if args.recurrent_burn_in < 0 or args.recurrent_unroll < 1:
        parser.error("invalid recurrent window lengths")
    if not 0.5 <= args.replay_success_fraction < 1:
        parser.error("replay success fraction must be in [0.5, 1)")
    return args


def main():
    args = _parse_args()
    config = build_ppo_config(vars(args))
    options = PPORunOptions(
        output=args.output,
        resume=args.resume,
        updates=args.updates,
        save_every=args.save_every,
        device=args.device,
        threads=args.threads,
        source_root=Path(__file__).resolve().parents[1],
    )
    run_ppo_training(config, options)


if __name__ == "__main__":
    main()

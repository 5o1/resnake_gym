"""Train autonomous recurrent dpad5 control with IMPALA V-trace actors."""

from __future__ import annotations

import argparse
from pathlib import Path

from resnake_gym.training.vtrace_checkpoint import (
    assert_credit_accounting,
    credit_accounting_errors,
    save_checkpoint,
)
from resnake_gym.training.vtrace_metrics import (
    METRICS_STATE_FORMAT,
    episode_metrics,
)
from resnake_gym.training.vtrace_run import (
    VTraceRunOptions,
    build_vtrace_config,
    run_vtrace_training,
)

# Private compatibility aliases for existing diagnostic imports. New code uses
# the package modules above rather than loading this script as a library.
_METRICS_STATE_FORMAT = METRICS_STATE_FORMAT
_episode_metrics = episode_metrics
_credit_accounting_errors = credit_accounting_errors
_assert_credit_accounting = assert_credit_accounting
_save_checkpoint = save_checkpoint


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument(
        "--max-fresh-logic-ticks",
        type=int,
        help=(
            "stop after this cumulative number of fresh environment ticks; "
            "the final collection batch may overshoot"
        ),
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--initial-length", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--rho-bar", type=float, default=1.0)
    parser.add_argument("--c-bar", type=float, default=1.0)
    parser.add_argument("--pg-rho-bar", type=float, default=1.0)
    parser.add_argument("--max-logic-steps", type=int, default=200000)
    parser.add_argument("--actor-processes", type=int, default=8)
    parser.add_argument("--envs-per-actor", type=int, default=2)
    parser.add_argument("--actor-sync-steps", type=int, default=128)
    parser.add_argument("--unroll-length", type=int, default=128)
    parser.add_argument("--recurrent-burn-in", type=int, default=64)
    parser.add_argument("--bptt-window", type=int, default=128)
    parser.add_argument("--credit-trace-max-transitions", type=int, default=2048)
    parser.add_argument("--batch-min-transitions", type=int, default=2048)
    parser.add_argument("--batch-max-transitions", type=int, default=8192)
    parser.add_argument("--batch-food-target", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=16)
    parser.add_argument("--shaping-scale", type=float, default=0.25)
    parser.add_argument("--death-cost", type=float, default=3.0)
    parser.add_argument("--solver-ms", type=float, default=50.0)
    parser.add_argument("--decoder", choices=["parallel", "gru"], default="parallel")
    parser.add_argument(
        "--action-head",
        choices=["dpad5", "held_dpad5"],
        default="dpad5",
        help=(
            "dpad5 samples every chunk slot independently; held_dpad5 samples "
            "one category and repeats its full report across the chunk"
        ),
    )
    parser.add_argument("--no-spatial-pool", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or args.save_every < 1 or args.threads < 1:
        parser.error("updates, save-every and threads must be positive")
    if args.max_fresh_logic_ticks is not None and args.max_fresh_logic_ticks < 1:
        parser.error("max-fresh-logic-ticks must be positive")
    return args


def main():
    args = _parse_args()
    config = build_vtrace_config(vars(args))
    options = VTraceRunOptions(
        output=args.output,
        resume=args.resume,
        updates=args.updates,
        save_every=args.save_every,
        device=args.device,
        threads=args.threads,
        max_fresh_logic_ticks=args.max_fresh_logic_ticks,
        source_root=Path(__file__).resolve().parents[1],
    )
    run_vtrace_training(config, options)


if __name__ == "__main__":
    main()

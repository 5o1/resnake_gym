"""Wall-clock runner for a plugin policy or an explicitly named timing fixture."""

import argparse
import json
from pathlib import Path

from resnake_gym.realtime_runner import RealtimeRunOptions, run_realtime_policy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--policy",
        help="module:factory returning the documented reset/act policy adapter",
    )
    source.add_argument("--fixture-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--logic-fps", type=float, default=10.0)
    parser.add_argument("--inference-delay-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=520000)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="default: policy chunk length, or 8 for an adapter without metadata",
    )
    parser.add_argument("--history-limit", type=int, default=128)
    parser.add_argument(
        "--timing-v2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: enabled when required by the policy adapter",
    )
    parser.add_argument(
        "--solver-time-ms",
        type=float,
        default=None,
        help="default: use the announcement lookahead window",
    )
    args = parser.parse_args()
    if (args.horizon is not None and args.horizon < 1) or args.inference_delay_ms < 0:
        parser.error("invalid horizon or delay")
    return args


def main():
    args = parse_args()
    report = run_realtime_policy(
        RealtimeRunOptions(
            output=args.output,
            policy_spec=args.policy,
            ticks=args.ticks,
            logic_fps=args.logic_fps,
            inference_delay_ms=args.inference_delay_ms,
            seed=args.seed,
            horizon=args.horizon,
            history_limit=args.history_limit,
            timing_v2=args.timing_v2,
            solver_time_ms=args.solver_time_ms,
        )
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()

"""Evaluate a PPO/SIL or V-trace gamepad policy checkpoint."""

import argparse
import json
from pathlib import Path

from resnake_gym.evaluation_run import PolicyEvaluationOptions, run_policy_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["31x20"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=520000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    output = run_policy_evaluation(
        PolicyEvaluationOptions(
            checkpoint=args.checkpoint,
            output=args.output,
            sizes=tuple(args.sizes),
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            stochastic=args.stochastic,
        )
    )
    print(
        json.dumps(
            {
                "episodes": len(output["episodes"]),
                "wins": sum(row["won"] for row in output["episodes"]),
                "mean_score": sum(row["score"] for row in output["episodes"])
                / len(output["episodes"]),
            }
        )
    )


if __name__ == "__main__":
    main()

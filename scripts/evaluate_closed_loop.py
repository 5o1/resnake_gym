"""Task-123 evaluation entry point; no training or robot/VLA implementation."""

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from resnake_gym.closed_loop_eval import EvaluationConfig, evaluate_episode, summarize


class NeutralFixture:
    """Only for testing the evaluation tool, not a learned baseline."""

    def __init__(self, horizon):
        self.horizon = horizon

    def reset(self, *, seed):
        pass

    def act(self, observation):
        return np.zeros((self.horizon, 20), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--policy", help="module:factory; factory() returns reset/act policy"
    )
    source.add_argument("--fixture-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=520000)
    parser.add_argument(
        "--donor-seeds", type=int, nargs="+", default=[510000, 510001, 510002]
    )
    parser.add_argument("--training-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--max-ticks", type=int, default=3000)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--logic-fps", type=float, default=10.0)
    parser.add_argument("--solver-time-ms", type=float, default=50.0)
    parser.add_argument("--timing-v2", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    evaluation_seeds = list(
        range(args.eval_seed_start, args.eval_seed_start + args.episodes)
    )
    if set(evaluation_seeds) & (set(args.training_seeds) | set(args.donor_seeds)):
        parser.error(
            "evaluation seeds must not overlap declared training or donor seeds"
        )
    config = EvaluationConfig(
        max_logic_steps=args.max_ticks,
        logic_fps=args.logic_fps,
        perturbation_interval=(args.interval, args.interval),
        solver_time_limit_ms=args.solver_time_ms,
        time_features=args.timing_v2,
    )
    if args.fixture_smoke:
        policy = NeutralFixture(config.chunk_length)
    else:
        module, factory = args.policy.split(":", 1)
        policy = getattr(importlib.import_module(module), factory)()
    donors = [
        evaluate_episode(policy, config, seed=seed, condition="none")
        for seed in args.donor_seeds
    ]
    donor = max(donors, key=lambda row: (row["score"], row["ticks"]))
    records = []
    for condition in ("none", "obstacle", "rotation", "combined"):
        for seed in evaluation_seeds:
            for mode in ("closed", "frozen", "replay"):
                row = evaluate_episode(
                    policy,
                    config,
                    seed=seed,
                    condition=condition,
                    mode=mode,
                    replay=donor["actual_reports"],
                )
                records.append(row)
    summary = summarize(records)
    if args.fixture_smoke or not args.training_seeds or donor["score"] == 0:
        summary["evidence_gate"] = "insufficient"
        summary["reasons"].extend(
            reason
            for present, reason in (
                (args.fixture_smoke, "fixture_not_learned_policy"),
                (not args.training_seeds, "training_seed_split_not_declared"),
                (donor["score"] == 0, "weak_zero_score_replay_donor"),
            )
            if present
        )
    for row in records:
        del row["actual_reports"]
    result = {
        "config": asdict(config),
        "policy_source": args.policy or "neutral_fixture",
        "training_seeds_declared": args.training_seeds,
        "replay_donor_seed": donor["seed"],
        "replay_donor_score": donor["score"],
        "replay_donor_sha256": donor["action_sha256"],
        "donors": [{k: row[k] for k in ("seed", "score", "ticks")} for row in donors],
        "records": records,
        "summary": summary,
        "limitations": [
            "safety rejection can change applied disturbances between policies",
            "paired seeds share requested conditions, not realized trajectories",
            "no wall-clock claim from this event simulation",
        ],
    }
    with args.output.open("x") as target:
        json.dump(result, target, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

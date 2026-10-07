"""31x20 generator/verifier diagnostics, NOT an RL capability benchmark."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv
from resnake_gym.solvability import sample_obstacles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--time-limit-ms", type=float, default=50)
    parser.add_argument("--cycle-nodes", type=int, default=2000)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("cases must be positive")
    rng = np.random.default_rng(args.seed)
    rows = []
    # Label fixtures separately; never mix these selected witnesses into rates.
    fixtures = [
        (((1, 1), (4, 3)), "separated_fixture"),
        (((2, 2), (3, 2), (2, 3), (3, 3)), "interior_patch_fixture"),
    ]
    for i in range(args.cases + len(fixtures)):
        env = PerturbedGamepadEnv(
            perturbation_interval=None,
            notice_ticks=1,
            solver_budget={
                "time_limit_ms": args.time_limit_ms,
                "cycle_nodes": args.cycle_nodes,
            },
        )
        env.reset(seed=1, options={"food": (1, 0)})
        cells, family = (
            fixtures[i]
            if i < len(fixtures)
            else sample_obstacles(31, 20, set(env.state.snake) | {(1, 0)}, rng)
        )
        row = {"family": family, "cells": cells, "fixture": i < len(fixtures)}
        if cells:
            env.request_obstacles(cells)
            for _ in range(2):
                env.set_committed_reports(np.zeros((3, 20), np.float32))
                _, _, _, _, info = env.step(np.zeros(20, np.float32))
            event = info["perturbation_events"][0]
            row.update({key: event[key] for key in ("status", "reason", "validation")})
        else:
            row.update(status="rejected", reason="proposal_geometry", validation=None)
        rows.append(row)
        env.close()
    random_rows = [row for row in rows if not row["fixture"]]

    def outcome(row):
        return row["validation"]["status"] if row["validation"] else "not_validated"

    result = {
        "capability_result": False,
        "width": 31,
        "height": 20,
        "seed": args.seed,
        "time_limit_ms": args.time_limit_ms,
        "cycle_nodes": args.cycle_nodes,
        "random_cases": args.cases,
        "random_verdicts": dict(Counter(map(outcome, random_rows))),
        "by_family": {
            family: dict(
                Counter(outcome(row) for row in random_rows if row["family"] == family)
            )
            for family in ("scatter", "line", "patch")
        },
        "rows": rows,
        "note": "Static initial body + committed prefix; not a policy rollout. "
        "Fixtures excluded from rates. Wall budget affects repeatability.",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()

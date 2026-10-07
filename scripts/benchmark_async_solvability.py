"""Paced full-size certificate timing fixtures; no trained-policy claims."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv


def run(mode, cells, fps, notice, ticks):
    env = PerturbedGamepadEnv(
        solver_mode=mode,
        perturbation_interval=None,
        notice_ticks=notice,
        logic_fps=fps,
        solver_budget={"time_limit_ms": notice / fps * 1000},
    )
    timeline, events = [], []
    try:
        env.reset(seed=1, options={"food": (10, 0)})
        epoch = time.monotonic_ns()
        env.request_obstacles(cells)
        for tick in range(ticks):
            deadline = epoch + round((tick + 1) / fps * 1e9)
            time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9))
            env.set_committed_reports(np.zeros((3, 20), np.float32))
            start = time.monotonic_ns()
            _, _, terminated, truncated, info = env.step(np.zeros(20, np.float32))
            end = time.monotonic_ns()
            timeline.append(
                {
                    "tick": tick,
                    "scheduled_ns": deadline,
                    "started_ns": start,
                    "completed_ns": end,
                }
            )
            events.extend(info["perturbation_events"])
            if terminated or truncated:
                break
        activation = next((e for e in events if e["status"] != "announced"), None)
        validation = (activation or {}).get("validation") or {}
        began = validation.get("solver_started_ns", 0)
        finished = validation.get("solver_completed_ns", 0)
        return {
            "mode": mode,
            "cells": cells,
            "activation": activation,
            "ticks_during_solver": sum(
                began < row["started_ns"] < finished for row in timeline
            ),
            "max_step_ms": max(
                row["completed_ns"] - row["started_ns"] for row in timeline
            )
            / 1e6,
            "max_start_lateness_ms": max(
                row["started_ns"] - row["scheduled_ns"] for row in timeline
            )
            / 1e6,
            "timeline": timeline,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--notice", type=int, default=6)
    args = parser.parse_args()
    rows = [
        run(mode, cells, args.fps, args.notice, args.notice + 3)
        for cells in (((1, 1), (4, 3)), ((2, 2), (3, 2), (2, 3), (3, 3)))
        for mode in ("sync", "async")
    ]
    result = {
        "width": 31,
        "height": 20,
        "logic_fps": args.fps,
        "notice_ticks": args.notice,
        "capability_result": False,
        "note": "Selected fixtures with neutral controls; not acceptance rates.",
        "rows": rows,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "timeline"}
                for row in rows
            ]
        )
    )


if __name__ == "__main__":
    main()

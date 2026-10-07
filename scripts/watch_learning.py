"""Evaluate saved snapshots while training runs; never declares task completion.

Validation seeds are reused for diagnosis, not a final held-out acceptance set.
Run on the GPU node, with logs and evaluation artifacts beside the training run.
"""

import argparse
from pathlib import Path

from resnake_gym.analysis.learning_watch import (
    LearningWatchOptions,
    training_alive,
    watch_learning,
)
from resnake_gym.analysis.learning_watch import valid_evaluation as _valid_evaluation
from resnake_gym.artifact_io import sha256_file


def valid_evaluation(path, policy, episodes, seed, stochastic):
    """Compatibility adapter retaining the script's patchable hash helper."""

    return _valid_evaluation(
        path,
        policy,
        episodes,
        seed,
        stochastic,
        sha256_fn=sha256_file,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=610000)
    args = parser.parse_args()
    if min(args.every, args.episodes, args.pid) < 1:
        parser.error("counts and pid must be positive")
    return args


def main():
    args = parse_args()
    watch_learning(
        LearningWatchOptions(
            run=args.run,
            evaluator=Path(__file__).with_name("evaluate_gamepad_ppo.py"),
            pid=args.pid,
            every=args.every,
            episodes=args.episodes,
            device=args.device,
            seed=args.seed,
        ),
        training_alive_fn=training_alive,
        valid_evaluation_fn=valid_evaluation,
    )


__all__ = [
    "main",
    "parse_args",
    "sha256_file",
    "training_alive",
    "valid_evaluation",
]


if __name__ == "__main__":
    main()

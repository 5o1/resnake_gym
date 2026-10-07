"""Backfill and follow scalar logs without importing or interrupting training.

Uses W&B custom step axes for delayed evaluation. Source event IDs permit
deduplication against server history after a restart. Only metrics and explicit
experiment configuration are uploaded, never checkpoints or replay files.
"""

import argparse
import time
from pathlib import Path

from resnake_gym.analysis.wandb_events import (
    episode_scalars,
    events,
    percentile,
    read_json,
)
from resnake_gym.analysis.wandb_sync import (
    WandbSyncOptions,
    alive,
    evaluation_alive,
    sync_wandb,
)
from resnake_gym.analysis.wandb_sync import wait_for_metadata as _wait_for_metadata

__all__ = [
    "alive",
    "episode_scalars",
    "evaluation_alive",
    "events",
    "main",
    "percentile",
    "read_json",
    "wait_for_metadata",
]


def wait_for_metadata(directory, training_pid, *, poll_seconds=0.25):
    """Compatibility adapter for callers that patch the legacy script symbols."""

    return _wait_for_metadata(
        directory,
        training_pid,
        poll_seconds=poll_seconds,
        read_json_fn=read_json,
        alive_fn=alive,
        sleep_fn=time.sleep,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--evaluation-pid", type=int)
    parser.add_argument("--proxy", default="http://127.0.0.1:17891")
    args = parser.parse_args()
    sync_wandb(
        WandbSyncOptions(
            directory=args.directory,
            entity=args.entity,
            project=args.project,
            training_pid=args.training_pid,
            evaluation_pid=args.evaluation_pid,
            proxy=args.proxy,
        )
    )


if __name__ == "__main__":
    main()

"""Evaluate immutable policy snapshots while their trainer is running."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resnake_gym.artifact_io import sha256_file
from resnake_gym.evaluation_protocol import (
    frozen_gamepad_evaluation_request,
    protocol_sha256,
)
from resnake_gym.process_status import process_command_matches


@dataclass(frozen=True, slots=True)
class LearningWatchOptions:
    """Run identity and frozen evaluation settings for one watcher."""

    run: Path
    evaluator: Path
    pid: int
    every: int = 50
    episodes: int = 30
    device: str = "cuda:3"
    seed: int = 610000


def training_alive(pid: int | None, run: Path) -> bool:
    """Return whether ``pid`` is a supported trainer for this exact run."""

    return process_command_matches(
        pid,
        (b"train_gamepad_ppo.py", b"train_gamepad_vtrace.py"),
        run,
    )


def valid_evaluation(
    path: Path,
    policy: Path,
    episodes: int,
    seed: int,
    stochastic: bool,
    *,
    sha256_fn: Callable[[Path], str] | None = None,
) -> bool:
    """Validate every identity field used to reuse a frozen evaluation."""

    hash_file = sha256_file if sha256_fn is None else sha256_fn
    try:
        value = json.loads(path.read_text())
        policy_sha256 = hash_file(policy)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    expected_request = frozen_gamepad_evaluation_request(
        sizes=["31x20"],
        episodes_per_size=episodes,
        base_seed=seed,
        stochastic=stochastic,
    )
    actual_protocol = value.get("evaluation_protocol")
    return (
        isinstance(value.get("episodes"), list)
        and len(value["episodes"]) == episodes
        and value.get("checkpoint") == str(policy)
        and value.get("checkpoint_sha256") == policy_sha256
        and value.get("evaluation_request") == expected_request
        and isinstance(actual_protocol, dict)
        and value.get("evaluation_protocol_sha256") == protocol_sha256(actual_protocol)
        and value.get("deterministic_policy") is (not stochastic)
    )


def _evaluation_command(
    options: LearningWatchOptions,
    policy: Path,
    output: Path,
    *,
    stochastic: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(options.evaluator),
        str(policy),
        "--output",
        str(output),
        "--sizes",
        "31x20",
        "--episodes",
        str(options.episodes),
        "--seed",
        str(options.seed),
        "--device",
        options.device,
    ]
    if stochastic:
        command.append("--stochastic")
    return command


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def _evaluate_variant(
    options: LearningWatchOptions,
    policy: Path,
    update: int,
    *,
    stochastic: bool,
    environment: dict[str, str],
    validate: Callable[[Path, Path, int, int, bool], bool],
    now: Callable[[], float],
    sleep: Callable[[float], None],
    run_command: Callable[..., Any],
    emit: Callable[[dict[str, Any]], None],
) -> None:
    suffix = "-stochastic" if stochastic else ""
    output = options.run / f"validation-{update:06d}{suffix}.json"
    if output.exists() and validate(
        output,
        policy,
        options.episodes,
        options.seed,
        stochastic,
    ):
        return
    if output.exists():
        output.unlink()
    # Policy files are written after the full checkpoint. Avoid a newly
    # created file whose write might still be in progress.
    if now() - policy.stat().st_mtime < 10:
        sleep(10)
    run_command(
        _evaluation_command(
            options,
            policy,
            output,
            stochastic=stochastic,
        ),
        check=True,
        env=environment,
    )
    result = json.loads(output.read_text())
    wins = sum(episode["won"] for episode in result["episodes"])
    emit(
        {
            "update": update,
            "stochastic": stochastic,
            "wins": wins,
            "status": "candidate_win_requires_audit" if wins else "not_learned",
            "evaluation": str(output),
        }
    )


def watch_learning(
    options: LearningWatchOptions,
    *,
    training_alive_fn: Callable[[int | None, Path], bool] | None = None,
    valid_evaluation_fn: Callable[[Path, Path, int, int, bool], bool] | None = None,
    now_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    run_command_fn: Callable[..., Any] | None = None,
    emit_fn: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Follow checkpoints until the trainer exits, preserving legacy order."""

    process_alive = training_alive if training_alive_fn is None else training_alive_fn
    validate = valid_evaluation if valid_evaluation_fn is None else valid_evaluation_fn
    now = time.time if now_fn is None else now_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    run_command = subprocess.run if run_command_fn is None else run_command_fn
    emit = _emit if emit_fn is None else emit_fn
    run = options.run.resolve()
    options = LearningWatchOptions(
        run=run,
        evaluator=options.evaluator,
        pid=options.pid,
        every=options.every,
        episodes=options.episodes,
        device=options.device,
        seed=options.seed,
    )
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    while True:
        alive = process_alive(options.pid, run)
        policies = sorted(run.glob("policy-*.pt"))
        for policy in policies:
            update = int(policy.stem.split("-")[-1])
            if update % options.every and (alive or policy != policies[-1]):
                continue
            for stochastic in (False, True):
                _evaluate_variant(
                    options,
                    policy,
                    update,
                    stochastic=stochastic,
                    environment=environment,
                    validate=validate,
                    now=now,
                    sleep=sleep,
                    run_command=run_command,
                    emit=emit,
                )
        if not alive:
            emit({"status": "training_process_exited", "run": str(run)})
            return
        sleep(30)


__all__ = [
    "LearningWatchOptions",
    "training_alive",
    "valid_evaluation",
    "watch_learning",
]

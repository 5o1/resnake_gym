"""Backfill and follow persisted scalar events into a W&B run."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from resnake_gym.process_status import process_command_matches

from .wandb_events import events, read_json

_PROXY_VARIABLES = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)
_SEMANTIC_KEYS = (
    "format",
    "checkpoint_format",
    "experiment",
    "training_objective",
    "collection_semantics",
    "recurrent_state",
    "recurrent_update",
    "replay_objective",
    "action_encoding",
    "reward_version",
    "scene_encoding",
    "time_encoding",
    "metric_semantics",
)


class _WandbModule(Protocol):
    """Minimal SDK surface used by the synchronizer."""

    Settings: Any
    Api: Any

    def init(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class WandbSyncOptions:
    """Inputs for one backfill/follow process."""

    directory: Path
    entity: str
    project: str
    training_pid: int | None = None
    proxy: str = "http://127.0.0.1:17891"


def alive(pid: int | None, directory: Path) -> bool:
    """Return whether ``pid`` is a trainer for this exact run directory."""

    return process_command_matches(
        pid,
        (b"train_gamepad_ppo.py", b"train_gamepad_vtrace.py"),
        directory,
    )


def wait_for_metadata(
    directory: Path,
    training_pid: int | None,
    *,
    poll_seconds: float = 0.25,
    read_json_fn: Callable[[Path], dict[str, Any] | None] | None = None,
    alive_fn: Callable[[int | None, Path], bool] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Wait through the trainer/config publication race, then fail explicitly."""

    reader = read_json if read_json_fn is None else read_json_fn
    process_alive = alive if alive_fn is None else alive_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    path = directory / "config.json"
    while True:
        metadata = reader(path)
        if isinstance(metadata, dict) and isinstance(metadata.get("config"), dict):
            return metadata
        if training_pid and process_alive(training_pid, directory):
            sleep(poll_seconds)
            continue
        if training_pid:
            raise RuntimeError(
                "training process exited before publishing a valid config.json"
            )
        raise RuntimeError(f"no valid training config found at {path}")


def _configure_proxy(proxy: str) -> None:
    # Never inherit a different system proxy. Authentication remains in netrc.
    for key in _PROXY_VARIABLES:
        os.environ.pop(key, None)
    os.environ.update(
        HTTP_PROXY=proxy,
        HTTPS_PROXY=proxy,
        NO_PROXY="localhost,127.0.0.1",
        WANDB_SILENT="true",
    )


def _run_identifier(directory: Path) -> str:
    return "snake-" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]


def _wandb_config(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "training": metadata["config"],
        "checkpoint_semantics": {
            key: metadata[key] for key in _SEMANTIC_KEYS if key in metadata
        },
        "source_sha256": metadata.get("source_sha256", {}),
        "metric_origin": "backfilled_and_followed_JSONL",
        "sync_timestamp_is_not_collection_timestamp": True,
        "wall_clock_realtime_test": False,
        "demonstrations": False,
    }


def _initialize_run(
    options: WandbSyncOptions, metadata: dict[str, Any], wandb: _WandbModule
) -> tuple[Any, str]:
    identifier = _run_identifier(options.directory)
    run = wandb.init(
        entity=options.entity,
        project=options.project,
        id=identifier,
        resume="allow",
        name=options.directory.name + "-" + options.directory.parent.name,
        dir=str(options.directory),
        mode="online",
        job_type="metrics-sync",
        config=_wandb_config(metadata),
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            console="off",
            x_disable_stats=True,
            x_disable_meta=True,
            x_save_requirements=False,
            disable_job_creation=True,
        ),
    )
    return run, identifier


def _remote_event_ids(
    run: Any, wandb: _WandbModule, options: WandbSyncOptions, identifier: str
) -> set[str]:
    if not run.resumed:
        return set()
    remote = wandb.Api().run(f"{options.entity}/{options.project}/{identifier}")
    return {row["source_event"] for row in remote.scan_history(keys=["source_event"])}


def _define_group_metrics(run: Any, payload: dict[str, Any], defined: set[str]) -> None:
    group = next(iter(payload)).split("/")[0]
    if group in defined:
        return
    received_step = f"{group}/received_logic_ticks"
    step_metric = received_step if received_step in payload else f"{group}/update"
    run.define_metric(step_metric)
    run.define_metric(f"{group}/*", step_metric=step_metric)
    defined.add(group)


def _upload_new_events(
    directory: Path, run: Any, seen: set[str], defined: set[str]
) -> int:
    uploaded = 0
    for _, event_id, payload in events(directory):
        if event_id in seen:
            continue
        _define_group_metrics(run, payload, defined)
        run.log({**payload, "source_event": event_id})
        seen.add(event_id)
        uploaded += 1
    return uploaded


def _should_stop(
    options: WandbSyncOptions,
    *,
    trainer_running: bool,
    uploaded: int,
    ended_polls: int,
) -> tuple[bool, int]:
    if not options.training_pid:
        return True, ended_polls
    all_finished = not trainer_running
    ended_polls = ended_polls + 1 if all_finished and uploaded == 0 else 0
    return ended_polls >= 6, ended_polls


def sync_wandb(
    options: WandbSyncOptions, *, wandb_module: _WandbModule | None = None
) -> None:
    """Backfill current scalar files, then follow them until producers exit."""

    directory = options.directory.resolve()
    options = WandbSyncOptions(
        directory=directory,
        entity=options.entity,
        project=options.project,
        training_pid=options.training_pid,
        proxy=options.proxy,
    )
    _configure_proxy(options.proxy)
    wandb = importlib.import_module("wandb") if wandb_module is None else wandb_module
    metadata = wait_for_metadata(directory, options.training_pid)
    run, identifier = _initialize_run(options, metadata, wandb)
    seen = _remote_event_ids(run, wandb, options, identifier)
    defined: set[str] = set()
    print(json.dumps({"url": run.url, "resumed_events": len(seen)}), flush=True)
    ended_polls = 0
    try:
        while True:
            uploaded = _upload_new_events(directory, run, seen, defined)
            run.summary["synced_source_events"] = len(seen)
            trainer_running = alive(options.training_pid, directory)
            run.summary["training_process_alive"] = trainer_running
            print(
                json.dumps({"new_events": uploaded, "total_events": len(seen)}),
                flush=True,
            )
            stop, ended_polls = _should_stop(
                options,
                trainer_running=trainer_running,
                uploaded=uploaded,
                ended_polls=ended_polls,
            )
            if stop:
                break
            time.sleep(30)
    finally:
        run.finish()

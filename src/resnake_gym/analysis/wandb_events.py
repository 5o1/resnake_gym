"""Convert persisted training and evaluation artifacts into W&B events."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    """Read one complete JSON object, tolerating an in-progress writer."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def percentile(values: list[int | float], q: int | float) -> float:
    """Return the linearly interpolated percentile used by existing reports."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a percentile of no values")
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def episode_scalars(
    episodes: list[dict[str, Any]], prefix: str
) -> dict[str, int | float]:
    """Convert a raw episode population without changing its accounting axis."""

    scores = [int(episode["score"]) for episode in episodes]
    lifetimes = [
        int(episode.get("ticks", episode.get("controller_tick", 0)))
        for episode in episodes
    ]
    result: dict[str, int | float] = {
        f"{prefix}/episode_count": len(episodes),
        f"{prefix}/mean_score": sum(scores) / len(scores),
        f"{prefix}/max_score": max(scores),
        f"{prefix}/score_p50": percentile(scores, 50),
        f"{prefix}/score_p90": percentile(scores, 90),
        f"{prefix}/score_p99": percentile(scores, 99),
        f"{prefix}/win_rate": sum(bool(e["won"]) for e in episodes) / len(episodes),
        f"{prefix}/mean_lifetime_ticks": sum(lifetimes) / len(lifetimes),
        f"{prefix}/lifetime_ticks_p50": percentile(lifetimes, 50),
        f"{prefix}/lifetime_ticks_p90": percentile(lifetimes, 90),
        f"{prefix}/lifetime_ticks_p99": percentile(lifetimes, 99),
    }
    reasons = sorted(
        {
            str(episode.get("termination", episode.get("termination_reason")))
            for episode in episodes
            if episode.get("termination", episode.get("termination_reason")) is not None
        }
    )
    death_reasons = {"wall_collision", "self_collision", "obstacle_collision"}
    deaths = 0
    for reason in reasons:
        count = sum(
            episode.get("termination", episode.get("termination_reason")) == reason
            for episode in episodes
        )
        result[f"{prefix}/termination_{reason}_count"] = count
        result[f"{prefix}/termination_{reason}_rate"] = count / len(episodes)
        if reason in death_reasons:
            deaths += count
    result[f"{prefix}/death_count"] = deaths
    result[f"{prefix}/death_rate"] = deaths / len(episodes)
    return result


def _training_events(directory: Path) -> list[tuple[int, str, dict[str, Any]]]:
    try:
        metric_lines = (directory / "metrics.jsonl").read_text().splitlines()
    except OSError:
        metric_lines = []
    result = []
    for line in metric_lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        update = row.get("update", row.get("training_iteration"))
        if update is None:
            continue
        payload = {
            f"train/{key}": value
            for key, value in row.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        payload["train/update"] = update
        received_axis = row.get(
            "cumulative_received_logic_ticks",
            row.get("cumulative_fresh_logic_ticks"),
        )
        if isinstance(received_axis, (int, float)) and math.isfinite(received_axis):
            payload["train/received_logic_ticks"] = received_axis
        episodes = row.get("episodes", [])
        if not isinstance(episodes, list) or not all(
            isinstance(episode, dict) for episode in episodes
        ):
            episodes = []
        if episodes:
            payload.update(episode_scalars(episodes, "train"))
        result.append((update, f"train:{update}", payload))
    return result


def _evaluation_update(data: dict[str, Any]) -> int | None:
    update = data.get("source_training_iteration")
    if update is not None:
        return int(update)
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, str):
        return None
    match = re.fullmatch(r"policy-(\d+)", Path(checkpoint).stem)
    if match is None:
        # Old evaluations of a mutable full checkpoint do not contain enough
        # information to place them on the training axis.
        return None
    return int(match.group(1))


def _evaluation_signature(data: dict[str, Any], episodes: list[dict[str, Any]]) -> str:
    seeds = sorted(episode["seed"] for episode in episodes)
    checkpoint_sha256 = data.get("checkpoint_sha256")
    protocol_sha256 = data.get("evaluation_protocol_sha256")
    if checkpoint_sha256 is None and protocol_sha256 is None:
        identity = json.dumps(seeds).encode()
    else:
        identity = json.dumps(
            {
                "seeds": seeds,
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_protocol_sha256": protocol_sha256,
            },
            sort_keys=True,
        ).encode()
    return hashlib.sha256(identity).hexdigest()[:12]


def _evaluation_events(directory: Path) -> list[tuple[int, str, dict[str, Any]]]:
    result = []
    files = sorted(
        {*directory.glob("eval-*.json"), *directory.glob("validation-*.json")}
    )
    for path in files:
        data = read_json(path)
        if data is None:
            continue
        episodes = data.get("episodes")
        if not isinstance(episodes, list) or not all(
            isinstance(episode, dict) for episode in episodes
        ):
            continue
        update = _evaluation_update(data)
        if update is None:
            continue
        prefix = (
            "eval_sampled"
            if data.get("deterministic_policy") is False
            else "eval_deterministic"
        )
        # Keep different boards separate; do not average unlike evaluation tasks.
        for size in sorted({episode["size"] for episode in episodes}):
            group = [episode for episode in episodes if episode["size"] == size]
            key = f"{prefix}_{size}"
            signature = _evaluation_signature(data, group)
            payload: dict[str, Any] = {
                f"{key}/update": update,
                f"{key}/episodes": len(group),
            }
            received_axis = data.get("source_received_logic_ticks")
            if isinstance(received_axis, (int, float)) and math.isfinite(received_axis):
                payload[f"{key}/received_logic_ticks"] = received_axis
            payload.update(episode_scalars(group, key))
            result.append((update, f"{key}:{update}:{signature}", payload))
    return result


def events(directory: Path) -> list[tuple[int, str, dict[str, Any]]]:
    """Return stable-ID scalar events ordered by training update and ID."""

    result = [*_training_events(directory), *_evaluation_events(directory)]
    return sorted(result, key=lambda item: (item[0], item[1]))

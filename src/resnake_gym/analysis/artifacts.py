"""Reading and validation helpers for V-trace v3 JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .protocol import (
    DEATH_REASONS,
    EXPECTED_COLLECTION,
    EXPECTED_FORMAT,
    EXPECTED_METRICS,
    EXPECTED_RECURRENT,
    FORMAL_ARMS,
    FORMAL_COMMON_CONFIG,
    SCORE_THRESHOLDS,
)


class Audit:
    """Collect machine-readable checks without hiding later diagnostics."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        ok = bool(condition)
        row: dict[str, Any] = {"name": name, "ok": ok}
        if detail:
            row["detail"] = detail
        self.checks.append(row)
        if not ok:
            self.errors.append(f"{name}: {detail}" if detail else name)
        return ok

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def payload(self) -> dict[str, Any]:
        return {
            "valid": not self.errors,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return _sha256_bytes(encoded)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no metric rows")
    return rows


def _finite_issues(value: Any, path: str) -> list[str]:
    issues = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return issues
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append(path)
        return issues
    if isinstance(value, int):
        return issues
    if isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_finite_issues(item, f"{path}[{index}]"))
        return issues
    if isinstance(value, dict):
        for key, item in value.items():
            issues.extend(_finite_issues(item, f"{path}.{key}"))
    return issues


def _percentile(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _episode_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(episode["score"]) for episode in episodes]
    lifetimes = [
        int(episode.get("ticks", episode.get("logic_steps", 0))) for episode in episodes
    ]
    terminations = Counter(
        str(episode.get("termination", episode.get("termination_reason")))
        for episode in episodes
    )
    count = len(episodes)
    deaths = sum(terminations[reason] for reason in DEATH_REASONS)
    return {
        "episodes": count,
        "wins": sum(bool(episode.get("won")) for episode in episodes),
        "win_rate": sum(bool(episode.get("won")) for episode in episodes) / count
        if count
        else None,
        "score": {
            "mean": sum(scores) / count if count else None,
            "p50": _percentile(scores, 50),
            "p90": _percentile(scores, 90),
            "p99": _percentile(scores, 99),
            "max": max(scores) if scores else None,
        },
        "score_thresholds": {
            str(threshold): {
                "count": sum(score >= threshold for score in scores),
                "rate": sum(score >= threshold for score in scores) / count
                if count
                else None,
            }
            for threshold in SCORE_THRESHOLDS
        },
        "lifetime_ticks": {
            "mean": sum(lifetimes) / count if count else None,
            "p50": _percentile(lifetimes, 50),
            "p90": _percentile(lifetimes, 90),
            "p99": _percentile(lifetimes, 99),
            "max": max(lifetimes) if lifetimes else None,
        },
        "termination": {
            reason: {
                "count": reason_count,
                "rate": reason_count / count if count else None,
            }
            for reason, reason_count in sorted(terminations.items())
        },
        "death": {
            "count": deaths,
            "rate": deaths / count if count else None,
        },
    }


def _expected_action_semantics(action_head: Any) -> tuple[str, str] | None:
    if action_head == "dpad5":
        return (
            "snake-dpad5-event-trace-vtrace-v2",
            "xinput-dpad5-categorical-v1",
        )
    if action_head == "held_dpad5":
        return (
            "snake-held-dpad5-event-trace-vtrace-h1-v2",
            "xinput-held-dpad5-categorical-h1-v1",
        )
    return None


def _validate_config(
    metadata: dict[str, Any],
    audit: Audit,
    *,
    arm: str | None,
    expected_budget: int,
) -> dict[str, Any]:
    config = metadata.get("config")
    if not audit.check("config.object", isinstance(config, dict)):
        return {}
    assert isinstance(config, dict)
    audit.check("config.format", metadata.get("format") == EXPECTED_FORMAT)
    audit.check(
        "config.collection_semantics",
        metadata.get("collection_semantics") == EXPECTED_COLLECTION,
    )
    audit.check(
        "config.recurrent_state", metadata.get("recurrent_state") == EXPECTED_RECURRENT
    )
    audit.check(
        "config.metric_semantics",
        metadata.get("metric_semantics") == EXPECTED_METRICS,
    )
    semantics = _expected_action_semantics(config.get("action_head"))
    audit.check("config.action_head_known", semantics is not None)
    if semantics is not None:
        objective, encoding = semantics
        audit.check(
            "config.training_objective",
            metadata.get("training_objective") == objective,
        )
        audit.check(
            "config.action_encoding", metadata.get("action_encoding") == encoding
        )
    budget = metadata.get("training_budget")
    if audit.check("config.training_budget_object", isinstance(budget, dict)):
        assert isinstance(budget, dict)
        audit.check(
            "config.primary_budget_axis",
            budget.get("primary_axis") == "cumulative_received_logic_ticks",
        )
        audit.check(
            "config.primary_budget_value",
            budget.get("max_fresh_logic_ticks") == expected_budget,
            f"expected {expected_budget}, got {budget.get('max_fresh_logic_ticks')}",
        )
    audit.check(
        "config.fragment_replay_disabled",
        config.get("replay_batches_per_update", 0) == 0
        and metadata.get("fragment_replay_training_enabled") is False,
    )
    audit.check(
        "config.bptt_window_bounded",
        isinstance(config.get("bptt_window"), int) and 0 < config["bptt_window"] <= 128,
    )
    if arm in FORMAL_ARMS:
        expected = {**FORMAL_COMMON_CONFIG, **FORMAL_ARMS[arm]}
        for key, expected_value in expected.items():
            audit.check(
                f"formal_arm.{arm}.{key}",
                config.get(key) == expected_value,
                f"expected {expected_value!r}, got {config.get(key)!r}",
            )
        audit.check(
            f"formal_arm.{arm}.not_resumed",
            metadata.get("resume") is None and metadata.get("run_generation") == 0,
        )
    return config


def _validate_source_manifest(
    metadata: dict[str, Any],
    audit: Audit,
    *,
    source_root: Path | None,
) -> tuple[dict[str, str], str | None, dict[str, Any]]:
    manifest = metadata.get("source_sha256")
    if not audit.check("source_sha256.object", isinstance(manifest, dict)):
        return {}, None, {"checked_against_local_files": False}
    assert isinstance(manifest, dict)
    normalized = {str(path): value for path, value in manifest.items()}
    audit.check("source_sha256.nonempty", bool(normalized))
    audit.check(
        "source_sha256.values", all(_is_sha256(value) for value in normalized.values())
    )
    digest = _canonical_sha256(normalized)
    local = {
        "checked_against_local_files": source_root is not None,
        "root": str(source_root) if source_root is not None else None,
        "missing": [],
        "mismatched": [],
    }
    if source_root is not None:
        root = source_root.resolve()
        for relative, expected in sorted(normalized.items()):
            candidate = (root / relative).resolve()
            try:
                inside_root = candidate.is_relative_to(root)
            except AttributeError:  # pragma: no cover - Python 3.8/3.9 fallback
                inside_root = root == candidate or root in candidate.parents
            if not inside_root or not candidate.is_file():
                local["missing"].append(relative)
                continue
            actual = _sha256_bytes(candidate.read_bytes())
            if actual != expected:
                local["mismatched"].append(
                    {"path": relative, "expected": expected, "actual": actual}
                )
        audit.check(
            "source_sha256.local_match",
            not local["missing"] and not local["mismatched"],
            f"missing={len(local['missing'])}, mismatched={len(local['mismatched'])}",
        )
    return normalized, digest, local


def _discover_validation_paths(
    run_directory: Path,
    *,
    validation_glob: str | None,
    final_received_ticks: int | None,
) -> list[Path]:
    if validation_glob is not None:
        return sorted(run_directory.glob(validation_glob))
    for pattern in ("validation-final*.json", "validation-full*.json"):
        matches = sorted(run_directory.glob(pattern))
        if matches:
            return matches
    matches = sorted(run_directory.glob("validation-*.json"))
    if not matches:
        return []
    selected = []
    for path in matches:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("source_received_logic_ticks") == final_received_ticks:
            selected.append(path)
    return selected


def _validation_summary(
    paths: list[Path],
    audit: Audit,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    final_received_ticks: int | None,
    final_iteration: int | None,
    required: bool,
) -> dict[str, Any]:
    audit.check(
        "validation.present", bool(paths) or not required, "no final validation JSON"
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    files = []
    finite_issues = []
    identities = set()
    for path in paths:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            audit.check(f"validation.{path.name}.readable", False, str(error))
            continue
        finite_issues.extend(_finite_issues(payload, f"validation.{path.name}"))
        episodes = payload.get("episodes")
        if not audit.check(
            f"validation.{path.name}.episodes",
            isinstance(episodes, list) and bool(episodes),
        ):
            continue
        assert isinstance(episodes, list)
        audit.check(
            f"validation.{path.name}.checkpoint_sha256",
            _is_sha256(payload.get("checkpoint_sha256")),
        )
        if "evaluation_protocol_sha256" in payload:
            audit.check(
                f"validation.{path.name}.protocol_sha256",
                _is_sha256(payload.get("evaluation_protocol_sha256")),
            )
        audit.check(
            f"validation.{path.name}.source_received_ticks",
            payload.get("source_received_logic_ticks") == final_received_ticks,
        )
        audit.check(
            f"validation.{path.name}.source_iteration",
            payload.get("source_training_iteration") == final_iteration,
        )
        audit.check(
            f"validation.{path.name}.source_action_head",
            payload.get("source_action_head") == config.get("action_head"),
        )
        for source_key, metadata_key in (
            ("source_action_encoding", "action_encoding"),
            ("source_training_objective", "training_objective"),
            ("source_checkpoint_format", "format"),
            ("source_collection_semantics", "collection_semantics"),
            ("source_recurrent_state", "recurrent_state"),
        ):
            audit.check(
                f"validation.{path.name}.{source_key}",
                payload.get(source_key) == metadata.get(metadata_key),
            )
        audit.check(
            f"validation.{path.name}.source_credit_cap",
            payload.get("source_credit_trace_max_transitions")
            == config.get("credit_trace_max_transitions"),
        )
        audit.check(
            f"validation.{path.name}.source_bptt_window",
            payload.get("source_bptt_window") == config.get("bptt_window"),
        )
        mode = (
            "deterministic"
            if payload.get("deterministic_policy") is True
            else "stochastic"
        )
        for index, episode in enumerate(episodes):
            valid = (
                isinstance(episode, dict)
                and isinstance(episode.get("score"), int)
                and isinstance(episode.get("won"), bool)
                and isinstance(episode.get("seed"), int)
            )
            audit.check(f"validation.{path.name}.episodes[{index}]", valid)
            if not valid:
                continue
            size = str(
                episode.get("size", f"{config.get('width')}x{config.get('height')}")
            )
            identity = (mode, size, episode["seed"])
            audit.check(
                f"validation.{path.name}.episodes[{index}].unique",
                identity not in identities,
            )
            identities.add(identity)
            groups.setdefault((mode, size), []).append(episode)
        files.append(
            {
                "path": str(path),
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_sha256": payload.get("checkpoint_sha256"),
                "mode": mode,
                "episodes": len(episodes),
                "source_received_logic_ticks": payload.get(
                    "source_received_logic_ticks"
                ),
                "source_training_iteration": payload.get("source_training_iteration"),
            }
        )
    audit.check(
        "validation.numeric_finiteness",
        not finite_issues,
        ", ".join(finite_issues[:20]),
    )
    return {
        "files": files,
        "groups": {
            f"{mode}_{size}": _episode_summary(episodes)
            for (mode, size), episodes in sorted(groups.items())
        },
        "numeric_finiteness": {
            "all_finite": not finite_issues,
            "nonfinite_paths": finite_issues,
        },
    }

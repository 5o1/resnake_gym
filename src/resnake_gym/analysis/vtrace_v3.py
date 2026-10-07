"""Public facade for V-trace v3 artifact analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import (
    Audit,
    _canonical_sha256,
    _discover_validation_paths,
    _episode_summary,
    _expected_action_semantics,
    _finite_issues,
    _is_sha256,
    _percentile,
    _read_json,
    _read_jsonl,
    _sha256_bytes,
    _validate_config,
    _validate_source_manifest,
    _validation_summary,
)
from .protocol import (
    DEATH_REASONS,
    EXPECTED_COLLECTION,
    EXPECTED_FORMAT,
    EXPECTED_METRICS,
    EXPECTED_RECURRENT,
    FORMAL_ARMS,
    FORMAL_COMMON_CONFIG,
    FORMAL_CONTRASTS,
    SCORE_THRESHOLDS,
)
from .training import (
    _preferred_validation_mean,
    _training_summary,
    compare_runs,
)

__all__ = [
    "Audit",
    "DEATH_REASONS",
    "EXPECTED_COLLECTION",
    "EXPECTED_FORMAT",
    "EXPECTED_METRICS",
    "EXPECTED_RECURRENT",
    "FORMAL_ARMS",
    "FORMAL_COMMON_CONFIG",
    "FORMAL_CONTRASTS",
    "SCORE_THRESHOLDS",
    "_canonical_sha256",
    "_discover_validation_paths",
    "_episode_summary",
    "_expected_action_semantics",
    "_finite_issues",
    "_is_sha256",
    "_percentile",
    "_preferred_validation_mean",
    "_read_json",
    "_read_jsonl",
    "_sha256_bytes",
    "_training_summary",
    "_validate_config",
    "_validate_source_manifest",
    "_validation_summary",
    "analyze_run",
    "compare_runs",
]


def analyze_run(
    run_directory: Path,
    *,
    arm: str | None = None,
    expected_budget: int = 1_000_000,
    validation_glob: str | None = None,
    source_root: Path | None = None,
    require_validation: bool = True,
) -> dict[str, Any]:
    """Analyze one run directory and return a self-contained JSON value."""

    run_directory = run_directory.resolve()
    audit = Audit()
    try:
        config_path = run_directory / "config.json"
        metrics_path = run_directory / "metrics.jsonl"
        metadata = _read_json(config_path)
        rows = _read_jsonl(metrics_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        audit.check("artifacts.readable", False, str(error))
        return {
            "run": str(run_directory),
            "arm": arm,
            "integrity": audit.payload(),
        }
    audit.check("artifacts.readable", True)
    config = _validate_config(metadata, audit, arm=arm, expected_budget=expected_budget)
    manifest, manifest_digest, local_source = _validate_source_manifest(
        metadata, audit, source_root=source_root
    )
    training = _training_summary(rows, audit, expected_budget=expected_budget)
    validation_paths = _discover_validation_paths(
        run_directory,
        validation_glob=validation_glob,
        final_received_ticks=training.get("final_received_logic_ticks"),
    )
    validation = _validation_summary(
        validation_paths,
        audit,
        config=config,
        metadata=metadata,
        final_received_ticks=training.get("final_received_logic_ticks"),
        final_iteration=training.get("final_training_iteration"),
        required=require_validation,
    )
    config_bytes = config_path.read_bytes()
    return {
        "run": str(run_directory),
        "arm": arm,
        "config": {
            "file_sha256": _sha256_bytes(config_bytes),
            "canonical_sha256": _canonical_sha256(metadata),
            "source_manifest_sha256": manifest_digest,
            "source_file_count": len(manifest),
            "source_local_verification": local_source,
            "seed": config.get("seed"),
            "action_head": config.get("action_head"),
            "entropy_coefficient": config.get("entropy_coefficient"),
            "batch_min_transitions": config.get("batch_min_transitions"),
            "credit_trace_max_transitions": config.get("credit_trace_max_transitions"),
        },
        "training": training,
        "validation": validation,
        "integrity": audit.payload(),
    }

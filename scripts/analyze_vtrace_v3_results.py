"""Audit and summarize completed V-trace v3 training runs offline.

The analyzer deliberately reads only JSON artifacts.  It never loads a model,
starts an environment, or infers a missing food event from a batch total.  In
particular, formal food-rate windows are computed from the exact, one-based
``fresh_received_food_tick_positions`` written by the collector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from resnake_gym.analysis.protocol import (
    DEATH_REASONS,
    EXPECTED_COLLECTION,
    EXPECTED_FORMAT,
    EXPECTED_METRICS,
    EXPECTED_RECURRENT,
    FORMAL_ARMS,
    FORMAL_COMMON_CONFIG,
    FORMAL_CONTRASTS,
    REPORT_FORMAT,
    SCORE_THRESHOLDS,
)
from resnake_gym.analysis.vtrace_v3 import (
    Audit,
    _canonical_sha256,
    _discover_validation_paths,
    _episode_summary,
    _expected_action_semantics,
    _finite_issues,
    _is_sha256,
    _percentile,
    _preferred_validation_mean,
    _read_json,
    _read_jsonl,
    _sha256_bytes,
    _training_summary,
    _validate_config,
    _validate_source_manifest,
    _validation_summary,
    analyze_run,
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
    "REPORT_FORMAT",
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
    "main",
]


def _parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label or not raw_path:
            raise argparse.ArgumentTypeError("run must be NAME=PATH or PATH")
        return label, Path(raw_path)
    path = Path(value)
    return path.name, path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", metavar="[NAME=]RUN", help="one or more v3 run directories"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-budget", type=int, default=1_000_000)
    parser.add_argument(
        "--validation-glob",
        help=(
            "glob relative to each run; by default final/full files are preferred "
            "and otherwise files matching the final received-tick checkpoint are used"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository/snapshot root used to verify config source_sha256",
    )
    parser.add_argument(
        "--skip-local-source-check",
        action="store_true",
        help="validate source hash syntax and cross-run identity only",
    )
    parser.add_argument(
        "--allow-missing-validation",
        action="store_true",
        help="permit a training-only audit before final validation exists",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="return zero even when an integrity check fails",
    )
    args = parser.parse_args(argv)
    if args.expected_budget < 1:
        parser.error("--expected-budget must be positive")
    parsed = [_parse_run_spec(value) for value in args.runs]
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        parser.error("run labels must be unique")
    runs = {
        label: analyze_run(
            path,
            arm=label,
            expected_budget=args.expected_budget,
            validation_glob=args.validation_glob,
            source_root=None if args.skip_local_source_check else args.source_root,
            require_validation=not args.allow_missing_validation,
        )
        for label, path in parsed
    }
    report = {
        "format": REPORT_FORMAT,
        "window_semantics": (
            "food receipt positions are one-based; windows are (0,250000] and "
            "(750000,1000000] on cumulative received logic ticks"
        ),
        "runs": runs,
        "comparison": compare_runs(runs),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _write_json(args.output, report)
    else:
        sys.stdout.write(encoded)
    valid = report["comparison"]["all_runs_valid"]
    if len(runs) > 1 and not report["comparison"]["same_source_manifest"]:
        valid = False
    return 0 if valid or args.allow_invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())

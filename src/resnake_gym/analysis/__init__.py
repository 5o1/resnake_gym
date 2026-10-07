"""Offline analysis utilities for resnake-gym experiment artifacts."""

from .protocol import (
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
from .vtrace_v3 import Audit, analyze_run, compare_runs

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
    "analyze_run",
    "compare_runs",
]

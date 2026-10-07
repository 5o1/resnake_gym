"""Compatibility entrypoint for the legacy-v0 policy latency benchmark."""

from resnake_gym_legacy.v0.benchmark_policy_latency import (
    _action,
    _advance,
    _deadline_summary,
    _percentile,
    _positive_finite,
    _reset,
    _sha256,
    _summarize_ms,
    main,
    parse_args,
    run_paced,
    run_unpaced,
)

__all__ = [
    "_action",
    "_advance",
    "_deadline_summary",
    "_percentile",
    "_positive_finite",
    "_reset",
    "_sha256",
    "_summarize_ms",
    "main",
    "parse_args",
    "run_paced",
    "run_unpaced",
]


if __name__ == "__main__":
    main()

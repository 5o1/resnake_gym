"""Compatibility entrypoint for the legacy-v0 environment benchmark."""

from resnake_gym_legacy.v0.benchmark_env import (
    _benchmark_case,
    _benchmark_lengths,
    _nonnegative_int,
    _positive_float,
    _positive_int,
    _run_window,
    _serialize_json,
    _window_setup,
    _write_json_atomic,
    benchmark_environment,
    build_parser,
    main,
)

__all__ = [
    "_benchmark_case",
    "_benchmark_lengths",
    "_nonnegative_int",
    "_positive_float",
    "_positive_int",
    "_run_window",
    "_serialize_json",
    "_window_setup",
    "_write_json_atomic",
    "benchmark_environment",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

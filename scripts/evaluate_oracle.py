"""Compatibility entrypoint for the legacy-v0 Hamiltonian-oracle evaluator."""

from resnake_gym_legacy.v0.evaluate_oracle import (
    _initial_state,
    _nonnegative_int,
    _percentile,
    _positive_int,
    _serialize_json,
    _write_json_atomic,
    build_parser,
    evaluate_oracle,
    main,
)

__all__ = [
    "_initial_state",
    "_nonnegative_int",
    "_percentile",
    "_positive_int",
    "_serialize_json",
    "_write_json_atomic",
    "build_parser",
    "evaluate_oracle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

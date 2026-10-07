"""Compatibility entrypoint for the legacy-v0 behavior-cloning trainer."""

from resnake_gym_legacy.v0.train_bc import (
    OracleBatchGenerator,
    _accuracy,
    main,
    parse_args,
)

__all__ = ["OracleBatchGenerator", "_accuracy", "main", "parse_args"]


if __name__ == "__main__":
    main()

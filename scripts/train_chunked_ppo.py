"""Compatibility entrypoint for the historical relative-action PPO trainer."""

from resnake_gym_legacy.relative_chunked.train_ppo import (
    _discount_per_control_period,
    _make_chunked_env,
    _package_version,
    _timing_kwargs_from_args,
    _validate_args,
    main,
    parse_args,
)

__all__ = [
    "_discount_per_control_period",
    "_make_chunked_env",
    "_package_version",
    "_timing_kwargs_from_args",
    "_validate_args",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()

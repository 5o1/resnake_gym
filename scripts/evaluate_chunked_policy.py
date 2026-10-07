"""Compatibility entrypoint for the historical relative-action evaluator."""

from resnake_gym_legacy.relative_chunked.evaluate_policy import (
    _ActiveEpisode,
    _episode_result,
    _evaluate_episodes,
    _make_env,
    _package_version,
    _percentile,
    _predict_raw_chunks,
    _stack_observations,
    _validate_args,
    _validate_timing_range,
    _wilson_interval,
    main,
    parse_args,
)

__all__ = [
    "_ActiveEpisode",
    "_episode_result",
    "_evaluate_episodes",
    "_make_env",
    "_package_version",
    "_percentile",
    "_predict_raw_chunks",
    "_stack_observations",
    "_validate_args",
    "_validate_timing_range",
    "_wilson_interval",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()

"""Compatibility entrypoint for the legacy-v0 policy evaluator."""

from resnake_gym_legacy.v0.evaluate_policy import (
    _ActiveEpisode,
    _episode_result,
    _evaluate_episodes,
    _MajorityVotePolicy,
    _percentile,
    _predict_actions,
    _wilson_interval,
    main,
    parse_args,
)

__all__ = [
    "_ActiveEpisode",
    "_episode_result",
    "_evaluate_episodes",
    "_MajorityVotePolicy",
    "_percentile",
    "_predict_actions",
    "_wilson_interval",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()

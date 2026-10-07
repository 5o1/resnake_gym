"""Compatibility entrypoint for the legacy-v0 PPO trainer."""

from resnake_gym_legacy.v0.train_ppo import _finite, _positive, main, parse_args

__all__ = ["_finite", "_positive", "main", "parse_args"]


if __name__ == "__main__":
    main()

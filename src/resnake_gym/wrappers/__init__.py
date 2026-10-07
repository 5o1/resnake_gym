"""Current wrappers plus lazy compatibility names for historical experiments."""

from resnake_gym.wrappers.gamepad_async import AsyncGamepad


def __getattr__(name: str):
    """Load relative-action wrappers only when an old caller requests them."""
    if name in ("ChunkedControlWrapper", "LegacyChunkedControlWrapper"):
        from resnake_gym_legacy.relative_chunked.chunked_control import (
            ChunkedControlWrapper,
        )

        return ChunkedControlWrapper
    if name in ("DistanceRewardWrapper", "LegacyDistanceRewardWrapper"):
        from resnake_gym_legacy.v0.distance_reward import DistanceRewardWrapper

        return DistanceRewardWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncGamepad",
    "LegacyChunkedControlWrapper",
    "LegacyDistanceRewardWrapper",
    "ChunkedControlWrapper",
    "DistanceRewardWrapper",
]

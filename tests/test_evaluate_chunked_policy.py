"""Regression tests for raw chunk-policy evaluation."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "scripts/evaluate_chunked_policy.py"
)
parse_args = _SCRIPT["parse_args"]
_evaluate_episodes = _SCRIPT["_evaluate_episodes"]
_wilson_interval = _SCRIPT["_wilson_interval"]


class _StraightChunkPolicy:
    """Fake SB3 policy returning raw all-straight chunks."""

    def __init__(self, chunk_length: int) -> None:
        self.chunk_length = chunk_length
        self.batch_sizes: list[int] = []

    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        assert set(observation) == {
            "delayed_state",
            "observation_age",
            "new_observation",
            "previous_chunk",
            "executed_actions",
        }
        batch_size = observation["delayed_state"].shape[0]
        assert all(value.shape[0] == batch_size for value in observation.values())
        self.batch_sizes.append(batch_size)
        return np.zeros((batch_size, self.chunk_length), dtype=np.int64), None


def _evaluation_arguments() -> dict[str, object]:
    return {
        "width": 7,
        "height": 6,
        "episode_count": 20,
        "start_seed": 45_000,
        "max_logic_steps": 30,
        "logic_fps": 12.0,
        "control_interval_ticks": 3,
        "chunk_length": 5,
        "observation_age_range": (0, 2),
        "command_delay_range": (0, 2),
    }


def test_chunked_evaluation_defaults_are_full_board_and_explicit_timing() -> None:
    args = parse_args(["model.zip"])

    assert (args.width, args.height) == (31, 20)
    assert args.logic_fps == pytest.approx(10.0)
    assert args.control_interval_ticks == 3
    assert args.chunk_length == 6
    assert (args.observation_age_min_ticks, args.observation_age_max_ticks) == (
        0,
        2,
    )
    assert (args.command_delay_min_ticks, args.command_delay_max_ticks) == (0, 2)
    assert args.batch_envs == 256


def test_batched_chunk_evaluation_matches_batch_one_by_seed() -> None:
    arguments = _evaluation_arguments()
    scalar_policy = _StraightChunkPolicy(chunk_length=5)
    scalar, scalar_timing = _evaluate_episodes(
        scalar_policy,
        batch_envs=1,
        **arguments,
    )
    batched_policy = _StraightChunkPolicy(chunk_length=5)
    batched, batched_timing = _evaluate_episodes(
        batched_policy,
        batch_envs=6,
        **arguments,
    )

    assert batched == scalar
    assert [episode["seed"] for episode in batched] == list(range(45_000, 45_020))
    assert all(size == 1 for size in scalar_policy.batch_sizes)
    assert max(batched_policy.batch_sizes) == 6
    assert scalar_timing["policy_decisions"] == batched_timing["policy_decisions"]
    assert batched_timing["max_inference_batch_size"] == 6

    for episode in batched:
        assert episode["raw_submitted_action_count"] == (
            episode["control_decisions"] * 5
        )
        assert episode["raw_submitted_action_counts"] == {
            "straight": episode["raw_submitted_action_count"],
            "right": 0,
            "left": 0,
        }
        assert 0 <= episode["max_observation_age_ticks"] <= 2
        assert 0 <= episode["max_sampled_command_delay_ticks"] <= 2
        assert episode["logic_steps"] <= 30


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (0, 1000, (0.0, 0.0038267584855551234)),
        (997, 1000, (0.9912169859464969, 0.9989792161188614)),
        (1000, 1000, (0.9961732415144448, 1.0)),
    ],
)
def test_chunked_wilson_interval(
    successes: int, trials: int, expected: tuple[float, float]
) -> None:
    assert _wilson_interval(successes, trials) == pytest.approx(expected)


@pytest.mark.parametrize(("successes", "trials"), [(0, 0), (-1, 10), (11, 10)])
def test_chunked_wilson_interval_rejects_invalid_counts(
    successes: int, trials: int
) -> None:
    with pytest.raises(ValueError):
        _wilson_interval(successes, trials)

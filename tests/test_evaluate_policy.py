"""Regression tests for seeded, batched policy evaluation."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = runpy.run_path(Path(__file__).parents[1] / "scripts/evaluate_policy.py")
_MajorityVotePolicy = _SCRIPT["_MajorityVotePolicy"]
_evaluate_episodes = _SCRIPT["_evaluate_episodes"]
_wilson_interval = _SCRIPT["_wilson_interval"]


class _GreedyPolicy:
    """Small deterministic policy implementing the SB3 ``predict`` contract."""

    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        self.input_shapes.append(observation.shape)
        scalar = observation.ndim == 3
        observations = observation[None, ...] if scalar else observation
        actions = np.asarray(
            [self._action(item) for item in observations],
            dtype=np.int64,
        )
        return (actions[0] if scalar else actions), None

    @staticmethod
    def _action(observation: np.ndarray) -> int:
        head_y, head_x = np.argwhere(observation[2] == 1.0)[0]
        food_y, food_x = np.argwhere(observation[4] == 1.0)[0]
        direction = int(np.argmax(observation[5:, head_y, head_x]))
        if head_x != food_x:
            desired = 1 if food_x > head_x else 3
        else:
            desired = 2 if food_y > head_y else 0
        turn = (desired - direction) % 4
        if turn == 0:
            return 0
        if turn in (1, 2):
            return 1
        return 2


class _FixedPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray | np.integer, None]:
        assert deterministic
        if observation.ndim == 3:
            return np.int64(self.action), None
        return np.full(observation.shape[0], self.action, dtype=np.int64), None


def test_batched_evaluation_matches_scalar_by_seed_and_keeps_order() -> None:
    arguments = {
        "width": 6,
        "height": 6,
        "episode_count": 40,
        "start_seed": 100_000,
        "max_logic_steps": 80,
    }
    scalar_policy = _GreedyPolicy()
    scalar, _ = _evaluate_episodes(scalar_policy, batch_envs=1, **arguments)
    batched_policy = _GreedyPolicy()
    batched, _ = _evaluate_episodes(batched_policy, batch_envs=7, **arguments)

    assert batched == scalar
    assert [result["seed"] for result in batched] == list(range(100_000, 100_040))
    assert len({result["logic_steps"] for result in batched}) > 1
    assert all(len(shape) == 3 for shape in scalar_policy.input_shapes)
    assert all(len(shape) == 4 for shape in batched_policy.input_shapes)
    assert max(shape[0] for shape in batched_policy.input_shapes) == 7


def test_evaluation_applies_explicit_timebase_and_frame_skip() -> None:
    policy = _GreedyPolicy()

    episodes, _ = _evaluate_episodes(
        policy,
        width=6,
        height=6,
        episode_count=1,
        start_seed=123,
        max_logic_steps=2,
        batch_envs=1,
        logic_fps=20.0,
        frame_skip=2,
    )

    assert episodes[0]["logic_steps"] == 2
    assert len(policy.input_shapes) == 1


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (0, 1000, (0.0, 0.0038267584855551234)),
        (997, 1000, (0.9912169859464969, 0.9989792161188614)),
        (1000, 1000, (0.9961732415144448, 1.0)),
    ],
)
def test_wilson_interval_known_values(
    successes: int,
    trials: int,
    expected: tuple[float, float],
) -> None:
    assert _wilson_interval(successes, trials) == pytest.approx(expected)


@pytest.mark.parametrize(("successes", "trials"), [(0, 0), (-1, 10), (11, 10)])
def test_wilson_interval_rejects_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        _wilson_interval(successes, trials)


def test_majority_vote_supports_scalar_and_batched_observations() -> None:
    policy = _MajorityVotePolicy([_FixedPolicy(1), _FixedPolicy(2), _FixedPolicy(1)])

    scalar, _ = policy.predict(np.zeros((9, 4, 4)), deterministic=True)
    batched, _ = policy.predict(np.zeros((5, 9, 4, 4)), deterministic=True)

    assert scalar == 1
    np.testing.assert_array_equal(batched, np.ones(5, dtype=np.int64))


def test_majority_vote_uses_primary_policy_for_three_way_tie() -> None:
    policy = _MajorityVotePolicy([_FixedPolicy(2), _FixedPolicy(0), _FixedPolicy(1)])

    action, _ = policy.predict(np.zeros((9, 4, 4)), deterministic=True)

    assert action == 2


def test_majority_vote_requires_multiple_policies() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _MajorityVotePolicy([_FixedPolicy(0)])

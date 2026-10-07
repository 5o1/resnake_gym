"""Tests for the deterministic Hamiltonian-cycle baseline."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from resnake_gym.baselines import (
    HamiltonianOracle,
    absolute_action,
    cycle_aligned_snake,
    hamiltonian_cycle,
    is_cycle_aligned,
    relative_action,
)
from resnake_gym.envs import SnakeEnv
from resnake_gym.envs.snake_env import DOWN, LEFT, RIGHT, UP


def _wrapped_pairs(
    cycle: tuple[tuple[int, int], ...],
) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
    return zip(cycle, cycle[1:] + cycle[:1], strict=True)


@pytest.mark.parametrize(
    ("width", "height"),
    [(7, 6), (6, 5), (31, 20)],
    ids=["height-even", "width-even", "default-grid"],
)
def test_hamiltonian_cycle_covers_grid_once_and_wraps_adjacent(
    width: int,
    height: int,
) -> None:
    cycle = hamiltonian_cycle(width, height)

    assert len(cycle) == width * height
    assert len(set(cycle)) == width * height
    assert set(cycle) == {(x, y) for y in range(height) for x in range(width)}
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in _wrapped_pairs(cycle)
    )


def test_hamiltonian_cycle_rejects_odd_by_odd_grid() -> None:
    with pytest.raises(ValueError, match="even dimension"):
        hamiltonian_cycle(5, 3)


def test_cycle_aligned_snake_supports_wraparound_and_full_length() -> None:
    cycle = hamiltonian_cycle(6, 5)

    wrapped = cycle_aligned_snake(cycle, length=4, head_index=1)
    full = cycle_aligned_snake(cycle, length=len(cycle), head_index=17)

    assert wrapped == (cycle[1], cycle[0], cycle[-1], cycle[-2])
    assert len(set(full)) == len(cycle)
    assert is_cycle_aligned(wrapped, cycle)
    assert is_cycle_aligned(full, cycle)
    assert not is_cycle_aligned((cycle[1], cycle[2]), cycle)


def test_default_environment_snake_is_cycle_aligned() -> None:
    cycle = hamiltonian_cycle(31, 20)
    snake = ((15, 10), (14, 10), (13, 10))
    oracle = HamiltonianOracle(31, 20)

    assert is_cycle_aligned(snake, cycle)
    assert oracle.absolute_action(snake) == RIGHT


@pytest.mark.parametrize(
    ("current", "next_cell", "expected"),
    [
        ((2, 2), (2, 1), UP),
        ((2, 2), (3, 2), RIGHT),
        ((2, 2), (2, 3), DOWN),
        ((2, 2), (1, 2), LEFT),
    ],
)
def test_absolute_action_mapping(
    current: tuple[int, int],
    next_cell: tuple[int, int],
    expected: int,
) -> None:
    assert absolute_action(current, next_cell) == expected


@pytest.mark.parametrize(
    ("direction", "target", "expected"),
    [
        (UP, UP, 0),
        (UP, RIGHT, 1),
        (UP, LEFT, 2),
        (RIGHT, DOWN, 1),
        (RIGHT, UP, 2),
        (DOWN, LEFT, 1),
        (LEFT, DOWN, 2),
    ],
)
def test_relative_action_mapping(
    direction: int,
    target: int,
    expected: int,
) -> None:
    assert relative_action(direction, target) == expected


def test_relative_action_rejects_reversal() -> None:
    with pytest.raises(ValueError, match="cannot reverse"):
        relative_action(UP, DOWN)


def test_oracle_returns_absolute_and_relative_actions_at_a_turn() -> None:
    oracle = HamiltonianOracle(5, 4)
    snake = cycle_aligned_snake(oracle.cycle, length=3, head_index=4)

    assert snake[0] == (4, 0)
    assert oracle.act(snake, RIGHT, action_mode="absolute") == DOWN
    assert oracle.act(snake, RIGHT, action_mode="relative") == 1


@pytest.mark.parametrize("action_mode", ["absolute", "relative"])
def test_oracle_eventually_fills_small_board(action_mode: str) -> None:
    width, height = 4, 3
    oracle = HamiltonianOracle(width, height)
    snake = cycle_aligned_snake(oracle.cycle, length=3, head_index=2)
    direction = absolute_action(snake[1], snake[0])
    env = SnakeEnv(width=width, height=height, action_mode=action_mode)
    try:
        env.reset(seed=2026, options={"snake": snake, "direction": direction})
        terminated = False
        info: dict[str, object] = {}
        for _ in range((width * height) ** 2):
            action = oracle.act(
                env.snake,
                env.direction,
                action_mode=action_mode,
            )
            _, _, terminated, truncated, info = env.step(action)
            assert not truncated
            if terminated:
                break

        assert terminated
        assert info["won"] is True
        assert info["termination_reason"] == "board_filled"
        assert info["length"] == width * height
    finally:
        env.close()

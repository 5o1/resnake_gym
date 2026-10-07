"""A deterministic Hamiltonian-cycle oracle for rectangular Snake boards."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Literal, TypeAlias

Coord: TypeAlias = tuple[int, int]
ActionMode: TypeAlias = Literal["absolute", "relative"]

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

_DIRECTION_BY_DELTA: dict[Coord, int] = {
    (0, -1): UP,
    (1, 0): RIGHT,
    (0, 1): DOWN,
    (-1, 0): LEFT,
}


def hamiltonian_cycle(width: int, height: int) -> tuple[Coord, ...]:
    """Return a deterministic directed Hamiltonian cycle for a grid.

    A rectangular grid has such a cycle when both dimensions are at least two
    and at least one dimension is even. Coordinates use ``(x, y)`` with the
    origin in the upper-left corner.
    """

    width = _validate_dimension(width, name="width")
    height = _validate_dimension(height, name="height")
    if width % 2 and height % 2:
        raise ValueError("a Hamiltonian cycle requires at least one even dimension")

    if height % 2 == 0:
        return _even_height_cycle(width, height)

    # Transposing an even-height cycle covers the width-even case.
    transposed = _even_height_cycle(height, width)
    return tuple((y, x) for x, y in transposed)


def cycle_aligned_snake(
    cycle: Sequence[Coord],
    length: int,
    head_index: int = 0,
) -> tuple[Coord, ...]:
    """Create a head-first snake occupying predecessors of ``head_index``."""

    checked_cycle = _validated_cycle(cycle)
    if not isinstance(length, Integral) or isinstance(length, bool):
        raise TypeError("length must be an integer")
    length = int(length)
    if not 1 <= length <= len(checked_cycle):
        raise ValueError("length must be between 1 and the cycle length")
    if not isinstance(head_index, Integral) or isinstance(head_index, bool):
        raise TypeError("head_index must be an integer")
    head_index = int(head_index)
    if not 0 <= head_index < len(checked_cycle):
        raise ValueError("head_index must identify a cell in the cycle")

    return tuple(
        checked_cycle[(head_index - offset) % len(checked_cycle)]
        for offset in range(length)
    )


def is_cycle_aligned(
    snake: Sequence[Coord],
    cycle: Sequence[Coord],
) -> bool:
    """Return whether a head-first snake follows the cycle's predecessors."""

    try:
        checked_cycle = _validated_cycle(cycle)
        checked_snake = _coerce_coords(snake, name="snake")
        indices = {cell: index for index, cell in enumerate(checked_cycle)}
        _aligned_head_index(checked_snake, checked_cycle, indices)
    except (TypeError, ValueError):
        return False
    return True


def absolute_action(current: Coord, next_cell: Coord) -> int:
    """Return the absolute action that moves between two adjacent cells."""

    current = _coerce_coord(current, name="current")
    next_cell = _coerce_coord(next_cell, name="next_cell")
    delta = (next_cell[0] - current[0], next_cell[1] - current[1])
    try:
        return _DIRECTION_BY_DELTA[delta]
    except KeyError as exc:
        raise ValueError("current and next_cell must be edge-adjacent") from exc


def relative_action(direction: int, target_direction: int) -> int:
    """Map an absolute target to 0 straight, 1 right, or 2 left.

    A reversal cannot be represented by ReSnake's relative action space and is
    rejected.
    """

    direction = _validate_direction(direction, name="direction")
    target_direction = _validate_direction(
        target_direction,
        name="target_direction",
    )
    turn = (target_direction - direction) % 4
    if turn == 0:
        return 0
    if turn == 1:
        return 1
    if turn == 3:
        return 2
    raise ValueError("a relative action cannot reverse direction")


class HamiltonianOracle:
    """Policy that advances an aligned snake around a Hamiltonian cycle."""

    __slots__ = ("_indices", "cycle", "height", "width")

    width: int
    height: int
    cycle: tuple[Coord, ...]
    _indices: dict[Coord, int]

    def __init__(self, width: int, height: int) -> None:
        self.width = _validate_dimension(width, name="width")
        self.height = _validate_dimension(height, name="height")
        self.cycle = hamiltonian_cycle(self.width, self.height)
        self._indices = {cell: index for index, cell in enumerate(self.cycle)}

    def absolute_action(self, snake: Sequence[Coord]) -> int:
        """Return the next absolute direction from a snake's head cell."""

        if isinstance(snake, (str, bytes)) or not isinstance(snake, Sequence):
            raise TypeError("snake must be a sequence of coordinates")
        if not snake:
            raise ValueError("snake must contain at least one cell")
        return self.absolute_action_from_head(snake[0])

    def absolute_action_from_head(self, head: Coord) -> int:
        """Return the next absolute direction without scanning the snake body."""

        checked_head = _coerce_coord(head, name="head")
        try:
            head_index = self._indices[checked_head]
        except KeyError as exc:
            raise ValueError("snake head is not on the cycle") from exc
        next_cell = self.cycle[(head_index + 1) % len(self.cycle)]
        return absolute_action(checked_head, next_cell)

    def relative_action(self, snake: Sequence[Coord], direction: int) -> int:
        """Return the next relative action for an aligned snake."""

        target_direction = self.absolute_action(snake)
        return relative_action(direction, target_direction)

    def relative_action_from_head(self, head: Coord, direction: int) -> int:
        """Return the relative action without scanning the snake body."""

        target_direction = self.absolute_action_from_head(head)
        return relative_action(direction, target_direction)

    def act(
        self,
        snake: Sequence[Coord],
        direction: int | None = None,
        *,
        action_mode: ActionMode = "absolute",
    ) -> int:
        """Return the next action in the requested environment action mode."""

        if action_mode == "absolute":
            return self.absolute_action(snake)
        if action_mode == "relative":
            if direction is None:
                raise ValueError("direction is required for relative actions")
            return self.relative_action(snake, direction)
        raise ValueError("action_mode must be 'absolute' or 'relative'")

    def act_head(
        self,
        head: Coord,
        direction: int | None = None,
        *,
        action_mode: ActionMode = "absolute",
    ) -> int:
        """Fast action path for rollouts that have already checked alignment."""

        if action_mode == "absolute":
            return self.absolute_action_from_head(head)
        if action_mode == "relative":
            if direction is None:
                raise ValueError("direction is required for relative actions")
            return self.relative_action_from_head(head, direction)
        raise ValueError("action_mode must be 'absolute' or 'relative'")


def _even_height_cycle(width: int, height: int) -> tuple[Coord, ...]:
    cycle = [(x, 0) for x in range(width)]
    for y in range(1, height):
        if y % 2:
            cycle.extend((x, y) for x in range(width - 1, 0, -1))
        else:
            cycle.extend((x, y) for x in range(1, width))
    cycle.extend((0, y) for y in range(height - 1, 0, -1))
    return tuple(cycle)


def _validate_dimension(value: int, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 2:
        raise ValueError(f"{name} must be at least 2")
    return value


def _validate_direction(value: int, *, name: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not 0 <= int(value) <= 3
    ):
        raise ValueError(f"{name} must be an integer in [0, 3]")
    return int(value)


def _coerce_coord(value: object, *, name: str) -> Coord:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{name} must be an (x, y) pair")
    x, y = value
    if (
        not isinstance(x, Integral)
        or isinstance(x, bool)
        or not isinstance(y, Integral)
        or isinstance(y, bool)
    ):
        raise TypeError(f"{name} coordinates must be integers")
    return int(x), int(y)


def _coerce_coords(values: Sequence[Coord], *, name: str) -> tuple[Coord, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of coordinates")
    return tuple(_coerce_coord(value, name=f"{name} cell") for value in values)


def _validated_cycle(cycle: Sequence[Coord]) -> tuple[Coord, ...]:
    checked_cycle = _coerce_coords(cycle, name="cycle")
    if not checked_cycle:
        raise ValueError("cycle must contain at least one cell")
    if len(set(checked_cycle)) != len(checked_cycle):
        raise ValueError("cycle cells must be unique")
    wrapped_pairs = zip(
        checked_cycle,
        checked_cycle[1:] + checked_cycle[:1],
        strict=True,
    )
    if any(not _edge_adjacent(first, second) for first, second in wrapped_pairs):
        raise ValueError("consecutive cycle cells, including wraparound, must touch")
    return checked_cycle


def _aligned_head_index(
    snake: tuple[Coord, ...],
    cycle: tuple[Coord, ...],
    indices: dict[Coord, int],
) -> int:
    if not snake:
        raise ValueError("snake must contain at least one cell")
    if len(snake) > len(cycle):
        raise ValueError("snake cannot be longer than the cycle")
    try:
        head_index = indices[snake[0]]
    except KeyError as exc:
        raise ValueError("snake head is not on the cycle") from exc

    expected = tuple(
        cycle[(head_index - offset) % len(cycle)] for offset in range(len(snake))
    )
    if snake != expected:
        raise ValueError("snake is not aligned with the directed cycle")
    return head_index


def _edge_adjacent(first: Coord, second: Coord) -> bool:
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


__all__ = [
    "ActionMode",
    "Coord",
    "HamiltonianOracle",
    "absolute_action",
    "cycle_aligned_snake",
    "hamiltonian_cycle",
    "is_cycle_aligned",
    "relative_action",
]

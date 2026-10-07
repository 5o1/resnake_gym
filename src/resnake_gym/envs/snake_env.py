"""A deterministic, research-oriented Snake environment for Gymnasium."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled, ResetNeeded

Coord = tuple[int, int]

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

_DIRECTION_VECTORS: tuple[Coord, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
_DIRECTION_NAMES = {
    "up": UP,
    "right": RIGHT,
    "down": DOWN,
    "left": LEFT,
}


@dataclass(frozen=True, slots=True)
class SnakeState:
    """Immutable snapshot of the kernel state exposed to environment adapters.

    The mutable deque, set and rank grid remain implementation details.  Code
    which computes rewards, perturbations or diagnostics can consume one
    coherent snapshot without depending on those representations.
    """

    width: int
    height: int
    snake: tuple[Coord, ...]
    occupied: frozenset[Coord]
    direction: int
    food: Coord | None
    score: int
    logic_steps: int
    elapsed_seconds: float
    episode_started: bool
    episode_done: bool
    termination_reason: str | None
    won: bool


class SnakeEnv(gym.Env[np.ndarray, int]):
    """Grid Snake with a full-scene tensor observation.

    Coordinates use ``(x, y)`` with the origin in the upper-left corner. The
    snake sequence is head-first. One agent step advances ``frame_skip`` logic
    ticks, unless the episode ends sooner.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 10,
    }
    observation_channels = (
        "snake",
        "body_progress",
        "head",
        "tail",
        "food",
        "facing_up",
        "facing_right",
        "facing_down",
        "facing_left",
    )

    def __init__(
        self,
        *,
        width: int = 31,
        height: int = 20,
        grid_size: tuple[int, int] | None = None,
        logic_fps: float = 10.0,
        frame_skip: int = 1,
        max_logic_steps: int | None = None,
        max_episode_steps: int | None = None,
        max_episode_seconds: float | None = None,
        initial_length: int = 3,
        action_mode: str = "absolute",
        render_mode: str | None = None,
        cell_size: int = 20,
        food_reward: float = 1.0,
        death_reward: float = -1.0,
        living_reward: float = 0.0,
        win_reward: float = 1.0,
        timeout_reward: float = 0.0,
    ) -> None:
        super().__init__()

        if grid_size is not None:
            if len(grid_size) != 2:
                raise ValueError("grid_size must be a (width, height) pair")
            width, height = grid_size

        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("width and height must be integers")
        if width < 3 or height < 3:
            raise ValueError("width and height must both be at least 3")
        if not isinstance(initial_length, int) or not 1 <= initial_length <= width:
            raise ValueError("initial_length must be in [1, width]")
        if action_mode not in {"absolute", "relative"}:
            raise ValueError("action_mode must be 'absolute' or 'relative'")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None"
            )
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        if max_logic_steps is not None and max_episode_steps is not None:
            raise ValueError(
                "use max_logic_steps; max_episode_steps is only a "
                "direct-construction alias"
            )
        if max_logic_steps is None:
            max_logic_steps = max_episode_steps
        if max_logic_steps is not None and (
            not isinstance(max_logic_steps, int) or max_logic_steps < 1
        ):
            raise ValueError("max_logic_steps must be a positive integer or None")
        if max_episode_seconds is not None:
            max_episode_seconds = float(max_episode_seconds)
            if not np.isfinite(max_episode_seconds) or max_episode_seconds <= 0:
                raise ValueError(
                    "max_episode_seconds must be finite, positive, or None"
                )
        if not isinstance(cell_size, int) or cell_size < 1:
            raise ValueError("cell_size must be a positive integer")
        timeout_reward = float(timeout_reward)
        if not np.isfinite(timeout_reward):
            raise ValueError("timeout_reward must be finite")

        self.width = width
        self.height = height
        self.initial_length = initial_length
        self.action_mode = action_mode
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_logic_steps = max_logic_steps
        # Kept as a read-only-style compatibility attribute for direct users of
        # early releases. Gymnasium reserves this keyword in gym.make().
        self.max_episode_steps = max_logic_steps
        self.max_episode_seconds = max_episode_seconds
        self.cell_size = cell_size
        self.food_reward = float(food_reward)
        self.death_reward = float(death_reward)
        self.living_reward = float(living_reward)
        self.win_reward = float(win_reward)
        self.timeout_reward = timeout_reward

        self._logic_fps = 0.0
        self.logic_fps = logic_fps
        self.metadata = dict(type(self).metadata)
        self.metadata["render_fps"] = self.logic_fps

        action_count = 4 if action_mode == "absolute" else 3
        self.action_space = spaces.Discrete(action_count)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9, self.height, self.width),
            dtype=np.float32,
        )

        self._snake: deque[Coord] = deque()
        self._occupied: set[Coord] = set()
        # Tail=1 and head=length. Keeping this dense rank grid makes the
        # Markov body-order plane a vectorized operation even for long snakes.
        self._body_rank = np.zeros((self.height, self.width), dtype=np.int32)
        self._direction = RIGHT
        self._food: Coord | None = None
        self._score = 0
        self._logic_steps = 0
        self._elapsed_seconds = 0.0
        self._episode_started = False
        self._episode_done = False
        self._termination_reason: str | None = None
        self._won = False

        self._pygame: Any | None = None
        self._window: Any | None = None
        self._clock: Any | None = None

    @property
    def logic_fps(self) -> float:
        """Number of simulated logic ticks represented by one game second."""

        return self._logic_fps

    @logic_fps.setter
    def logic_fps(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("logic_fps must be a finite positive number")
        self._logic_fps = value
        if "metadata" in self.__dict__:
            self.metadata["render_fps"] = value

    @property
    def snake(self) -> tuple[Coord, ...]:
        """Current snake coordinates, ordered from head to tail."""

        return tuple(self._snake)

    @property
    def head(self) -> Coord:
        """Current head coordinate without copying the full snake sequence."""

        if not self._snake:
            raise ResetNeeded("call reset() before reading head")
        return self._snake[0]

    @property
    def direction(self) -> int:
        """Current absolute direction (0 up, 1 right, 2 down, 3 left)."""

        return self._direction

    @property
    def food(self) -> Coord | None:
        """Current food coordinate, or ``None`` when the board is full."""

        return self._food

    @property
    def state(self) -> SnakeState:
        """Return an immutable, internally consistent kernel snapshot."""

        return SnakeState(
            width=self.width,
            height=self.height,
            snake=tuple(self._snake),
            occupied=frozenset(self._occupied),
            direction=self._direction,
            food=self._food,
            score=self._score,
            logic_steps=self._logic_steps,
            elapsed_seconds=self._elapsed_seconds,
            episode_started=self._episode_started,
            episode_done=self._episode_done,
            termination_reason=self._termination_reason,
            won=self._won,
        )

    def clone_for_simulation(self) -> SnakeEnv:
        """Deep-copy the kernel for counterfactual rollouts.

        Callers must use the returned kernel only for simulation.  In
        particular, it must not be stepped concurrently with this instance.
        """

        return copy.deepcopy(self)

    def set_time_limits(
        self,
        *,
        max_logic_steps: int | None,
        max_episode_seconds: float | None,
    ) -> None:
        """Change evaluation cutoffs without modifying geometric state."""

        if max_logic_steps is not None and (
            not isinstance(max_logic_steps, int) or max_logic_steps < 1
        ):
            raise ValueError("max_logic_steps must be a positive integer or None")
        if max_episode_seconds is not None:
            max_episode_seconds = float(max_episode_seconds)
            if not np.isfinite(max_episode_seconds) or max_episode_seconds <= 0:
                raise ValueError(
                    "max_episode_seconds must be finite, positive, or None"
                )
        self.max_logic_steps = max_logic_steps
        self.max_episode_steps = max_logic_steps
        self.max_episode_seconds = max_episode_seconds

    def relocate_food(self, cell: Coord) -> None:
        """Move food in an active episode for a controlled experiment."""

        if not self._episode_started or self._episode_done:
            raise ResetNeeded("food can only move during an active episode")
        cell = self._coerce_coord(cell, name="food")
        if not self.contains_cell(cell):
            raise ValueError("food must be inside the board")
        if cell in self._occupied:
            raise ValueError("food cannot overlap the snake")
        self._food = cell

    def contains_cell(self, cell: Coord) -> bool:
        """Return whether an ``(x, y)`` coordinate is on the current board."""

        return self._inside(cell)

    def observation(self) -> np.ndarray:
        """Return the current Markov scene tensor."""

        if not self._episode_started:
            raise ResetNeeded("call reset() before reading an observation")
        return self._get_observation()

    def render_rgb_array(self) -> np.ndarray:
        """Render the current state independently of ``render_mode``."""

        if not self._episode_started:
            raise ResetNeeded("call reset() before rendering")
        return self._render_rgb_array()

    def render_ansi(self) -> str:
        """Render the current state as text independently of ``render_mode``."""

        if not self._episode_started:
            raise ResetNeeded("call reset() before rendering")
        return self._render_ansi()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode, optionally from an explicitly supplied state.

        Supported options are ``snake`` (head-first coordinates), ``direction``,
        ``food`` and ``logic_fps``. Explicit states are useful for controlled
        experiments and regression tests.
        """

        super().reset(seed=seed)
        options = {} if options is None else dict(options)

        unknown = set(options) - {"snake", "direction", "food", "logic_fps"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown reset option(s): {names}")

        if "logic_fps" in options:
            self.logic_fps = options["logic_fps"]

        if "snake" in options:
            snake = self._validate_snake(options["snake"])
        else:
            y = self.height // 2
            head_x = max(self.initial_length - 1, self.width // 2)
            snake = [(head_x - offset, y) for offset in range(self.initial_length)]
        if len(snake) == self.width * self.height:
            raise ValueError("a reset state must leave at least one free food cell")

        inferred_direction: int | None = None
        if len(snake) >= 2:
            dx = snake[0][0] - snake[1][0]
            dy = snake[0][1] - snake[1][1]
            inferred_direction = _DIRECTION_VECTORS.index((dx, dy))
        if "direction" in options:
            direction = self._parse_direction(options["direction"])
            if inferred_direction is not None and direction != inferred_direction:
                raise ValueError("direction must point from the neck toward the head")
        elif inferred_direction is not None:
            direction = inferred_direction
        else:
            direction = RIGHT

        self._snake = deque(snake)
        self._occupied = set(snake)
        self._body_rank.fill(0)
        snake_length = len(snake)
        for index, (x, y) in enumerate(snake):
            self._body_rank[y, x] = snake_length - index
        self._direction = direction

        if "food" in options:
            requested_food = options["food"]
            if requested_food is None:
                raise ValueError("food cannot be None in an active episode")
            else:
                food = self._coerce_coord(requested_food, name="food")
                if not self._inside(food):
                    raise ValueError("food must be inside the board")
                if food in self._occupied:
                    raise ValueError("food cannot overlap the snake")
                self._food = food
        else:
            self._food = self._sample_food()

        self._score = 0
        self._logic_steps = 0
        self._elapsed_seconds = 0.0
        self._episode_started = True
        self._episode_done = False
        self._termination_reason = None
        self._won = False

        observation = self._get_observation()
        info = self._get_info(ticks_advanced=0)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action and advance up to ``frame_skip`` logic ticks."""

        if not self._episode_started:
            raise ResetNeeded("call reset() before step()")
        if self._episode_done:
            raise ResetNeeded("the episode is over; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}")

        action = int(action)
        self._apply_action(action)

        total_reward = 0.0
        terminated = False
        truncated = False
        ticks_advanced = 0

        for _ in range(self.frame_skip):
            tick_reward, terminated = self._advance_one_tick()
            total_reward += tick_reward
            ticks_advanced += 1

            self._logic_steps += 1
            self._elapsed_seconds += 1.0 / self.logic_fps

            if not terminated and self._time_limit_reached():
                truncated = True
                self._termination_reason = "time_limit"
                total_reward += self.timeout_reward

            if self.render_mode == "human":
                self.render()

            if terminated or truncated:
                break

        self._episode_done = terminated or truncated
        observation = self._get_observation()
        info = self._get_info(ticks_advanced=ticks_advanced)
        return observation, float(total_reward), terminated, truncated, info

    def _apply_action(self, action: int) -> None:
        if self.action_mode == "relative":
            if action == 1:
                self._direction = (self._direction + 1) % 4
            elif action == 2:
                self._direction = (self._direction - 1) % 4
            return

        requested = action
        if requested != (self._direction + 2) % 4:
            self._direction = requested

    def _advance_one_tick(self) -> tuple[float, bool]:
        reward = self.living_reward
        dx, dy = _DIRECTION_VECTORS[self._direction]
        head_x, head_y = self._snake[0]
        new_head = (head_x + dx, head_y + dy)

        if not self._inside(new_head):
            self._termination_reason = "wall_collision"
            return reward + self.death_reward, True

        eating = new_head == self._food
        old_tail = self._snake[-1]
        tail_will_move = not eating
        if new_head in self._occupied and not (tail_will_move and new_head == old_tail):
            self._termination_reason = "self_collision"
            return reward + self.death_reward, True

        if tail_will_move:
            np.subtract(
                self._body_rank,
                1,
                out=self._body_rank,
                where=self._body_rank > 0,
            )
            self._snake.pop()
            self._occupied.remove(old_tail)
        new_rank = len(self._snake) + 1
        self._snake.appendleft(new_head)
        self._occupied.add(new_head)
        self._body_rank[new_head[1], new_head[0]] = new_rank
        if eating:
            self._score += 1
            reward += self.food_reward
            if len(self._snake) == self.width * self.height:
                self._food = None
                self._termination_reason = "board_filled"
                self._won = True
                return reward + self.win_reward, True
            self._food = self._sample_food()
        return reward, False

    def _time_limit_reached(self) -> bool:
        if (
            self.max_logic_steps is not None
            and self._logic_steps >= self.max_logic_steps
        ):
            return True
        return (
            self.max_episode_seconds is not None
            and self._elapsed_seconds + 1e-12 >= self.max_episode_seconds
        )

    def _get_observation(self) -> np.ndarray:
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        length = len(self._snake)
        observation[0] = self._body_rank > 0
        np.divide(self._body_rank, length, out=observation[1], casting="unsafe")

        head_x, head_y = self._snake[0]
        tail_x, tail_y = self._snake[-1]
        observation[2, head_y, head_x] = 1.0
        observation[3, tail_y, tail_x] = 1.0

        if self._food is not None:
            food_x, food_y = self._food
            observation[4, food_y, food_x] = 1.0

        observation[5 + self._direction, head_y, head_x] = 1.0
        return observation

    def _get_info(self, *, ticks_advanced: int) -> dict[str, Any]:
        return {
            "length": len(self._snake),
            "score": self._score,
            "logic_fps": self.logic_fps,
            "logic_dt": 1.0 / self.logic_fps,
            "decision_fps": self.logic_fps / self.frame_skip,
            "frame_skip": self.frame_skip,
            "logic_steps": self._logic_steps,
            "elapsed_seconds": self._elapsed_seconds,
            "ticks_advanced": ticks_advanced,
            "termination_reason": self._termination_reason,
            "won": self._won,
        }

    def _sample_food(self) -> Coord | None:
        free_count = self.width * self.height - len(self._occupied)
        if free_count == 0:
            return None

        selected = int(self.np_random.integers(free_count))
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                if cell in self._occupied:
                    continue
                if selected == 0:
                    return cell
                selected -= 1
        raise RuntimeError("failed to sample a free food cell")

    def _validate_snake(self, value: Any) -> list[Coord]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("snake must be a non-empty sequence of coordinates")
        snake = [self._coerce_coord(cell, name="snake cell") for cell in value]
        if not snake:
            raise ValueError("snake must contain at least one cell")
        if len(snake) > self.width * self.height:
            raise ValueError("snake is longer than the board capacity")
        if any(not self._inside(cell) for cell in snake):
            raise ValueError("every snake cell must be inside the board")
        if len(set(snake)) != len(snake):
            raise ValueError("snake cells must be unique")
        for first, second in zip(snake, snake[1:], strict=False):
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
                raise ValueError("consecutive snake cells must be edge-adjacent")
        return snake

    @staticmethod
    def _coerce_coord(value: Any, *, name: str) -> Coord:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise TypeError(f"{name} must be an (x, y) pair")
        x, y = value
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            raise TypeError(f"{name} coordinates must be integers")
        return int(x), int(y)

    def _parse_direction(self, value: Any) -> int:
        if isinstance(value, str):
            try:
                return _DIRECTION_NAMES[value.lower()]
            except KeyError as exc:
                raise ValueError(
                    "direction must be up, right, down, left, or an integer in [0, 3]"
                ) from exc
        if isinstance(value, (int, np.integer)) and 0 <= int(value) <= 3:
            return int(value)
        raise ValueError(
            "direction must be up, right, down, left, or an integer in [0, 3]"
        )

    def _inside(self, cell: Coord) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def render(self) -> np.ndarray | str | None:
        if not self._episode_started:
            raise ResetNeeded("call reset() before render()")
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()

        frame = self._render_rgb_array()
        if self.render_mode == "rgb_array":
            return frame

        self._render_human(frame)
        return None

    def _render_rgb_array(self) -> np.ndarray:
        grid = np.full((self.height, self.width, 3), (24, 27, 32), dtype=np.uint8)

        length = len(self._snake)
        for index, (x, y) in enumerate(reversed(self._snake)):
            fraction = (index + 1) / length
            grid[y, x] = (
                38,
                int(105 + 95 * fraction),
                int(90 + 65 * fraction),
            )

        head_x, head_y = self._snake[0]
        grid[head_y, head_x] = (76, 225, 150)
        if self._food is not None:
            food_x, food_y = self._food
            grid[food_y, food_x] = (244, 82, 91)

        frame = np.repeat(grid, self.cell_size, axis=0)
        frame = np.repeat(frame, self.cell_size, axis=1)

        if self.cell_size >= 4:
            frame[:: self.cell_size, :, :] = (47, 52, 59)
            frame[:, :: self.cell_size, :] = (47, 52, 59)
        return frame

    def _render_ansi(self) -> str:
        board = [["." for _ in range(self.width)] for _ in range(self.height)]
        if self._food is not None:
            food_x, food_y = self._food
            board[food_y][food_x] = "*"
        for x, y in reversed(self._snake):
            board[y][x] = "o"
        head_x, head_y = self._snake[0]
        board[head_y][head_x] = "H"
        return "\n".join("".join(row) for row in board)

    def _render_human(self, frame: np.ndarray) -> None:
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise DependencyNotInstalled(
                    "human rendering requires pygame; install `resnake-gym[render]`"
                ) from exc
            self._pygame = pygame
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("ReSnake Gym")
            self._clock = pygame.time.Clock()

        assert self._window is not None
        assert self._clock is not None
        surface = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surface, (0, 0))
        self._pygame.event.pump()
        self._pygame.display.flip()
        self._clock.tick(self.logic_fps)

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._window = None
        self._clock = None


__all__ = ["SnakeEnv", "SnakeState", "UP", "RIGHT", "DOWN", "LEFT"]

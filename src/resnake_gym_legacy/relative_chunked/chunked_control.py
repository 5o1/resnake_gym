"""Historical receding-horizon control for the relative-action Snake env."""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from resnake_gym.envs.snake_env import SnakeEnv

TickRange = int | tuple[int, int]


@dataclass(frozen=True, slots=True)
class _CommandDelivery:
    """Result of installing one timestamped command into the tick schedule."""

    sequence: int
    origin_tick: int
    target_start_tick: int
    target_end_tick: int
    expired_prefix_count: int
    installed_action_count: int
    overwritten_action_count: int
    shadowed_action_count: int


class ChunkedControlWrapper(gym.Wrapper):
    """Expose fixed-rate policy calls with delayed observations and action chunks.

    The wrapped :class:`~resnake_gym.envs.SnakeEnv` remains the deterministic
    one-tick game kernel.  One wrapper ``step`` submits a new relative-action
    chunk and then advances up to ``control_interval_ticks`` game ticks.  The
    period start is the chunk's origin tick: element ``j`` targets the action
    executed at ``origin + j``.  A delayed packet therefore loses its expired
    prefix instead of shifting the whole plan later in time.  Newly submitted
    sequences replace older plans only at overlapping target ticks; an older
    packet that arrives late cannot overwrite a newer plan.  If no action is
    scheduled for the current tick, action 0 (continue straight) is applied;
    no board-dependent action correction is performed.

    Observations are sampled from the recent state history.  Their timestamps
    are clamped so a consumer never receives time in reverse, which can cause a
    repeated observation when a newly sampled delay would point before the
    previously delivered state.  The returned period reward is
    ``sum(reward_discount_per_tick**j * tick_reward[j])``; ``info`` also keeps
    the undiscounted sum and raw tick rewards for diagnostics.
    """

    _NO_ACTION = np.int8(-1)

    def __init__(
        self,
        env: SnakeEnv,
        *,
        control_interval_ticks: int,
        chunk_length: int,
        observation_age_ticks: TickRange = 0,
        command_delay_ticks: TickRange = 0,
        reward_discount_per_tick: float = 1.0,
    ) -> None:
        if not isinstance(env, SnakeEnv):
            raise TypeError("ChunkedControlWrapper must directly wrap SnakeEnv")
        if env.frame_skip != 1:
            raise ValueError("the wrapped SnakeEnv must use frame_skip=1")
        if env.action_mode != "relative":
            raise ValueError("the wrapped SnakeEnv must use action_mode='relative'")
        if not _is_positive_int(control_interval_ticks):
            raise ValueError("control_interval_ticks must be a positive integer")
        if not _is_positive_int(chunk_length):
            raise ValueError("chunk_length must be a positive integer")
        reward_discount_per_tick = float(reward_discount_per_tick)
        if not math.isfinite(reward_discount_per_tick) or not (
            0.0 <= reward_discount_per_tick <= 1.0
        ):
            raise ValueError("reward_discount_per_tick must be finite and in [0, 1]")

        super().__init__(env)
        self.control_interval_ticks = int(control_interval_ticks)
        self.chunk_length = int(chunk_length)
        self.observation_age_range = _normalize_tick_range(
            observation_age_ticks,
            name="observation_age_ticks",
        )
        self.command_delay_range = _normalize_tick_range(
            command_delay_ticks,
            name="command_delay_ticks",
        )
        self.reward_discount_per_tick = reward_discount_per_tick

        self.action_space = spaces.MultiDiscrete(
            np.full(self.chunk_length, 3, dtype=np.int64)
        )
        self.observation_space = spaces.Dict(
            {
                "delayed_state": env.observation_space,
                "observation_age": spaces.Box(
                    low=0,
                    high=self.observation_age_range[1],
                    shape=(1,),
                    dtype=np.int32,
                ),
                "new_observation": spaces.MultiBinary(1),
                "previous_chunk": spaces.MultiDiscrete(
                    np.full(self.chunk_length, 3, dtype=np.int64)
                ),
                "executed_actions": spaces.Box(
                    low=-1,
                    high=2,
                    shape=(self.control_interval_ticks,),
                    dtype=np.int8,
                ),
            }
        )

        self._timing_rng = np.random.default_rng()
        self._state_history: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.observation_age_range[1] + 1
        )
        # target tick -> (action, sequence, origin tick, chunk index)
        self._scheduled_actions: dict[int, tuple[int, int, int, int]] = {}
        # Heap order is arrival tick, then sequence number.
        self._pending_commands: list[tuple[int, int, int, tuple[int, ...]]] = []
        self._previous_chunk = np.zeros(self.chunk_length, dtype=np.int64)
        self._last_executed_actions = np.full(
            self.control_interval_ticks,
            self._NO_ACTION,
            dtype=np.int8,
        )
        self._current_tick = 0
        self._last_observation_tick = 0
        self._next_command_sequence = 0
        self._latest_installed_command_sequence = -1
        self._episode_started = False
        self._episode_done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset game and asynchronous timing state."""

        if seed is not None:
            self._timing_rng = np.random.default_rng(seed)

        state, base_info = self.env.reset(seed=seed, options=options)
        self._current_tick = int(base_info["logic_steps"])
        self._last_observation_tick = self._current_tick
        self._state_history.clear()
        self._state_history.append((self._current_tick, state.copy()))
        self._scheduled_actions.clear()
        self._pending_commands.clear()
        self._previous_chunk.fill(0)
        self._last_executed_actions.fill(self._NO_ACTION)
        self._next_command_sequence = 0
        self._latest_installed_command_sequence = -1
        self._episode_started = True
        self._episode_done = False

        observation = self._make_observation(
            state=state,
            observation_tick=self._current_tick,
            is_new=True,
        )
        info = dict(base_info)
        info.update(
            {
                "control_interval_ticks": self.control_interval_ticks,
                "chunk_length": self.chunk_length,
                "control_ticks_requested": 0,
                "control_ticks_advanced": 0,
                "period_start_tick": self._current_tick,
                "period_end_tick": self._current_tick,
                "submitted_chunk_origin_tick": None,
                "submitted_target_start_tick": None,
                "submitted_target_end_tick": None,
                "executed_actions": np.empty(0, dtype=np.int8),
                "executed_action_sources": np.empty(0, dtype=np.int64),
                "executed_action_target_ticks": np.empty(0, dtype=np.int64),
                "executed_action_chunk_indices": np.empty(0, dtype=np.int64),
                "tick_rewards": np.empty(0, dtype=np.float64),
                "reward_discount_per_tick": self.reward_discount_per_tick,
                "undiscounted_period_reward": 0.0,
                "discounted_period_reward": 0.0,
                "queue_remaining": 0,
                "queue_target_ticks": (),
                "queue_action_sources": (),
                "pending_command_count": 0,
                "expired_action_count": 0,
                "expired_command_prefixes": (),
                "observation_tick": self._current_tick,
                "requested_observation_age": 0,
                "observation_age": 0,
                "new_observation": True,
                "terminated_early": False,
            }
        )
        return observation, info

    def step(
        self, action: np.ndarray | Sequence[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Submit one chunk and execute one fixed control period."""

        if not self._episode_started:
            raise gym.error.ResetNeeded("call reset() before step()")
        if self._episode_done:
            raise gym.error.ResetNeeded(
                "the episode is over; call reset() before step()"
            )

        chunk = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(chunk):
            raise ValueError(f"invalid action chunk {action!r} for {self.action_space}")
        # Copy caller-owned memory because the command may arrive in a later step.
        chunk = chunk.copy()

        period_start_tick = self._current_tick
        command_sequence = self._next_command_sequence
        self._next_command_sequence += 1
        sampled_command_delay = self._sample_range(self.command_delay_range)
        command_origin_tick = period_start_tick
        command_target_end_tick = command_origin_tick + self.chunk_length - 1
        command_arrival_tick = command_origin_tick + sampled_command_delay
        heapq.heappush(
            self._pending_commands,
            (
                command_arrival_tick,
                command_sequence,
                command_origin_tick,
                tuple(map(int, chunk)),
            ),
        )
        self._previous_chunk = chunk

        executed_actions: list[int] = []
        executed_action_sources: list[int] = []
        executed_action_target_ticks: list[int] = []
        executed_action_chunk_indices: list[int] = []
        tick_rewards: list[float] = []
        deliveries: list[_CommandDelivery] = []
        queue_exhausted_tick_offsets: list[int] = []
        terminated = False
        truncated = False
        base_info: dict[str, Any] = {}
        latest_state: np.ndarray | None = None

        for tick_offset in range(self.control_interval_ticks):
            deliveries.extend(self._deliver_commands())

            target_tick = self._current_tick
            scheduled = self._scheduled_actions.pop(target_tick, None)
            if scheduled is not None:
                (
                    tick_action,
                    source_sequence,
                    _source_origin_tick,
                    source_chunk_index,
                ) = scheduled
            else:
                tick_action = 0
                source_sequence = -1
                source_chunk_index = -1
                queue_exhausted_tick_offsets.append(tick_offset)

            latest_state, tick_reward, terminated, truncated, base_info = self.env.step(
                tick_action
            )
            advanced = int(base_info["ticks_advanced"])
            if advanced != 1:
                raise RuntimeError(
                    "frame_skip=1 SnakeEnv must advance exactly one tick per live step"
                )
            self._current_tick = int(base_info["logic_steps"])
            self._state_history.append((self._current_tick, latest_state.copy()))
            executed_actions.append(tick_action)
            executed_action_sources.append(source_sequence)
            executed_action_target_ticks.append(target_tick)
            executed_action_chunk_indices.append(source_chunk_index)
            tick_rewards.append(float(tick_reward))
            if terminated or truncated:
                break

        assert latest_state is not None
        actual_ticks = len(executed_actions)
        undiscounted_period_reward = math.fsum(tick_rewards)
        discounted_period_reward = math.fsum(
            reward * self.reward_discount_per_tick**tick_offset
            for tick_offset, reward in enumerate(tick_rewards)
        )
        self._last_executed_actions.fill(self._NO_ACTION)
        self._last_executed_actions[:actual_ticks] = executed_actions

        requested_observation_age = self._sample_range(self.observation_age_range)
        requested_observation_tick = max(
            0, self._current_tick - requested_observation_age
        )
        observation_tick = max(
            self._last_observation_tick,
            requested_observation_tick,
        )
        delayed_state = self._state_at_tick(observation_tick)
        is_new_observation = observation_tick > self._last_observation_tick
        self._last_observation_tick = observation_tick
        actual_observation_age = self._current_tick - observation_tick

        observation = self._make_observation(
            state=delayed_state,
            observation_tick=observation_tick,
            is_new=is_new_observation,
        )
        self._episode_done = terminated or truncated

        arrived_sequences = [delivery.sequence for delivery in deliveries]
        installed_sequences = [
            delivery.sequence
            for delivery in deliveries
            if delivery.installed_action_count > 0
        ]
        discarded_sequences = [
            delivery.sequence
            for delivery in deliveries
            if delivery.shadowed_action_count > 0
            and delivery.installed_action_count == 0
        ]
        fully_expired_sequences = [
            delivery.sequence
            for delivery in deliveries
            if delivery.expired_prefix_count == self.chunk_length
        ]
        expired_prefixes = tuple(
            (delivery.sequence, delivery.expired_prefix_count)
            for delivery in deliveries
            if delivery.expired_prefix_count > 0
        )
        arrived_target_ranges = tuple(
            (
                delivery.sequence,
                delivery.target_start_tick,
                delivery.target_end_tick,
            )
            for delivery in deliveries
        )
        installed_action_count = sum(
            delivery.installed_action_count for delivery in deliveries
        )
        overwritten_action_count = sum(
            delivery.overwritten_action_count for delivery in deliveries
        )
        shadowed_action_count = sum(
            delivery.shadowed_action_count for delivery in deliveries
        )
        queue_target_ticks = tuple(sorted(self._scheduled_actions))
        queue_action_sources = tuple(
            self._scheduled_actions[target_tick][1]
            for target_tick in queue_target_ticks
        )

        info = dict(base_info)
        info.update(
            {
                "ticks_advanced": actual_ticks,
                "control_interval_ticks": self.control_interval_ticks,
                "chunk_length": self.chunk_length,
                "control_ticks_requested": self.control_interval_ticks,
                "control_ticks_advanced": actual_ticks,
                "period_start_tick": period_start_tick,
                "period_end_tick": self._current_tick,
                "submitted_chunk": chunk.copy(),
                "submitted_command_sequence": command_sequence,
                "submitted_chunk_origin_tick": command_origin_tick,
                "submitted_target_start_tick": command_origin_tick,
                "submitted_target_end_tick": command_target_end_tick,
                "sampled_command_delay_ticks": sampled_command_delay,
                "command_arrival_tick": command_arrival_tick,
                "command_arrived_this_period": command_sequence in arrived_sequences,
                "arrived_command_sequences": tuple(arrived_sequences),
                "arrived_command_target_ranges": arrived_target_ranges,
                "installed_command_sequences": tuple(installed_sequences),
                "discarded_stale_command_sequences": tuple(discarded_sequences),
                "fully_expired_command_sequences": tuple(fully_expired_sequences),
                "installed_command_sequence": (self._latest_installed_command_sequence),
                "expired_action_count": sum(
                    delivery.expired_prefix_count for delivery in deliveries
                ),
                "expired_command_prefixes": expired_prefixes,
                "installed_action_count": installed_action_count,
                "overwritten_action_count": overwritten_action_count,
                "shadowed_action_count": shadowed_action_count,
                "executed_actions": np.asarray(executed_actions, dtype=np.int8),
                "executed_action_sources": np.asarray(
                    executed_action_sources, dtype=np.int64
                ),
                "executed_action_target_ticks": np.asarray(
                    executed_action_target_ticks, dtype=np.int64
                ),
                "executed_action_chunk_indices": np.asarray(
                    executed_action_chunk_indices, dtype=np.int64
                ),
                "tick_rewards": np.asarray(tick_rewards, dtype=np.float64),
                "reward_discount_per_tick": self.reward_discount_per_tick,
                "undiscounted_period_reward": undiscounted_period_reward,
                "discounted_period_reward": discounted_period_reward,
                "queue_remaining": len(self._scheduled_actions),
                "queue_target_ticks": queue_target_ticks,
                "queue_action_sources": queue_action_sources,
                "queue_exhausted_tick_offsets": tuple(queue_exhausted_tick_offsets),
                "pending_command_count": len(self._pending_commands),
                "requested_observation_age": requested_observation_age,
                "requested_observation_tick": requested_observation_tick,
                "observation_tick": observation_tick,
                "observation_age": actual_observation_age,
                "new_observation": is_new_observation,
                "terminated_early": actual_ticks < self.control_interval_ticks,
                "decision_fps": self.env.logic_fps / self.control_interval_ticks,
            }
        )
        return observation, discounted_period_reward, terminated, truncated, info

    def _deliver_commands(self) -> list[_CommandDelivery]:
        deliveries: list[_CommandDelivery] = []
        while (
            self._pending_commands
            and self._pending_commands[0][0] <= self._current_tick
        ):
            _, sequence, origin_tick, chunk = heapq.heappop(self._pending_commands)
            expired_prefix_count = min(
                len(chunk), max(0, self._current_tick - origin_tick)
            )
            installed_action_count = 0
            overwritten_action_count = 0
            shadowed_action_count = 0

            for chunk_index in range(expired_prefix_count, len(chunk)):
                target_tick = origin_tick + chunk_index
                existing = self._scheduled_actions.get(target_tick)
                if existing is not None and existing[1] > sequence:
                    shadowed_action_count += 1
                    continue
                if existing is not None and existing[1] < sequence:
                    overwritten_action_count += 1
                self._scheduled_actions[target_tick] = (
                    chunk[chunk_index],
                    sequence,
                    origin_tick,
                    chunk_index,
                )
                installed_action_count += 1

            if installed_action_count > 0:
                self._latest_installed_command_sequence = max(
                    self._latest_installed_command_sequence,
                    sequence,
                )
            deliveries.append(
                _CommandDelivery(
                    sequence=sequence,
                    origin_tick=origin_tick,
                    target_start_tick=origin_tick,
                    target_end_tick=origin_tick + len(chunk) - 1,
                    expired_prefix_count=expired_prefix_count,
                    installed_action_count=installed_action_count,
                    overwritten_action_count=overwritten_action_count,
                    shadowed_action_count=shadowed_action_count,
                )
            )
        return deliveries

    def _state_at_tick(self, tick: int) -> np.ndarray:
        for history_tick, state in reversed(self._state_history):
            if history_tick == tick:
                return state
        available = [history_tick for history_tick, _ in self._state_history]
        raise RuntimeError(
            f"observation tick {tick} is missing from retained history {available}"
        )

    def _make_observation(
        self,
        *,
        state: np.ndarray,
        observation_tick: int,
        is_new: bool,
    ) -> dict[str, np.ndarray]:
        return {
            "delayed_state": state.copy(),
            "observation_age": np.asarray(
                [self._current_tick - observation_tick], dtype=np.int32
            ),
            "new_observation": np.asarray([is_new], dtype=np.int8),
            "previous_chunk": self._previous_chunk.copy(),
            "executed_actions": self._last_executed_actions.copy(),
        }

    def _sample_range(self, bounds: tuple[int, int]) -> int:
        low, high = bounds
        return int(self._timing_rng.integers(low, high, endpoint=True))


def _is_positive_int(value: Any) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and value > 0
    )


def _normalize_tick_range(value: TickRange, *, name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        low = high = int(value)
    elif (
        isinstance(value, tuple)
        and len(value) == 2
        and all(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool)
            for item in value
        )
    ):
        low, high = map(int, value)
    else:
        raise TypeError(f"{name} must be an integer or an inclusive (min, max) pair")
    if low < 0 or high < low:
        raise ValueError(f"{name} must satisfy 0 <= min <= max")
    return low, high


__all__ = ["ChunkedControlWrapper"]

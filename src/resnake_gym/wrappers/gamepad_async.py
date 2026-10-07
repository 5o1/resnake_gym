"""Tick-event simulation of delayed observations and timestamped gamepad chunks.

Timing motivation: Black et al., arXiv:2506.07339, section 3. This is NOT
RTC flow inpainting and does not measure wall-clock inference performance.
"""

import heapq

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

import resnake_gym.gamepad as gamepad
from resnake_gym.envs.gamepad_env import GamepadSnakeEnv


def tick_range(value, minimum=0):
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or any(type(x) is not int for x in pair):
        raise ValueError("timing must be an integer or an inclusive integer pair")
    if not minimum <= pair[0] <= pair[1]:
        raise ValueError("invalid timing range")
    return pair


class AsyncGamepad(gym.Wrapper):
    """Each call submits L reports and advances K randomly sampled logic ticks.

    Frame delivery and command delivery are independent event queues. A frame
    captured at t can only be observed after arrival; reordered old frames are
    discarded. Chunk j targets origin+j, not arrival+j. Missing targets execute
    a neutral report. Feedback is ideal, immediate HID feedback at call time;
    it is not inferred from the requested controls. No state-based correction.
    """

    def __init__(
        self,
        env,
        *,
        chunk_length=8,
        decision_ticks=(2, 5),
        observation_delay=(0, 3),
        command_delay=(0, 2),
        frame_drop_probability=0.1,
        gamma=0.99,
        time_features=False,
    ):
        if not isinstance(env, GamepadSnakeEnv) or env.frame_skip != 1:
            raise ValueError("AsyncGamepad requires a one-tick GamepadSnakeEnv")
        if type(chunk_length) is not int or chunk_length < 1:
            raise ValueError("chunk_length must be positive")
        if not 0 <= frame_drop_probability <= 1 or not 0 <= gamma <= 1:
            raise ValueError("drop probability and gamma must be in [0, 1]")
        super().__init__(env)
        self.chunk_length = chunk_length
        self.decision_ticks = tick_range(decision_ticks, 1)
        self.observation_delay = tick_range(observation_delay)
        self.command_delay = tick_range(command_delay)
        self.drop_probability = frame_drop_probability
        self.gamma = gamma
        self.time_features = time_features
        self.fps = env.logic_fps
        k = self.decision_ticks[1]
        self.action_space = spaces.Box(
            np.tile(gamepad.LOW, (chunk_length, 1)),
            np.tile(gamepad.HIGH, (chunk_length, 1)),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "board": env.observation_space,
                "report": gamepad.action_space(),
                "timing": spaces.Box(0, np.inf, shape=(3,), dtype=np.float32),
                "history": spaces.Box(
                    np.tile(gamepad.LOW, (k, 1)),
                    np.tile(gamepad.HIGH, (k, 1)),
                    dtype=np.float32,
                ),
                "history_mask": spaces.MultiBinary(k),
                "previous_chunk": self.action_space,
                "previous_age": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self._ready = False
        if time_features:
            self.observation_space.spaces.update(
                {
                    "time_context": spaces.Box(0, np.inf, shape=(5,), dtype=np.float32),
                    "target_dt": spaces.Box(
                        -np.inf, np.inf, shape=(chunk_length,), dtype=np.float32
                    ),
                    "history_age": spaces.Box(0, np.inf, shape=(k,), dtype=np.float32),
                }
            )

    def _draw(self, bounds):
        return int(self._rng.integers(bounds[0], bounds[1] + 1))

    def reset(self, *, seed=None, options=None):
        board, info = self.env.reset(seed=seed, options=options)
        self.fps = self.env.logic_fps
        if seed is not None or not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(seed)
        self.tick = 0
        self._frame_tick = 0
        self._board = board.copy()
        self._frames = []
        self._commands = []
        self._schedule = {}
        self._sequence = 0
        self._ready = True
        self._previous = np.zeros((self.chunk_length, 20), dtype=np.float32)
        self._origin = 0
        return self._observe([], 0, True), info

    def _observe(self, reports, elapsed, new_frame):
        history = np.zeros((self.decision_ticks[1], 20), dtype=np.float32)
        mask = np.zeros(self.decision_ticks[1], dtype=np.int8)
        if reports:
            history[: len(reports)] = reports
            mask[: len(reports)] = 1
        observation = {
            "board": self._board.copy(),
            "report": self.env.last_report.copy(),
            "timing": np.array(
                [
                    (self.tick - self._frame_tick) / self.fps,
                    elapsed / self.fps,
                    float(new_frame),
                ],
                dtype=np.float32,
            ),
            "history": history,
            "history_mask": mask,
            "previous_chunk": self._previous.copy(),
            "previous_age": np.array(
                [(self.tick - self._origin) / self.fps], dtype=np.float32
            ),
        }
        if self.time_features:
            ages = np.zeros(self.decision_ticks[1], np.float32)
            ages[: len(reports)] = np.arange(len(reports) - 1, -1, -1) / self.fps
            observation.update(
                time_context=np.array(
                    [
                        (self.tick - self._frame_tick) / self.fps,
                        elapsed / self.fps,
                        1 / self.fps,
                        0,
                        0,
                    ],
                    np.float32,
                ),
                target_dt=np.arange(1, self.chunk_length + 1, dtype=np.float32)
                / self.fps,
                history_age=ages,
            )
        return observation

    def _deliver_commands(self):
        expired = 0
        while self._commands and self._commands[0][0] <= self.tick:
            _, seq, origin, chunk = heapq.heappop(self._commands)
            for j, report in enumerate(chunk):
                target = origin + j
                if target < self.tick:
                    expired += 1
                    continue
                current = self._schedule.get(target)
                if current is None or current[0] < seq:
                    self._schedule[target] = (seq, report.copy())
        return expired

    def _deliver_frames(self):
        while self._frames and self._frames[0][0] <= self.tick:
            _, captured, frame = heapq.heappop(self._frames)
            if captured > self._frame_tick:
                self._frame_tick, self._board = captured, frame

    def _committed_reports(self, count):
        forecast = {
            tick: (seq, report) for tick, (seq, report) in self._schedule.items()
        }
        for arrival, seq, origin, chunk in self._commands:
            for j, report in enumerate(chunk):
                target = origin + j
                current = forecast.get(target)
                if target >= max(arrival, self.tick) and (
                    current is None or seq > current[0]
                ):
                    forecast[target] = (seq, report)
        return [
            forecast.get(self.tick + j, (-1, gamepad.neutral()))[1].copy()
            for j in range(count)
        ]

    def step(self, action):
        if not self._ready:
            raise ResetNeeded("reset before stepping or after episode end")
        chunk = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(chunk):
            raise ValueError("expected finite [L,20] normalized gamepad chunk")
        self._sequence += 1
        self._origin, self._previous = self.tick, chunk.copy()
        arrival = self.tick + self._draw(self.command_delay)
        heapq.heappush(
            self._commands, (arrival, self._sequence, self.tick, chunk.copy())
        )
        old_frame = self._frame_tick
        rewards, reports, sources = [], [], []
        period_events, tick_scores = [], []
        food_events = []
        expired = 0
        for _ in range(self._draw(self.decision_ticks)):
            expired += self._deliver_commands()
            if hasattr(self.env, "set_committed_reports"):
                self.env.set_committed_reports(
                    self._committed_reports(self.env.protected_ticks)
                )
            seq, report = self._schedule.pop(self.tick, (-1, gamepad.neutral()))
            board, reward, terminated, truncated, info = self.env.step(report)
            period_events.extend(info.get("perturbation_events", []))
            if info.get("food_event") is not None:
                food_events.append(info["food_event"])
            tick_scores.append(info["score"])
            self.tick += 1
            reports.append(self.env.last_report.copy())
            sources.append(seq)
            rewards.append(float(reward))
            if self._rng.random() >= self.drop_probability:
                heapq.heappush(
                    self._frames,
                    (
                        self.tick + self._draw(self.observation_delay),
                        self.tick,
                        board.copy(),
                    ),
                )
            self._deliver_frames()
            if terminated or truncated:
                self._ready = False
                break
        elapsed = len(reports)
        info.update(
            {
                "ticks_advanced": elapsed,
                "capture_tick": self._frame_tick,
                "controller_tick": self.tick,
                "expired_reports": expired,
                "executed_sources": np.array(sources),
                "executed_reports": np.stack(reports).astype(np.float32),
                "submitted_sequence": self._sequence,
                "tick_rewards": rewards,
                "bootstrap_discount": self.gamma**elapsed,
                "undiscounted_reward": sum(rewards),
                "timebase": "simulation_tick",
                "command_origin_tick": self._origin,
                "command_arrival_tick": arrival,
                "execution_ticks": list(range(self.tick - elapsed, self.tick)),
                "tick_scores": tick_scores,
                "perturbation_events": period_events,
                "food_events": food_events,
            }
        )
        reward = sum(self.gamma**j * r for j, r in enumerate(rewards))
        return (
            self._observe(reports, elapsed, self._frame_tick != old_frame),
            reward,
            terminated,
            truncated,
            info,
        )

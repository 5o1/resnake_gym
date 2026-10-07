"""Independent wall-clock game process, not a sleep inside policy.step().

Python multiprocessing isolates the game from Python inference/GIL stalls.
RTC motivates timestamped chunks; this implements a transport/execution clock,
not RTC's learned flow inpainting. No action smoothing or board-aware correction.
"""

import copy
import heapq
import json
import multiprocessing as mp
import queue
import time
import traceback
import uuid
from collections import deque
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from resnake_gym.envs.gamepad_env import GamepadSnakeEnv
from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv
from resnake_gym.protocol import CommandSchedule, GamepadCommand


def policy_observation(
    snapshot,
    *,
    previous_decision_tick=0,
    previous_capture_tick=-1,
    chunk_length=8,
    history_limit=128,
    time_features=False,
    decision_ns=None,
    previous_decision_ns=None,
):
    """Strip diagnostic truth/certificates; expose delivered pixels and HID only.

    Retains the existing recurrent policy dictionary keys. Scene channel count
    is 14 for the disturbance task. Overlong feedback gaps fail explicitly.
    """
    tick = snapshot["controller_tick"]
    elapsed = tick - previous_decision_tick
    if elapsed < 0 or elapsed > history_limit:
        raise ValueError("decision history exceeds feedback capacity or runs backwards")
    feedback = [
        row for row in snapshot["feedback"] if row["tick"] >= previous_decision_tick
    ]
    if [row["tick"] for row in feedback] != list(range(previous_decision_tick, tick)):
        raise ValueError("incomplete actual-input history")
    history = np.zeros((history_limit, 20), np.float32)
    mask = np.zeros(history_limit, np.int8)
    if feedback:
        history[: len(feedback)] = [row["report"] for row in feedback]
        mask[: len(feedback)] = 1
    previous = np.zeros((chunk_length, 20), np.float32)
    previous_age = 0.0
    packet = snapshot["previous_command"]
    if packet is not None:
        old = np.asarray(packet["reports"], np.float32)
        if old.shape != previous.shape:
            raise ValueError("model horizon differs from previous command horizon")
        previous[:] = old
        previous_age = (tick - packet["origin_tick"]) * snapshot["period_ns"] / 1e9
    observation = {
        "board": snapshot["board"].copy(),
        "report": snapshot["actual_report"].copy(),
        "timing": np.array(
            [
                (tick - snapshot["capture_tick"]) * snapshot["period_ns"] / 1e9,
                elapsed * snapshot["period_ns"] / 1e9,
                float(snapshot["capture_tick"] != previous_capture_tick),
            ],
            np.float32,
        ),
        "history": history,
        "history_mask": mask,
        "previous_chunk": previous,
        "previous_age": np.array([previous_age], np.float32),
    }
    if time_features:
        now = time.monotonic_ns() if decision_ns is None else decision_ns
        period = snapshot["period_ns"]
        captured = snapshot["observation_captured_ns"]
        published = snapshot["snapshot_published_ns"]
        applied = (
            snapshot["feedback"][-1]["applied_ns"]
            if snapshot["feedback"]
            else snapshot["epoch_ns"]
        )
        if now < max(captured, published, applied) or (
            previous_decision_ns is not None and now < previous_decision_ns
        ):
            raise ValueError("policy timestamps must share a monotonic clock")
        ages = np.zeros(history_limit, np.float32)
        ages[: len(feedback)] = [(now - row["applied_ns"]) / 1e9 for row in feedback]
        observation.update(
            time_context=np.array(
                [
                    (now - captured) / 1e9,
                    (now - previous_decision_ns) / 1e9
                    if previous_decision_ns is not None
                    else elapsed * period / 1e9,
                    period / 1e9,
                    (now - published) / 1e9,
                    (now - applied) / 1e9,
                ],
                np.float32,
            ),
            target_dt=np.array(
                [
                    (snapshot["epoch_ns"] + (tick + j + 1) * period - now) / 1e9
                    for j in range(chunk_length)
                ],
                np.float32,
            ),
            history_age=ages,
        )
    return observation


@dataclass
class RealTimeConfig:
    env_kind: str = "perturbed"
    env_kwargs: dict = field(default_factory=dict)
    seed: int = 2026
    max_ticks: int = 3000
    feedback_history: int = 128
    observation_delay_ticks: tuple = (0, 0)
    frame_drop_probability: float = 0.0
    trace_path: str | None = None


def _publish_latest(mailbox, snapshot):
    """Bounded nonblocking publication. Consumer slowness never pauses game."""
    try:
        mailbox.put_nowait(snapshot)
        return 0
    except queue.Full:
        try:
            mailbox.get_nowait()
        except queue.Empty:
            return 1  # feeder has not flushed yet; drop, never wait on policy
        with suppress(queue.Full):
            mailbox.put_nowait(snapshot)
        return 1


@dataclass
class _GameLoop:
    """Mutable state owned exclusively by the wall-clock game process."""

    config: RealTimeConfig
    clock_id: str
    env: object
    inbox: object
    snapshots: object
    trace: object | None
    schedule: CommandSchedule
    period_ns: int
    epoch_ns: int
    rng: np.random.Generator
    frames: list
    history: deque
    capture_tick: int
    captured_ns: int
    available_ns: int
    latest_board: np.ndarray
    previous: dict | None
    command_received: dict
    stats: dict
    info: dict
    delay_low: int
    delay_high: int


def _initial_stats():
    return {
        "ticks": 0,
        "expired_reports": 0,
        "rejected_commands": 0,
        "snapshot_drops": 0,
        "late_over_one_period": 0,
        "max_start_lateness_ns": 0,
        "max_step_duration_ns": 0,
        "events_applied": 0,
        "events_rejected": 0,
    }


def _validate_realtime_environment(config, env):
    if env.frame_skip != 1 or env.render_mode == "human":
        raise ValueError(
            "real-time process requires one tick/step and a nonblocking renderer"
        )
    if config.max_ticks < 1 or config.feedback_history < 1:
        raise ValueError("max_ticks and feedback_history must be positive")
    low, high = config.observation_delay_ticks
    if not 0 <= low <= high or not 0 <= config.frame_drop_probability <= 1:
        raise ValueError("invalid frame transport parameters")


def _new_game_loop(config, clock_id, env, inbox, snapshots, trace, board, info):
    period_ns = round(1_000_000_000 / env.logic_fps)
    if period_ns < 1:
        raise ValueError("logic_fps exceeds nanosecond resolution")
    epoch_ns = time.monotonic_ns()
    low, high = config.observation_delay_ticks
    return _GameLoop(
        config=config,
        clock_id=clock_id,
        env=env,
        inbox=inbox,
        snapshots=snapshots,
        trace=trace,
        schedule=CommandSchedule(clock_id),
        period_ns=period_ns,
        epoch_ns=epoch_ns,
        rng=np.random.default_rng(config.seed),
        frames=[],
        history=deque(maxlen=config.feedback_history),
        capture_tick=0,
        captured_ns=epoch_ns,
        available_ns=epoch_ns,
        latest_board=board.copy(),
        previous=None,
        command_received={},
        stats=_initial_stats(),
        info=info,
        delay_low=low,
        delay_high=high,
    )


def _write_trace(trace, row):
    if trace:
        trace.write(json.dumps(row) + "\n")


def _drain_commands(loop, tick):
    """Consume bounded IPC input without letting a sender stall the game."""
    for _ in range(64):
        try:
            payload = loop.inbox.get_nowait()
        except queue.Empty:
            break
        received = time.monotonic_ns()
        try:
            command = GamepadCommand.from_wire(payload)
            if command.submitted_ns > received:
                raise ValueError("future submitted timestamp")
            result = loop.schedule.submit(command, tick)
            if result["accepted"]:
                loop.command_received[command.sequence] = received
                loop.previous = command.to_wire()
                loop.stats["expired_reports"] += result["expired"]
            else:
                loop.stats["rejected_commands"] += 1
            _write_trace(
                loop.trace,
                {
                    "type": "command",
                    "received_ns": received,
                    "packet": payload,
                    "result": result,
                },
            )
        except (ValueError, TypeError, KeyError) as error:
            loop.stats["rejected_commands"] += 1
            _write_trace(
                loop.trace,
                {
                    "type": "command_rejected",
                    "received_ns": received,
                    "reason": str(error),
                },
            )


def _update_execution_stats(loop, applied_ns, state_captured_ns, deadline, events):
    lateness = max(0, applied_ns - deadline)
    loop.stats["max_start_lateness_ns"] = max(
        loop.stats["max_start_lateness_ns"], lateness
    )
    loop.stats["max_step_duration_ns"] = max(
        loop.stats["max_step_duration_ns"], state_captured_ns - applied_ns
    )
    loop.stats["late_over_one_period"] += int(lateness >= loop.period_ns)
    loop.stats["events_applied"] += sum(
        event["status"] == "applied" for event in events
    )
    loop.stats["events_rejected"] += sum(
        event["status"] == "rejected" for event in events
    )


def _executed_tick(
    loop,
    *,
    tick,
    deadline,
    applied_ns,
    state_captured_ns,
    command,
    chunk_index,
    reward,
    events,
):
    return {
        "tick": tick,
        "clock_id": loop.clock_id,
        "scheduled_ns": deadline,
        "applied_ns": applied_ns,
        "state_captured_ns": state_captured_ns,
        "sequence": command.sequence if command else None,
        "chunk_index": chunk_index,
        "report": loop.env.last_report.tolist(),
        "command_received_ns": loop.command_received.get(command.sequence)
        if command
        else None,
        "observation_captured_ns": command.observation_captured_ns if command else None,
        "reward": float(reward),
        "perturbation_events": events,
    }


def _advance_frame_transport(loop, board, state_captured_ns):
    """Apply frame loss/delay using the same draw order as the old loop."""
    if loop.rng.random() >= loop.config.frame_drop_probability:
        arrival = loop.stats["ticks"] + int(
            loop.rng.integers(loop.delay_low, loop.delay_high + 1)
        )
        heapq.heappush(
            loop.frames,
            (arrival, loop.stats["ticks"], state_captured_ns, board.copy()),
        )
    while loop.frames and loop.frames[0][0] <= loop.stats["ticks"]:
        _, captured_tick, captured_ns, frame = heapq.heappop(loop.frames)
        if captured_tick > loop.capture_tick:
            loop.capture_tick = captured_tick
            loop.captured_ns = captured_ns
            loop.latest_board = frame
            loop.available_ns = time.monotonic_ns()


def _forget_inactive_command_receipts(loop):
    active_sequences = {entry[1].sequence for entry in loop.schedule.pending.values()}
    loop.command_received = {
        key: value
        for key, value in loop.command_received.items()
        if key in active_sequences
    }


def _advance_one_tick(loop, tick, deadline):
    """Execute exactly one scheduled game tick and return its terminal state."""
    if hasattr(loop.env, "set_committed_reports"):
        loop.env.set_committed_reports(
            loop.schedule.peek_reports(tick, loop.env.protected_ticks)
        )
    report, command, chunk_index = loop.schedule.take(tick)
    applied_ns = time.monotonic_ns()
    board, reward, terminated, truncated, info = loop.env.step(report)
    state_captured_ns = time.monotonic_ns()
    loop.info = info
    loop.stats["ticks"] += 1
    events = info.get("perturbation_events", [])
    _update_execution_stats(
        loop,
        applied_ns,
        state_captured_ns,
        deadline,
        events,
    )
    executed = _executed_tick(
        loop,
        tick=tick,
        deadline=deadline,
        applied_ns=applied_ns,
        state_captured_ns=state_captured_ns,
        command=command,
        chunk_index=chunk_index,
        reward=reward,
        events=events,
    )
    loop.history.append(executed)
    _write_trace(loop.trace, {"type": "tick", **executed})
    _advance_frame_transport(loop, board, state_captured_ns)
    _forget_inactive_command_receipts(loop)
    return terminated or truncated


def _snapshot(loop, tick, done):
    return {
        "clock_id": loop.clock_id,
        "epoch_ns": loop.epoch_ns,
        "period_ns": loop.period_ns,
        "controller_tick": tick,
        "capture_tick": loop.capture_tick,
        "observation_captured_ns": loop.captured_ns,
        "observation_available_ns": loop.available_ns,
        "snapshot_published_ns": time.monotonic_ns(),
        "board": loop.latest_board.copy(),
        "actual_report": loop.env.last_report.copy(),
        "feedback": list(loop.history),
        "previous_command": loop.previous,
        "done": done,
        "info": loop.info,
        "stats": dict(loop.stats),
    }


def _publish_snapshot(loop, tick, done=False):
    snapshot = _snapshot(loop, tick, done)
    loop.stats["snapshot_drops"] += _publish_latest(loop.snapshots, snapshot)


def _finished_summary(loop):
    return {
        "kind": "finished",
        "clock_id": loop.clock_id,
        "epoch_ns": loop.epoch_ns,
        "finished_ns": time.monotonic_ns(),
        "period_ns": loop.period_ns,
        "stats": loop.stats,
        "info": {
            key: value for key, value in loop.info.items() if key != "gamepad_report"
        },
    }


def _run_game(config, clock_id, inbox, snapshots, status, stop):
    env, trace = None, None
    resources = ExitStack()
    try:
        if config.env_kind not in ("base", "perturbed"):
            raise ValueError("env_kind must be base or perturbed")
        constructor = (
            PerturbedGamepadEnv if config.env_kind == "perturbed" else GamepadSnakeEnv
        )
        env_kwargs = dict(config.env_kwargs)
        if config.env_kind == "perturbed":
            env_kwargs.setdefault("solver_mode", "async")
        env = constructor(**env_kwargs)
        if config.env_kind == "perturbed":
            env.bind_solver_clock(clock_id)
        _validate_realtime_environment(config, env)
        board, info = env.reset(seed=config.seed)
        if config.trace_path:
            trace = resources.enter_context(
                Path(config.trace_path).open("x", buffering=1)  # noqa: SIM115
            )
        loop = _new_game_loop(
            config,
            clock_id,
            env,
            inbox,
            snapshots,
            trace,
            board,
            info,
        )
        status.put(
            {
                "kind": "started",
                "clock_id": clock_id,
                "epoch_ns": loop.epoch_ns,
                "period_ns": loop.period_ns,
            }
        )
        _publish_snapshot(loop, 0)
        done = False
        while loop.stats["ticks"] < config.max_ticks and not stop.is_set() and not done:
            tick = loop.stats["ticks"]
            deadline = loop.epoch_ns + (tick + 1) * loop.period_ns
            if stop.wait(max(0, (deadline - time.monotonic_ns()) / 1e9)):
                break
            _drain_commands(loop, tick)
            done = _advance_one_tick(loop, tick, deadline)
            _publish_snapshot(
                loop,
                loop.stats["ticks"],
                done or loop.stats["ticks"] >= config.max_ticks,
            )
        summary = _finished_summary(loop)
        _write_trace(trace, {"type": "summary", **summary})
        status.put(summary)
    except BaseException:
        status.put({"kind": "error", "traceback": traceback.format_exc()})
    finally:
        resources.close()
        if env:
            env.close()
        # Never cancel a partially transmitted snapshot: Queue.get may then
        # block inside recv_bytes even with a timeout. Parent drains on close.
        snapshots.close()
        snapshots.join_thread()


class RealTimeGame:
    """Single controller client; start/observe/submit/close, all with explicit time."""

    def __init__(self, config=None):
        self.config = config or RealTimeConfig()
        context = mp.get_context("spawn")
        self.clock_id = "monotonic-local:" + uuid.uuid4().hex
        self._inbox = context.Queue(maxsize=64)
        self._snapshots = context.Queue(maxsize=1)
        self._status = context.Queue(maxsize=4)
        self._stop = context.Event()
        self._process = context.Process(
            target=_run_game,
            args=(
                self.config,
                self.clock_id,
                self._inbox,
                self._snapshots,
                self._status,
                self._stop,
            ),
            daemon=False,  # game owns a certificate subprocess in async mode
        )
        self._latest = None
        self._frame_received = None
        self.summary = None
        self._closed = False

    def start(self, timeout=10):
        self._process.start()
        try:
            message = self._status.get(timeout=timeout)
        except queue.Empty as error:
            self.close()
            raise TimeoutError("game process did not start") from error
        if message["kind"] == "error":
            self.close()
            raise RuntimeError(message["traceback"])
        self.start_info = message
        return self

    def observe(self, timeout=1):
        try:
            snapshot = self._snapshots.get(timeout=timeout)
            while True:
                try:
                    snapshot = self._snapshots.get_nowait()
                except queue.Empty:
                    break
            received = time.monotonic_ns()
            if (
                self._frame_received is None
                or snapshot["capture_tick"] != self._frame_received[0]
            ):
                self._frame_received = (snapshot["capture_tick"], received)
            snapshot["observation_received_ns"] = self._frame_received[1]
            snapshot["snapshot_received_ns"] = received
            self._latest = snapshot
            return copy.deepcopy(snapshot)
        except queue.Empty as error:
            self.poll_status()
            if self._latest is not None and self.summary is not None:
                snapshot = copy.deepcopy(self._latest)
                snapshot["done"] = True
                return snapshot
            raise TimeoutError("no new game snapshot") from error

    def submit(self, command):
        if command.clock_id != self.clock_id:
            raise ValueError("clock domain mismatch; calibrate remote devices first")
        if self.summary is not None or not self._process.is_alive():
            raise RuntimeError("game process is not running")
        self._inbox.put_nowait(command.to_wire())

    def poll_status(self):
        while True:
            try:
                message = self._status.get_nowait()
            except queue.Empty:
                break
            if message["kind"] == "error":
                raise RuntimeError(message["traceback"])
            if message["kind"] == "finished":
                self.summary = message
        return self.summary

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._process.pid is not None:
            deadline = time.monotonic() + 5
            while self._process.is_alive() and time.monotonic() < deadline:
                # Drain while the producer finishes flushing its final frame.
                with suppress(queue.Empty):
                    self._snapshots.get(timeout=0.05)
                self._process.join(timeout=0.01)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            self.poll_status()
        for channel in (self._inbox, self._snapshots, self._status):
            channel.cancel_join_thread()
            channel.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

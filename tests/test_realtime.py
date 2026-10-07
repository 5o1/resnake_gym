import json
import time
from contextlib import suppress

import numpy as np

from resnake_gym.protocol import GamepadCommand
from resnake_gym.realtime import RealTimeConfig, RealTimeGame
from resnake_gym.timing_audit import audit_timing


def test_game_advances_during_cpu_bound_inference_and_records_timestamps(tmp_path):
    trace = tmp_path / "timing.jsonl"
    config = RealTimeConfig(
        env_kind="base",
        env_kwargs={"logic_fps": 20},
        max_ticks=12,
        trace_path=str(trace),
    )
    with RealTimeGame(config) as game:
        first = game.observe()
        start = time.monotonic_ns()
        # Deliberately CPU-bound, not sleep: game must not depend on parent GIL.
        end = start + 180_000_000
        while time.monotonic_ns() < end:
            sum(range(100))
        completed = time.monotonic_ns()
        reports = np.zeros((8, 20), np.float32)
        reports[:, 0] = 1
        game.submit(
            GamepadCommand(
                sequence=1,
                origin_tick=first["controller_tick"],
                observation_tick=first["capture_tick"],
                clock_id=game.clock_id,
                observation_captured_ns=first["observation_captured_ns"],
                observation_received_ns=first["observation_received_ns"],
                inference_started_ns=start,
                inference_completed_ns=completed,
                submitted_ns=time.monotonic_ns(),
                reports=reports,
            )
        )
        second = game.observe()
        assert second["controller_tick"] > first["controller_tick"]
        deadline = time.monotonic() + 3
        while game.poll_status() is None and time.monotonic() < deadline:
            with suppress(TimeoutError):
                game.observe(timeout=0.1)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    ticks = [row for row in rows if row["type"] == "tick"]
    commands = [row for row in rows if row["type"] == "command"]
    assert ticks and commands
    assert commands[0]["result"]["expired"] >= 1
    assert any(start < tick["applied_ns"] < completed for tick in ticks)
    for tick in ticks:
        assert tick["scheduled_ns"] <= tick["applied_ns"] <= tick["state_captured_ns"]
        if tick["sequence"] is not None:
            assert tick["chunk_index"] == tick["tick"] - first["controller_tick"]
            assert tick["command_received_ns"] <= tick["applied_ns"]
    assert game.summary["stats"]["ticks"] == len(ticks)
    assert audit_timing(rows)["valid_timeline"]


def test_realtime_async_pipeline_uses_shared_clock_without_waiting(tmp_path):
    trace = tmp_path / "async.jsonl"
    config = RealTimeConfig(
        env_kwargs={
            "logic_fps": 10,
            "perturbation_interval": 1,
            "notice_ticks": 3,
            "obstacle_probability": 0,
            "random_rotation": True,
        },
        max_ticks=6,
        trace_path=str(trace),
    )
    with RealTimeGame(config) as game:
        deadline = time.monotonic() + 5
        while game.poll_status() is None and time.monotonic() < deadline:
            with suppress(TimeoutError):
                game.observe(timeout=0.1)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert game.summary["info"]["solver_mode"] == "async"
    assert game.summary["info"]["solver_clock_id"] == game.clock_id
    events = [
        event
        for row in rows
        if row["type"] == "tick"
        for event in row["perturbation_events"]
    ]
    terminal = [event for event in events if event["status"] != "announced"]
    assert len(terminal) == 1
    # The real-time contract is fail-closed: scheduler or coverage overhead
    # may legitimately make the independent solver miss this short 300 ms
    # window.  Deterministic worker success and activation are tested with
    # bounded waits / a controlled worker in test_async_solvability.py.
    assert terminal[0]["status"] == "applied" or terminal[0]["reason"] in {
        "async_not_ready",
        "async_solver_late",
    }
    assert all(e.get("clock_id", game.clock_id) == game.clock_id for e in events)
    assert audit_timing(rows)["valid_timeline"]


def test_close_flushes_large_unconsumed_snapshot():
    # Far larger than a pipe buffer: cancelling the feeder can strand a reader.
    game = RealTimeGame(
        RealTimeConfig(
            env_kind="base",
            env_kwargs={"width": 101, "height": 100, "logic_fps": 50},
            max_ticks=3,
        )
    )
    game.start()
    time.sleep(0.15)
    game.close()
    assert not game._process.is_alive()
    assert game.summary["stats"]["ticks"] == 3

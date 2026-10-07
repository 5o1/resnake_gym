"""Package-level orchestration for the wall-clock gamepad runner.

The command-line wrapper only parses arguments.  This module owns policy
loading, timestamped decisions, process interaction, and timing artifacts.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from resnake_gym.protocol import GamepadCommand
from resnake_gym.realtime import RealTimeConfig, RealTimeGame, policy_observation
from resnake_gym.timing_audit import audit_timing


@dataclass(frozen=True, slots=True)
class RealtimeRunOptions:
    """Inputs and artifact location for one wall-clock policy run."""

    output: Path
    policy_spec: str | None
    ticks: int = 300
    logic_fps: float = 10.0
    inference_delay_ms: float = 0.0
    seed: int = 520000
    horizon: int = 8
    history_limit: int = 128
    timing_v2: bool = False
    solver_time_ms: float | None = None


@dataclass(slots=True)
class _DecisionCursor:
    tick: int = 0
    capture_tick: int = -1
    sequence: int = 0
    started_ns: int | None = None


def _load_policy(spec: str | None, seed: int) -> Any | None:
    if not spec:
        return None
    module, factory = spec.split(":", 1)
    policy = getattr(importlib.import_module(module), factory)()
    policy.reset(seed=seed)
    return policy


def _realtime_config(options: RealtimeRunOptions, trace: Path) -> RealTimeConfig:
    return RealTimeConfig(
        env_kwargs={
            "logic_fps": options.logic_fps,
            "solver_budget": (
                {"time_limit_ms": options.solver_time_ms}
                if options.solver_time_ms is not None
                else None
            ),
        },
        seed=options.seed,
        max_ticks=options.ticks,
        feedback_history=options.history_limit,
        trace_path=str(trace),
    )


def _next_command(
    game: RealTimeGame,
    snapshot: dict,
    policy: Any | None,
    options: RealtimeRunOptions,
    cursor: _DecisionCursor,
) -> GamepadCommand:
    started_ns = time.monotonic_ns()
    visible = policy_observation(
        snapshot,
        previous_decision_tick=cursor.tick,
        previous_capture_tick=cursor.capture_tick,
        chunk_length=options.horizon,
        history_limit=options.history_limit,
        time_features=options.timing_v2,
        decision_ns=started_ns,
        previous_decision_ns=cursor.started_ns,
    )
    cursor.started_ns = started_ns
    if options.inference_delay_ms:
        time.sleep(options.inference_delay_ms / 1000)
    if policy:
        reports = policy.act(visible)
    else:
        reports = np.zeros((options.horizon, 20), np.float32)
    completed_ns = time.monotonic_ns()
    cursor.tick = snapshot["controller_tick"]
    cursor.capture_tick = snapshot["capture_tick"]
    cursor.sequence += 1
    return GamepadCommand(
        cursor.sequence,
        cursor.tick,
        cursor.capture_tick,
        game.clock_id,
        snapshot["observation_captured_ns"],
        snapshot["observation_received_ns"],
        started_ns,
        completed_ns,
        time.monotonic_ns(),
        reports,
    )


def _drive_game(
    game: RealTimeGame,
    policy: Any | None,
    options: RealtimeRunOptions,
) -> None:
    cursor = _DecisionCursor()
    while game.poll_status() is None:
        try:
            snapshot = game.observe(timeout=1)
        except TimeoutError:
            continue
        if snapshot["done"]:
            break
        packet = _next_command(game, snapshot, policy, options, cursor)
        if game.poll_status() is None:
            try:
                game.submit(packet)
            except RuntimeError:
                if game.poll_status() is None:
                    raise


def _timing_report(
    trace: Path,
    options: RealtimeRunOptions,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    report = audit_timing(rows)
    report.update(
        {
            "policy_source": options.policy_spec or "neutral_fixture",
            "capability_result": False,
            "injected_inference_delay_ms": options.inference_delay_ms,
            "logic_fps_requested": options.logic_fps,
            "solver_time_limit_ms": options.solver_time_ms,
        }
    )
    return report


def run_realtime_policy(options: RealtimeRunOptions) -> dict[str, Any]:
    """Run one policy against the independent clock and write its audit files."""
    if options.horizon < 1 or options.inference_delay_ms < 0:
        raise ValueError("invalid horizon or delay")
    options.output.mkdir(parents=True, exist_ok=False)
    trace = options.output / "timeline.jsonl"
    policy = _load_policy(options.policy_spec, options.seed)
    config = _realtime_config(options, trace)
    with RealTimeGame(config) as game:
        _drive_game(game, policy, options)
    report = _timing_report(trace, options)
    (options.output / "timing-audit.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = ["RealtimeRunOptions", "run_realtime_policy"]

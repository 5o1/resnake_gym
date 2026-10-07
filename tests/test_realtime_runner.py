"""Entrypoint and artifact contracts for the package-level realtime runner."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

from resnake_gym.realtime_runner import RealtimeRunOptions, run_realtime_policy

ROOT = Path(__file__).parents[1]


def test_fixture_runner_preserves_trace_and_audit_artifacts(tmp_path):
    output = tmp_path / "realtime"
    report = run_realtime_policy(
        RealtimeRunOptions(
            output=output,
            policy_spec=None,
            ticks=4,
            logic_fps=40.0,
            seed=90210,
            horizon=3,
        )
    )

    stored = json.loads((output / "timing-audit.json").read_text())
    rows = [
        json.loads(line)
        for line in (output / "timeline.jsonl").read_text().splitlines()
    ]
    ticks = [row for row in rows if row["type"] == "tick"]

    assert stored == report
    assert report["valid_timeline"]
    assert report["policy_source"] == "neutral_fixture"
    assert report["capability_result"] is False
    assert report["logic_fps_requested"] == 40.0
    assert [row["tick"] for row in ticks] == list(range(4))
    assert rows[-1]["type"] == "summary"
    assert rows[-1]["stats"]["ticks"] == 4


def test_command_wrapper_only_maps_cli_to_runner(monkeypatch, tmp_path, capsys):
    namespace = runpy.run_path(ROOT / "scripts" / "run_realtime_gamepad.py")
    captured = {}

    def fake_run(options):
        captured["options"] = options
        return {"valid_timeline": True}

    namespace["main"].__globals__["run_realtime_policy"] = fake_run
    output = tmp_path / "cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_realtime_gamepad.py",
            "--fixture-smoke",
            "--output",
            str(output),
            "--ticks",
            "17",
            "--logic-fps",
            "12.5",
            "--inference-delay-ms",
            "3",
            "--seed",
            "44",
            "--horizon",
            "5",
            "--history-limit",
            "21",
            "--timing-v2",
            "--solver-time-ms",
            "7.5",
        ],
    )

    namespace["main"]()

    assert captured["options"] == RealtimeRunOptions(
        output=output,
        policy_spec=None,
        ticks=17,
        logic_fps=12.5,
        inference_delay_ms=3.0,
        seed=44,
        horizon=5,
        history_limit=21,
        timing_v2=True,
        solver_time_ms=7.5,
    )
    assert json.loads(capsys.readouterr().out) == {"valid_timeline": True}

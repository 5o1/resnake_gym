import copy

import numpy as np
import pytest

from resnake_gym.protocol import GamepadCommand
from resnake_gym.realtime import policy_observation
from resnake_gym.timing_audit import audit_timing


def trace_rows():
    packet = GamepadCommand(
        0, 0, 0, "unit-test", 1, 2, 3, 4, 5, np.zeros((2, 20), np.float32)
    )
    return [
        {
            "type": "command",
            "packet": packet.to_wire(),
            "received_ns": 6,
            "result": {"accepted": True},
        },
        *[
            {
                "type": "tick",
                "tick": i,
                "clock_id": "unit-test",
                "sequence": 0,
                "chunk_index": i,
                "scheduled_ns": (i + 1) * 10,
                "applied_ns": (i + 1) * 10 + 1,
                "state_captured_ns": (i + 1) * 10 + 2,
                "command_received_ns": 6,
                "observation_captured_ns": 1,
                "report": [0.0] * 20,
            }
            for i in range(2)
        ],
        {
            "type": "summary",
            "clock_id": "unit-test",
            "epoch_ns": 0,
            "period_ns": 10,
            "stats": {"ticks": 2},
        },
    ]


def test_audit_measures_actual_age_not_planned_time():
    report = audit_timing(trace_rows())
    assert report["valid_timeline"]
    assert report["observation_to_application_ns"]["max"] == 20
    assert report["inference_duration_ns"]["max"] == 1


def test_soft_buttons_are_thresholded_only_at_execution():
    rows = trace_rows()
    rows[0]["packet"]["reports"][0][0] = 0.3
    assert audit_timing(rows)["valid_timeline"]


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("chunk_index", 1, "late_prefix_was_shifted"),
        ("command_received_ns", 7, "receipt_timestamp_changed"),
        ("clock_id", "foreign", "clock_or_deadline_mismatch"),
        ("report", [1.0] * 20, "executed_report_differs_from_command"),
    ],
)
def test_audit_detects_corruption(key, value, error):
    rows = trace_rows()
    rows[1][key] = value
    report = audit_timing(rows)
    assert not report["valid_timeline"] and error in report["errors"]


def test_policy_observation_filters_truth_and_detects_missing_feedback():
    snapshot = {
        "controller_tick": 2,
        "capture_tick": 1,
        "period_ns": 100_000_000,
        "board": np.zeros((14, 31, 31), np.float32),
        "actual_report": np.zeros(20, np.float32),
        "previous_command": None,
        "feedback": [{"tick": i, "report": [0.0] * 20} for i in range(2)],
        "info": {"private_certificate": "not for policy"},
    }
    visible = policy_observation(snapshot)
    assert "info" not in visible
    np.testing.assert_allclose(visible["timing"], [0.1, 0.2, 1])
    assert visible["history_mask"].sum() == 2
    bad = copy.deepcopy(snapshot)
    bad["feedback"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        policy_observation(bad)

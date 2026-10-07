import numpy as np
import pytest

from resnake_gym import gamepad
from resnake_gym.protocol import CommandSchedule, GamepadCommand


def command(**kwargs):
    values = dict(
        sequence=1,
        origin_tick=0,
        observation_tick=0,
        clock_id="test",
        observation_captured_ns=1,
        observation_received_ns=2,
        inference_started_ns=3,
        inference_completed_ns=4,
        submitted_ns=5,
        reports=np.zeros((8, 20), np.float32),
    )
    values.update(kwargs)
    return GamepadCommand(**values)


def test_layout_is_frozen():
    assert gamepad.LAYOUT == (
        "dpad_up",
        "dpad_down",
        "dpad_left",
        "dpad_right",
        "start",
        "back",
        "left_thumb",
        "right_thumb",
        "lb",
        "rb",
        "a",
        "b",
        "x",
        "y",
        "lx",
        "ly",
        "rx",
        "ry",
        "lt",
        "rt",
    )


def test_wire_roundtrip_and_copy():
    reports = np.zeros((8, 20), np.float32)
    original = command(reports=reports)
    reports[:] = 1
    restored = GamepadCommand.from_wire(original.to_wire())
    assert not restored.reports.any()
    assert not restored.reports.flags.writeable


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": -1},
        {"origin_tick": 1.5},
        {"observation_tick": 1},
        {"clock_id": ""},
        {"inference_completed_ns": 1},
        {"reports": np.full((2, 20), np.nan)},
        {"reports": np.ones((3, 19))},
        {"reports": np.full((2, 20), 2)},
        {"reports": np.empty((0, 20))},
    ],
)
def test_invalid_packets_rejected(changes):
    with pytest.raises(ValueError):
        command(**changes)


def test_late_prefix_and_old_sequence():
    scheduler = CommandSchedule("test")
    assert scheduler.submit(command(), 3)["expired"] == 3
    assert scheduler.take(3)[2] == 3
    assert scheduler.submit(command(sequence=2, origin_tick=3), 3)["accepted"]
    assert not scheduler.submit(command(sequence=1), 4)["accepted"]
    assert scheduler.take(4)[1].sequence == 2
    assert scheduler.take(100)[1] is None
    assert scheduler.pending == {}
    with pytest.raises(ValueError, match="clock"):
        scheduler.submit(command(sequence=3, clock_id="remote"), 100)


def test_future_origin_and_unknown_schema_rejected():
    with pytest.raises(ValueError):
        CommandSchedule("test").submit(command(origin_tick=10), 2)
    payload = command().to_wire()
    payload["layout"] = "invented"
    with pytest.raises(ValueError):
        GamepadCommand.from_wire(payload)

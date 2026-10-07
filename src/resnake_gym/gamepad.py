"""XInput control semantics; normalized tensor layout is versioned by this module.

Positive stick Y means up. Buttons are held states, not turn commands.
Only the execution boundary thresholds soft buttons. No game-state correction.
"""

import numpy as np
from gymnasium import spaces

BUTTONS = (
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
)
AXES = ("lx", "ly", "rx", "ry", "lt", "rt")
LAYOUT = BUTTONS + AXES
LAYOUT_VERSION = "xinput-normalized-v1"
LOW = np.array([0.0] * 14 + [-1.0] * 4 + [0.0] * 2, dtype=np.float32)
HIGH = np.ones(20, dtype=np.float32)
DPAD5_LABELS = ("neutral", "up", "down", "left", "right")


def dpad5_reports():
    """Return the five legal full reports used by structured D-pad policies."""

    reports = np.zeros((len(DPAD5_LABELS), len(LAYOUT)), dtype=np.float32)
    for category, button in enumerate(range(4), start=1):
        reports[category, button] = 1.0
    return reports


def action_space():
    return spaces.Box(LOW.copy(), HIGH.copy(), dtype=np.float32)


def neutral():
    return np.zeros(20, dtype=np.float32)


def validate(report):
    value = np.asarray(report, dtype=np.float32)
    if value.shape != (20,) or not np.isfinite(value).all():
        raise ValueError("gamepad report must contain 20 finite values")
    if np.any(value < LOW) or np.any(value > HIGH):
        raise ValueError("gamepad report outside normalized control ranges")
    return value.copy()


def digital_report(report):
    value = validate(report)
    value[:14] = value[:14] >= 0.5
    return value


def direction_request(report, deadzone=0.2):
    """D-pad takes priority; conflicting opposites cancel, diagonal ties neutral.

    If any D-pad button is held the stick is ignored, including conflicts.
    Otherwise the dominant left-stick axis selects a screen direction.
    """
    up, down, left, right = report[:4]
    if np.any(report[:4]):
        x, y = right - left, up - down
    else:
        x, y = report[14:16]
        if np.hypot(x, y) <= deadzone:
            return None
    if abs(x) == abs(y):
        return None
    if abs(x) > abs(y):
        return 1 if x > 0 else 3
    return 0 if y > 0 else 2

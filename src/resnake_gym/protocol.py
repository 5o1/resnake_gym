"""Versioned execution-boundary contract. Torch tensors stay upstream of this API.

See docs/gamepad-api.html. Times are integer nanoseconds in one explicitly
named monotonic clock domain; ticks describe the game, not policy call count.
"""

from dataclasses import dataclass

import numpy as np

import resnake_gym.gamepad as gamepad

CONTRACT_VERSION = "gamepad-timed-v1"


@dataclass(frozen=True)
class GamepadCommand:
    sequence: int
    origin_tick: int
    observation_tick: int
    clock_id: str
    observation_captured_ns: int
    observation_received_ns: int
    inference_started_ns: int
    inference_completed_ns: int
    submitted_ns: int
    reports: np.ndarray

    def __post_init__(self):
        for name in (
            "sequence",
            "origin_tick",
            "observation_tick",
            "observation_captured_ns",
            "observation_received_ns",
            "inference_started_ns",
            "inference_completed_ns",
            "submitted_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.clock_id, str) or not self.clock_id:
            raise ValueError("clock_id is required")
        if self.observation_tick > self.origin_tick:
            raise ValueError("observation cannot be from a future tick")
        times = (
            self.observation_captured_ns,
            self.observation_received_ns,
            self.inference_started_ns,
            self.inference_completed_ns,
            self.submitted_ns,
        )
        if list(times) != sorted(times):
            raise ValueError("timestamps must be ordered within one clock domain")
        reports = np.asarray(self.reports, dtype=np.float32)
        if reports.ndim != 2 or reports.shape[1] != 20 or not len(reports):
            raise ValueError("reports must have shape [L,20], L >= 1")
        if not np.isfinite(reports).all():
            raise ValueError("reports must be finite")
        if np.any(reports < gamepad.LOW) or np.any(reports > gamepad.HIGH):
            raise ValueError("reports violate normalized control ranges")
        reports = reports.copy()
        reports.setflags(write=False)
        object.__setattr__(self, "reports", reports)

    def to_wire(self):
        return {
            "contract_version": CONTRACT_VERSION,
            "layout": gamepad.LAYOUT_VERSION,
            **{
                key: getattr(self, key)
                for key in self.__dataclass_fields__
                if key != "reports"
            },
            "reports": self.reports.tolist(),
        }

    @classmethod
    def from_wire(cls, payload):
        payload = dict(payload)
        if payload.pop("contract_version", None) != CONTRACT_VERSION:
            raise ValueError("unsupported timing contract")
        if payload.pop("layout", None) != gamepad.LAYOUT_VERSION:
            raise ValueError("unsupported control layout")
        return cls(**payload)


class CommandSchedule:
    """Latest sequence wins per target tick; never move a late prefix forward."""

    def __init__(self, clock_id):
        self.clock_id = clock_id
        self.pending = {}
        self.last_sequence = -1

    def submit(self, command, current_tick):
        if command.clock_id != self.clock_id:
            raise ValueError("uncalibrated clock domain mismatch")
        if command.sequence <= self.last_sequence:
            return {"accepted": False, "reason": "old_sequence", "expired": 0}
        if command.origin_tick > current_tick:
            raise ValueError("origin_tick must describe an existing decision snapshot")
        self.last_sequence = command.sequence
        expired = 0
        for index, report in enumerate(command.reports):
            target = command.origin_tick + index
            if target < current_tick:
                expired += 1
            else:
                self.pending[target] = (report.copy(), command, index)
        return {"accepted": True, "reason": "installed", "expired": expired}

    def take(self, tick):
        for stale in [key for key in self.pending if key < tick]:
            del self.pending[stale]
        return self.pending.pop(tick, (gamepad.neutral(), None, None))

    def peek_reports(self, tick, count):
        return [
            self.pending.get(tick + j, (gamepad.neutral(), None, None))[0].copy()
            for j in range(count)
        ]

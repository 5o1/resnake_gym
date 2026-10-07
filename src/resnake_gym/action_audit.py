"""Receipt-side causal execution metrics for structured gamepad chunks.

The metrics in this module are deliberately local to one environment step.
They do not claim that a late chunk slot was never executed later in the
episode.  Keeping integer numerators and denominators makes run-level
aggregation exact even when collection batches have different sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ActionExecutionAudit:
    action_head: str
    chunk_length: int
    transition_count: int = 0
    slot_executed: np.ndarray = field(init=False)
    slot_opportunities: np.ndarray = field(init=False)
    zero_current_source: int = 0
    j8_unexecuted_chosen_nll: float = 0.0
    j8_total_chosen_nll: float = 0.0
    held_zero_execution_chosen_nll: float = 0.0
    held_total_chosen_nll: float = 0.0
    executed_ticks: int = 0
    executed_neutral_ticks: int = 0
    fallback_ticks: int = 0
    within_transition_switches: int = 0
    within_transition_switch_opportunities: int = 0
    current_source_matches: int = 0
    current_source_comparisons: int = 0

    def __post_init__(self) -> None:
        if self.action_head not in ("dpad5", "held_dpad5"):
            raise ValueError("action audit requires dpad5 or held_dpad5")
        if type(self.chunk_length) is not int or self.chunk_length < 1:
            raise ValueError("chunk_length must be a positive int")
        self.slot_executed = np.zeros(self.chunk_length, dtype=np.int64)
        self.slot_opportunities = np.zeros(self.chunk_length, dtype=np.int64)

    def add_fragment(self, fragment: dict[str, Any]) -> None:
        """Accumulate one validated fragment exactly once at receipt time."""
        if fragment.get("action_head") != self.action_head:
            raise ValueError("fragment action head does not match action audit")
        requested = np.asarray(fragment["requested_reports"])
        if requested.ndim != 3 or requested.shape[1:] != (self.chunk_length, 20):
            raise ValueError("requested report shape does not match action audit")
        length = int(fragment["length"])
        if requested.shape[0] != length:
            raise ValueError("fragment length does not match requested reports")

        actions = np.asarray(fragment["policy_actions"])
        slot_log_probs = np.asarray(fragment["behavior_slot_log_probs"])
        if self.action_head == "dpad5":
            if actions.shape != (length, self.chunk_length):
                raise ValueError("J8 policy action shape is invalid")
            if slot_log_probs.shape != (length, self.chunk_length, 5):
                raise ValueError("J8 behavior distribution shape is invalid")
        else:
            if actions.shape != (length,):
                raise ValueError("H1 policy action shape is invalid")
            if slot_log_probs.shape != (length, 1, 5):
                raise ValueError("H1 behavior distribution shape is invalid")

        neutral = np.zeros(20, dtype=requested.dtype)
        for index in range(length):
            ticks = int(fragment["ticks_advanced"][index])
            origin = int(fragment["command_origin_ticks"][index])
            sequence = int(fragment["submitted_sequences"][index])
            execution_ticks = list(fragment["execution_ticks"][index])
            sources = np.asarray(fragment["executed_sources"][index])
            reports = np.asarray(fragment["executed_reports"][index])
            if not (len(execution_ticks) == len(sources) == len(reports) == ticks):
                raise ValueError("executed audit rows do not match ticks")

            current_slots = np.zeros(self.chunk_length, dtype=np.bool_)
            self.slot_opportunities[: min(ticks, self.chunk_length)] += 1
            self.transition_count += 1
            self.executed_ticks += ticks
            self.within_transition_switch_opportunities += max(ticks - 1, 0)

            for tick_index, (tick, source, report) in enumerate(
                zip(execution_ticks, sources, reports, strict=True)
            ):
                source = int(source)
                if np.array_equal(report, neutral):
                    self.executed_neutral_ticks += 1
                if source == -1:
                    self.fallback_ticks += 1
                if tick_index and not np.array_equal(report, reports[tick_index - 1]):
                    self.within_transition_switches += 1
                if source != sequence:
                    continue
                slot = int(tick) - origin
                if not 0 <= slot < self.chunk_length:
                    raise ValueError("current-source execution has no valid chunk slot")
                current_slots[slot] = True
                self.current_source_comparisons += 1
                self.current_source_matches += int(
                    np.array_equal(report, requested[index, slot])
                )

            self.slot_executed += current_slots
            zero_current = not bool(current_slots.any())
            self.zero_current_source += int(zero_current)
            if self.action_head == "dpad5":
                chosen = actions[index, :, None]
                chosen_nll = -np.take_along_axis(
                    slot_log_probs[index], chosen, axis=-1
                )[:, 0]
                if not np.isfinite(chosen_nll).all() or np.any(chosen_nll < 0):
                    raise ValueError("J8 chosen negative log probability is invalid")
                self.j8_total_chosen_nll += float(chosen_nll.sum())
                self.j8_unexecuted_chosen_nll += float(chosen_nll[~current_slots].sum())
            else:
                chosen_nll = -float(slot_log_probs[index, 0, int(actions[index])])
                if not np.isfinite(chosen_nll) or chosen_nll < 0:
                    raise ValueError("H1 chosen negative log probability is invalid")
                self.held_total_chosen_nll += chosen_nll
                if zero_current:
                    self.held_zero_execution_chosen_nll += chosen_nll

    @staticmethod
    def _rate(
        result: dict[str, int | float], key: str, numerator: int | float, denominator
    ) -> None:
        result[f"{key}_numerator"] = numerator
        result[f"{key}_denominator"] = denominator
        if denominator:
            result[f"{key}_rate"] = numerator / denominator

    @staticmethod
    def _fraction(
        result: dict[str, int | float], key: str, numerator: float, denominator: float
    ) -> None:
        result[f"{key}_numerator"] = numerator
        result[f"{key}_denominator"] = denominator
        if denominator:
            result[f"{key}_fraction"] = numerator / denominator

    def metrics(self, *, prefix: str = "received_action_") -> dict[str, int | float]:
        """Return flat scalars; absent denominators never become fake zero rates."""
        result: dict[str, int | float] = {
            f"{prefix}transition_count": self.transition_count,
        }
        for slot in range(self.chunk_length):
            self._rate(
                result,
                f"{prefix}slot_{slot}_same_transition_execution",
                int(self.slot_executed[slot]),
                self.transition_count,
            )
            self._rate(
                result,
                f"{prefix}slot_{slot}_opportunity_execution",
                int(self.slot_executed[slot]),
                int(self.slot_opportunities[slot]),
            )
        self._rate(
            result,
            f"{prefix}zero_current_source",
            self.zero_current_source,
            self.transition_count,
        )
        self._rate(
            result,
            f"{prefix}executed_neutral_tick",
            self.executed_neutral_ticks,
            self.executed_ticks,
        )
        self._rate(
            result,
            f"{prefix}fallback_tick",
            self.fallback_ticks,
            self.executed_ticks,
        )
        self._rate(
            result,
            f"{prefix}within_transition_tick_switch",
            self.within_transition_switches,
            self.within_transition_switch_opportunities,
        )
        self._rate(
            result,
            f"{prefix}current_source_report_match",
            self.current_source_matches,
            self.current_source_comparisons,
        )
        if self.action_head == "dpad5":
            self._fraction(
                result,
                f"{prefix}j8_unexecuted_chosen_nll",
                self.j8_unexecuted_chosen_nll,
                self.j8_total_chosen_nll,
            )
        else:
            self._fraction(
                result,
                f"{prefix}h1_zero_execution_chosen_nll",
                self.held_zero_execution_chosen_nll,
                self.held_total_chosen_nll,
            )
        return result

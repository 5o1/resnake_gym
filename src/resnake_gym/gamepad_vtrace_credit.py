"""Credit-trace assembly, validation, and FIFO collection."""

from __future__ import annotations

import queue
import uuid
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event
from typing import Any

import numpy as np

from resnake_gym.action_audit import ActionExecutionAudit
from resnake_gym.gamepad_vtrace_contract import (
    CREDIT_ASSEMBLER_FORMAT,
    CREDIT_TRACE_FORMAT,
    VTraceConfig,
)
from resnake_gym.gamepad_vtrace_fragments import (
    _copy_observation,
    _fragment_command_table,
    _validate_execution_causality,
    validate_fragment,
)

_STREAM_STATE_KEYS = {
    "episode_id",
    "next_fragment_sequence",
    "next_decision_start",
    "credit_sequence",
    "pending_transitions",
    "fragments",
    "last_boundary",
}
_STREAM_COUNTER_KEYS = (
    "episode_id",
    "next_fragment_sequence",
    "next_decision_start",
    "credit_sequence",
    "pending_transitions",
)
_BOUNDARY_SUMMARY_KEYS = {
    "run_generation",
    "actor_id",
    "env_id",
    "episode_id",
    "fragment_sequence",
    "decision_end",
    "score_end",
    "bootstrap_observation",
    "controller_tick",
    "capture_tick",
    "submitted_sequence",
    "credit_sequence",
}
_BOUNDARY_COUNTER_KEYS = (
    "run_generation",
    "actor_id",
    "env_id",
    "episode_id",
    "fragment_sequence",
    "decision_end",
    "score_end",
    "controller_tick",
    "capture_tick",
    "submitted_sequence",
    "credit_sequence",
)
_ASSEMBLER_COUNTER_KEYS = (
    "resume_flush_traces",
    "resume_flush_transitions",
    "cumulative_received_fragments",
    "cumulative_received_transitions",
    "cumulative_received_logic_ticks",
)


def _trace_array(trace: dict[str, Any], key: str) -> np.ndarray:
    return np.concatenate([fragment[key] for fragment in trace["fragments"]], axis=0)


def _observations_equal(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(left[key], right[key]) for key in left
    )


def _fragment_payload_nbytes(fragment: dict[str, Any]) -> int:
    arrays = [value for value in fragment.values() if isinstance(value, np.ndarray)]
    arrays.extend(fragment["observations"].values())
    arrays.extend(fragment["burn_observations"].values())
    arrays.extend(fragment["bootstrap_observation"].values())
    return sum(value.nbytes for value in arrays)


def _validate_fragment_chain(
    fragments: list[dict[str, Any]], config: VTraceConfig
) -> None:
    if not fragments:
        raise ValueError("a credit trace cannot be empty")
    identity = tuple(
        fragments[0][key]
        for key in ("run_generation", "actor_id", "env_id", "episode_id")
    )
    for index, fragment in enumerate(fragments):
        validate_fragment(fragment, config)
        current_identity = tuple(
            fragment[key]
            for key in ("run_generation", "actor_id", "env_id", "episode_id")
        )
        if current_identity != identity:
            raise ValueError("credit trace fragments cross a stream or episode")
        if index == 0:
            continue
        previous = fragments[index - 1]
        if fragment["fragment_sequence"] != previous["fragment_sequence"] + 1:
            raise ValueError("credit trace fragment sequence is not contiguous")
        if fragment["decision_start"] != previous["decision_end"]:
            raise ValueError("credit trace decision range is not contiguous")
        if previous["score_end"] != fragment["score_start"]:
            raise ValueError("credit trace score is not contiguous")
        if previous["contains_food_event"] or bool(
            previous["terminated"][-1] or previous["truncated"][-1]
        ):
            raise ValueError("credit trace continues beyond a semantic boundary")
        first_observation = {
            key: values[0] for key, values in fragment["observations"].items()
        }
        if not _observations_equal(
            previous["bootstrap_observation"], first_observation
        ):
            raise ValueError("credit trace observation boundary is discontinuous")
        previous_tick = int(previous["controller_ticks"][-1])
        if int(fragment["execution_ticks"][0][0]) != previous_tick:
            raise ValueError("credit trace execution ticks are discontinuous")
        if int(fragment["command_origin_ticks"][0]) != previous_tick:
            raise ValueError("credit trace command origin is discontinuous")
        if (
            int(fragment["submitted_sequences"][0])
            != int(previous["submitted_sequences"][-1]) + 1
        ):
            raise ValueError("credit trace submitted sequence is discontinuous")
        if int(fragment["capture_ticks"][0]) < int(previous["capture_ticks"][-1]):
            raise ValueError("credit trace capture ticks are discontinuous")
    known_commands = _fragment_command_table(fragments)
    for fragment in fragments:
        _validate_execution_causality(fragment, config, known_commands)


def _fragment_boundary_summary(
    fragment: dict[str, Any], *, credit_sequence: int
) -> dict[str, Any]:
    """Keep only the state required to audit the next credit-trace boundary."""
    return {
        "run_generation": fragment["run_generation"],
        "actor_id": fragment["actor_id"],
        "env_id": fragment["env_id"],
        "episode_id": fragment["episode_id"],
        "fragment_sequence": fragment["fragment_sequence"],
        "decision_end": fragment["decision_end"],
        "score_end": fragment["score_end"],
        "bootstrap_observation": _copy_observation(fragment["bootstrap_observation"]),
        "controller_tick": int(fragment["controller_ticks"][-1]),
        "capture_tick": int(fragment["capture_ticks"][-1]),
        "submitted_sequence": int(fragment["submitted_sequences"][-1]),
        "credit_sequence": credit_sequence,
    }


def _validate_boundary_summary(
    summary: dict[str, Any], fragment: dict[str, Any]
) -> None:
    """Reject a discontinuity even when it falls between two credit traces."""
    if fragment["episode_id"] != summary["episode_id"]:
        raise ValueError("credit stream episode is discontinuous")
    if fragment["fragment_sequence"] != summary["fragment_sequence"] + 1:
        raise ValueError(
            "credit stream fragment is missing, duplicated, or out of order"
        )
    if fragment["decision_start"] != summary["decision_end"]:
        raise ValueError("credit stream decision range is missing or duplicated")
    if fragment["score_start"] != summary["score_end"]:
        raise ValueError("credit stream score is discontinuous across traces")
    first_observation = {
        key: values[0] for key, values in fragment["observations"].items()
    }
    if not _observations_equal(summary["bootstrap_observation"], first_observation):
        raise ValueError("credit stream observation is discontinuous across traces")
    if int(fragment["execution_ticks"][0][0]) != summary["controller_tick"]:
        raise ValueError("credit stream execution tick is discontinuous across traces")
    if int(fragment["command_origin_ticks"][0]) != summary["controller_tick"]:
        raise ValueError("credit stream command origin is discontinuous across traces")
    if int(fragment["capture_ticks"][0]) < summary["capture_tick"]:
        raise ValueError("credit stream capture tick is discontinuous across traces")
    if int(fragment["submitted_sequences"][0]) != summary["submitted_sequence"] + 1:
        raise ValueError(
            "credit stream submitted sequence is discontinuous across traces"
        )


def _make_credit_trace(
    fragments: list[dict[str, Any]],
    config: VTraceConfig,
    *,
    credit_sequence: int,
    boundary: str,
) -> dict[str, Any]:
    _validate_fragment_chain(fragments, config)
    if type(credit_sequence) is not int or credit_sequence < 0:
        raise ValueError("credit trace sequence is invalid")
    if boundary not in ("food", "terminated", "truncated", "safety_cap", "resume"):
        raise ValueError("unknown credit trace boundary")
    length = sum(fragment["length"] for fragment in fragments)
    if length > config.credit_trace_max_transitions:
        raise ValueError("credit trace exceeds its transition safety bound")
    final = fragments[-1]
    ended = bool(final["terminated"][-1] or final["truncated"][-1])
    if boundary == "food" and (not final["contains_food_event"] or ended):
        raise ValueError("food credit trace has the wrong final boundary")
    if boundary == "terminated" and not bool(final["terminated"][-1]):
        raise ValueError("terminated credit trace has no terminal transition")
    if boundary == "truncated" and not bool(final["truncated"][-1]):
        raise ValueError("truncated credit trace has no truncation transition")
    if boundary in ("safety_cap", "resume") and (final["contains_food_event"] or ended):
        raise ValueError("artificial credit boundary crosses a semantic boundary")
    if boundary == "safety_cap" and length + config.unroll_length <= (
        config.credit_trace_max_transitions
    ):
        raise ValueError("credit safety boundary was applied prematurely")
    return {
        "kind": "credit_trace",
        "format": CREDIT_TRACE_FORMAT,
        "credit_trace_id": uuid.uuid4().hex,
        "credit_sequence": credit_sequence,
        "credit_boundary": boundary,
        "run_generation": final["run_generation"],
        "actor_id": final["actor_id"],
        "env_id": final["env_id"],
        "episode_id": final["episode_id"],
        "length": length,
        "segment_count": len(fragments),
        "score_start": fragments[0]["score_start"],
        "score_end": final["score_end"],
        "food_count": sum(fragment["food_count"] for fragment in fragments),
        "contains_food_event": any(
            fragment["contains_food_event"] for fragment in fragments
        ),
        "won": final["won"],
        "termination_reason": final["termination_reason"],
        "fragments": list(fragments),
    }


def validate_credit_trace(trace: dict[str, Any], config: VTraceConfig) -> None:
    if trace.get("kind") != "credit_trace":
        raise ValueError("expected a completed credit trace")
    if trace.get("format") != CREDIT_TRACE_FORMAT:
        raise ValueError("credit trace format mismatch")
    if (
        not isinstance(trace.get("credit_trace_id"), str)
        or not trace["credit_trace_id"]
    ):
        raise ValueError("credit trace id is invalid")
    fragments = trace.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("credit trace fragments are invalid")
    expected = _make_credit_trace(
        fragments,
        config,
        credit_sequence=trace.get("credit_sequence"),
        boundary=trace.get("credit_boundary"),
    )
    for key in (
        "credit_sequence",
        "credit_boundary",
        "run_generation",
        "actor_id",
        "env_id",
        "episode_id",
        "length",
        "segment_count",
        "score_start",
        "score_end",
        "food_count",
        "contains_food_event",
        "won",
        "termination_reason",
    ):
        if trace.get(key) != expected[key]:
            raise ValueError(f"credit trace {key} metadata mismatch")


def _validate_nonnegative_counters(
    values: dict[str, Any], names: tuple[str, ...], message: str
) -> None:
    if any(type(values.get(name)) is not int or values[name] < 0 for name in names):
        raise ValueError(message)


def _validate_boundary_checkpoint(
    key: tuple[int, int, int],
    state: dict[str, Any],
    fragments: list[dict[str, Any]],
) -> None:
    boundary = state["last_boundary"]
    if boundary is None:
        if state["credit_sequence"] != 0:
            raise ValueError("credit assembler has credit history without a boundary")
        if fragments:
            first = fragments[0]
            if (
                first["fragment_sequence"] != 0
                or first["decision_start"] != 0
                or first["score_start"] != 0
            ):
                raise ValueError("credit assembler pending episode prefix is missing")
        elif state["next_fragment_sequence"] != 0 or state["next_decision_start"] != 0:
            raise ValueError(
                "credit assembler empty episode prefix counters are invalid"
            )
        return
    if not isinstance(boundary, dict):
        raise ValueError("credit assembler boundary summary is invalid")
    if set(boundary) != _BOUNDARY_SUMMARY_KEYS:
        raise ValueError("credit assembler boundary summary schema mismatch")
    _validate_nonnegative_counters(
        boundary,
        _BOUNDARY_COUNTER_KEYS,
        "credit assembler boundary counters are invalid",
    )
    if (
        boundary["capture_tick"] > boundary["controller_tick"]
        or boundary["submitted_sequence"] != boundary["decision_end"]
    ):
        raise ValueError("credit assembler boundary timeline is invalid")
    if (
        boundary["run_generation"],
        boundary["actor_id"],
        boundary["env_id"],
    ) != key:
        raise ValueError("credit assembler stream key does not match boundary")
    if boundary["episode_id"] != state["episode_id"]:
        raise ValueError("credit assembler boundary episode mismatch")
    if state["credit_sequence"] != boundary["credit_sequence"] + 1:
        raise ValueError("credit assembler credit sequence is discontinuous")
    if fragments:
        _validate_boundary_summary(boundary, fragments[0])
        return
    if state["next_fragment_sequence"] != boundary["fragment_sequence"] + 1:
        raise ValueError("credit assembler boundary fragment mismatch")
    if state["next_decision_start"] != boundary["decision_end"]:
        raise ValueError("credit assembler boundary decision mismatch")


def _validate_stream_checkpoint(key: Any, state: Any, config: VTraceConfig) -> None:
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError("credit assembler stream key is invalid")
    if not isinstance(state, dict) or set(state) != _STREAM_STATE_KEYS:
        raise ValueError("credit assembler stream state schema mismatch")
    _validate_nonnegative_counters(
        state,
        _STREAM_COUNTER_KEYS,
        "credit assembler stream counters are invalid",
    )
    fragments = state["fragments"]
    if not isinstance(fragments, list):
        raise ValueError("credit assembler pending fragments are invalid")
    if fragments:
        _validate_fragment_chain(fragments, config)
        if CreditTraceAssembler._stream_key(fragments[0]) != key:
            raise ValueError("credit assembler stream key does not match fragments")
        if sum(item["length"] for item in fragments) != state["pending_transitions"]:
            raise ValueError("credit assembler pending length mismatch")
        final = fragments[-1]
        if state["episode_id"] != final["episode_id"]:
            raise ValueError("credit assembler pending episode mismatch")
        if state["next_fragment_sequence"] != final["fragment_sequence"] + 1:
            raise ValueError("credit assembler next fragment mismatch")
        if state["next_decision_start"] != final["decision_end"]:
            raise ValueError("credit assembler next decision mismatch")
    elif state["pending_transitions"] != 0:
        raise ValueError("empty credit stream has pending transitions")
    _validate_boundary_checkpoint(key, state, fragments)


def _checkpoint_counters(payload: dict[str, Any]) -> dict[str, int]:
    if any(type(payload.get(key)) is not int for key in _ASSEMBLER_COUNTER_KEYS):
        raise ValueError("credit assembler checkpoint counters are missing or invalid")
    counters = {key: payload[key] for key in _ASSEMBLER_COUNTER_KEYS}
    if (
        min(
            counters["cumulative_received_fragments"],
            counters["cumulative_received_transitions"],
            counters["cumulative_received_logic_ticks"],
        )
        < 0
    ):
        raise ValueError("credit assembler cumulative counters are invalid")
    return counters


def _validate_retained_accounting(
    streams: dict[tuple[int, int, int], dict[str, Any]],
    ready: list[dict[str, Any]],
    counters: dict[str, int],
) -> None:
    pending_fragments = sum(len(state["fragments"]) for state in streams.values())
    pending_transitions = sum(
        state["pending_transitions"] for state in streams.values()
    )
    pending_logic_ticks = sum(
        int(fragment["ticks_advanced"].sum())
        for state in streams.values()
        for fragment in state["fragments"]
    )
    ready_fragments = sum(trace["segment_count"] for trace in ready)
    ready_transitions = sum(trace["length"] for trace in ready)
    ready_logic_ticks = sum(
        int(_trace_array(trace, "ticks_advanced").sum()) for trace in ready
    )
    if (
        counters["cumulative_received_fragments"] < pending_fragments + ready_fragments
        or counters["cumulative_received_transitions"]
        < pending_transitions + ready_transitions
        or counters["cumulative_received_logic_ticks"]
        < pending_logic_ticks + ready_logic_ticks
    ):
        raise ValueError("credit assembler cumulative counters are below retained data")


class CreditTraceAssembler:
    """Stitch bounded actor payloads into event/episode credit sequences.

    The 2048-transition default is a host-memory safety boundary.  It is not
    the recurrent gradient window: targets span the completed trace, while
    autograd is separately limited by ``bptt_window``.
    """

    def __init__(self, config: VTraceConfig):
        self.config = config
        self._streams: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._ready: deque[dict[str, Any]] = deque()
        self.resume_flush_traces = 0
        self.resume_flush_transitions = 0
        self.cumulative_received_fragments = 0
        self.cumulative_received_transitions = 0
        self.cumulative_received_logic_ticks = 0

    @staticmethod
    def _stream_key(fragment: dict[str, Any]) -> tuple[int, int, int]:
        return (
            fragment["run_generation"],
            fragment["actor_id"],
            fragment["env_id"],
        )

    def _finish(self, state: dict[str, Any], boundary: str) -> dict[str, Any]:
        trace = _make_credit_trace(
            state["fragments"],
            self.config,
            credit_sequence=state["credit_sequence"],
            boundary=boundary,
        )
        state["last_boundary"] = _fragment_boundary_summary(
            state["fragments"][-1], credit_sequence=state["credit_sequence"]
        )
        state["fragments"] = []
        state["pending_transitions"] = 0
        state["credit_sequence"] += 1
        return trace

    def append(self, fragment: dict[str, Any]) -> list[dict[str, Any]]:
        validate_fragment(fragment, self.config)
        key = self._stream_key(fragment)
        state = self._streams.get(key)
        if state is None:
            if (
                fragment["episode_id"] != 0
                or fragment["fragment_sequence"] != 0
                or fragment["decision_start"] != 0
            ):
                raise ValueError("credit stream starts after a missing fragment")
            state = {
                "episode_id": fragment["episode_id"],
                "next_fragment_sequence": 0,
                "next_decision_start": 0,
                "credit_sequence": 0,
                "pending_transitions": 0,
                "fragments": [],
                "last_boundary": None,
            }
            self._streams[key] = state
        if fragment["episode_id"] != state["episode_id"]:
            raise ValueError(
                "credit stream changed episode without a terminal boundary"
            )
        if fragment["fragment_sequence"] != state["next_fragment_sequence"]:
            raise ValueError(
                "credit stream fragment is missing, duplicated, or out of order"
            )
        if fragment["decision_start"] != state["next_decision_start"]:
            raise ValueError("credit stream decision range is missing or duplicated")
        if state["fragments"]:
            _validate_fragment_chain([state["fragments"][-1], fragment], self.config)
        elif state["last_boundary"] is not None:
            _validate_boundary_summary(state["last_boundary"], fragment)

        self.cumulative_received_fragments += 1
        self.cumulative_received_transitions += fragment["length"]
        self.cumulative_received_logic_ticks += int(fragment["ticks_advanced"].sum())

        completed = []
        if state["fragments"] and (
            state["pending_transitions"] + fragment["length"]
            > self.config.credit_trace_max_transitions
        ):
            completed.append(self._finish(state, "safety_cap"))

        state["fragments"].append(fragment)
        state["pending_transitions"] += fragment["length"]
        state["next_fragment_sequence"] += 1
        state["next_decision_start"] = fragment["decision_end"]

        terminal = bool(fragment["terminated"][-1])
        truncated = bool(fragment["truncated"][-1])
        if terminal:
            boundary = "terminated"
        elif truncated:
            boundary = "truncated"
        elif fragment["contains_food_event"]:
            boundary = "food"
        elif state["pending_transitions"] >= self.config.credit_trace_max_transitions:
            boundary = "safety_cap"
        else:
            boundary = None
        if boundary is not None:
            completed.append(self._finish(state, boundary))
        if terminal or truncated:
            state["episode_id"] += 1
            state["next_fragment_sequence"] = 0
            state["next_decision_start"] = 0
            state["credit_sequence"] = 0
            # A real environment reset is the only boundary at which game
            # state, timestamps and submitted sequence may be discontinuous.
            state["last_boundary"] = None
        return completed

    def pop_ready(self) -> dict[str, Any] | None:
        return self._ready.popleft() if self._ready else None

    def retain_ready(self, traces: list[dict[str, Any]]) -> None:
        """Keep completed traces checkpointable until the learner consumes them."""

        self._ready.extend(traces)

    def flush_for_new_generation(self) -> None:
        """Turn crash/resume orphans into explicit bootstrapped traces."""
        for state in self._streams.values():
            if not state["fragments"]:
                continue
            trace = self._finish(state, "resume")
            self._ready.append(trace)
            self.resume_flush_traces += 1
            self.resume_flush_transitions += trace["length"]
        self._streams.clear()

    @property
    def pending_fragments(self) -> int:
        return sum(len(state["fragments"]) for state in self._streams.values())

    @property
    def pending_transitions(self) -> int:
        return sum(state["pending_transitions"] for state in self._streams.values())

    @property
    def pending_logic_ticks(self) -> int:
        return sum(
            int(fragment["ticks_advanced"].sum())
            for state in self._streams.values()
            for fragment in state["fragments"]
        )

    @property
    def pending_bytes(self) -> int:
        return sum(
            _fragment_payload_nbytes(fragment)
            for state in self._streams.values()
            for fragment in state["fragments"]
        )

    @property
    def ready_transitions(self) -> int:
        return sum(trace["length"] for trace in self._ready)

    @property
    def ready_logic_ticks(self) -> int:
        return sum(
            int(_trace_array(trace, "ticks_advanced").sum()) for trace in self._ready
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": CREDIT_ASSEMBLER_FORMAT,
            "credit_trace_max_transitions": self.config.credit_trace_max_transitions,
            "streams": self._streams,
            "ready": list(self._ready),
            "resume_flush_traces": self.resume_flush_traces,
            "resume_flush_transitions": self.resume_flush_transitions,
            "cumulative_received_fragments": self.cumulative_received_fragments,
            "cumulative_received_transitions": self.cumulative_received_transitions,
            "cumulative_received_logic_ticks": self.cumulative_received_logic_ticks,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != CREDIT_ASSEMBLER_FORMAT:
            raise ValueError("credit assembler checkpoint format mismatch")
        if payload.get("credit_trace_max_transitions") != (
            self.config.credit_trace_max_transitions
        ):
            raise ValueError("credit assembler safety bound mismatch")
        streams = payload.get("streams")
        ready = payload.get("ready")
        if not isinstance(streams, dict) or not isinstance(ready, list):
            raise ValueError("credit assembler checkpoint payload is invalid")
        for key, state in streams.items():
            _validate_stream_checkpoint(key, state, self.config)
        for trace in ready:
            validate_credit_trace(trace, self.config)
        counters = _checkpoint_counters(payload)
        _validate_retained_accounting(streams, ready, counters)
        # Commit only after the entire payload has been validated.  A corrupt
        # sidecar must not leave a partially restored live assembler behind.
        self._streams = streams
        self._ready = deque(ready)
        for key, value in counters.items():
            setattr(self, key, value)


def _food_receipt_tick_positions(
    fragment: dict[str, Any], receipt_tick_start: int
) -> list[int]:
    """Map episode-local food timestamps onto the global receipt-tick axis.

    Queue order defines the primary receipt axis.  Positions are one-based: an
    event on the first logic tick after ``receipt_tick_start`` has position
    ``receipt_tick_start + 1``.  Keeping these exact positions lets an offline
    analysis split a run at 250k ticks without assigning an entire collection
    batch to one side of the boundary.
    """
    positions = []
    transition_offsets = np.concatenate(
        (np.asarray([0]), np.cumsum(fragment["ticks_advanced"], dtype=np.int64))
    )
    for event, transition_index in zip(
        fragment["food_events"],
        fragment["food_event_transition_indices"],
        strict=True,
    ):
        try:
            episode_execution_tick = int(event["tick"]) - 1
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("food event has no valid episode tick") from error
        execution_ticks = fragment["execution_ticks"][int(transition_index)]
        try:
            within_transition = execution_ticks.index(episode_execution_tick)
        except ValueError as error:
            raise ValueError("food event tick is outside its transition") from error
        positions.append(
            int(
                receipt_tick_start
                + transition_offsets[int(transition_index)]
                + within_transition
                + 1
            )
        )
    return positions


def _raise_if_actor_exited(
    actor_processes: list[Any] | None, stop_event: Event
) -> None:
    if actor_processes is None:
        return
    exited = [
        (getattr(actor, "name", repr(actor)), actor.exitcode)
        for actor in actor_processes
        if getattr(actor, "exitcode", None) is not None
    ]
    if exited:
        stop_event.set()
        raise RuntimeError(
            "actor process exited without a queue error: " + repr(exited)
        ) from None


def _trace_logic_ticks(trace: dict[str, Any]) -> int:
    return sum(int(fragment["ticks_advanced"].sum()) for fragment in trace["fragments"])


def _received_episode(
    fragment: dict[str, Any], receipt_tick_start: int
) -> dict[str, Any]:
    controller_tick = int(fragment["controller_ticks"][-1])
    return {
        "actor_id": fragment["actor_id"],
        "run_generation": fragment["run_generation"],
        "env_id": fragment["env_id"],
        "episode_id": fragment["episode_id"],
        "score": fragment["score_end"],
        "won": fragment["won"],
        "termination": fragment["termination_reason"],
        "terminated": bool(fragment["terminated"][-1]),
        "truncated": bool(fragment["truncated"][-1]),
        "ticks": controller_tick,
        "controller_tick": controller_tick,
        "receipt_tick_position": int(
            receipt_tick_start + fragment["ticks_advanced"].sum()
        ),
    }


def _trained_episode(trace: dict[str, Any]) -> dict[str, Any]:
    final = trace["fragments"][-1]
    controller_tick = int(final["controller_ticks"][-1])
    return {
        "actor_id": trace["actor_id"],
        "run_generation": trace["run_generation"],
        "env_id": trace["env_id"],
        "episode_id": trace["episode_id"],
        "score": trace["score_end"],
        "won": trace["won"],
        "termination": trace["termination_reason"],
        "terminated": bool(final["terminated"][-1]),
        "truncated": bool(final["truncated"][-1]),
        "ticks": controller_tick,
        "controller_tick": controller_tick,
    }


@dataclass
class _FreshCollection:
    """Receipt and training populations accumulated for one learner batch."""

    config: VTraceConfig
    traces: list[dict[str, Any]] = field(default_factory=list)
    transitions: int = 0
    foods: int = 0
    logic_ticks: int = 0
    received_fragments: int = 0
    received_transitions: int = 0
    received_logic_ticks: int = 0
    received_foods: int = 0
    received_food_tick_positions: list[int] = field(default_factory=list)
    event_traces: int = 0
    safety_traces: int = 0
    resume_traces: int = 0
    minimum_traces: int | None = None
    minimum_transitions: int | None = None
    minimum_logic_ticks: int | None = None
    quota_required_after_minimum: bool = False
    trained_episodes: list[dict[str, Any]] = field(default_factory=list)
    received_episodes: list[dict[str, Any]] = field(default_factory=list)
    received_action_audit: ActionExecutionAudit = field(init=False)

    def __post_init__(self) -> None:
        self.received_action_audit = ActionExecutionAudit(
            self.config.action_head,
            self.config.chunk_length,
        )

    def needs_more(self) -> bool:
        return self.transitions < self.config.batch_min_transitions or (
            self.foods < self.config.batch_food_target
            and self.transitions < self.config.batch_max_transitions
        )

    def observe_received_transport(self, fragment: dict[str, Any]) -> None:
        self.received_action_audit.add_fragment(fragment)
        self.received_fragments += 1
        self.received_transitions += fragment["length"]
        self.received_logic_ticks += int(fragment["ticks_advanced"].sum())

    def observe_received_outcomes(
        self, fragment: dict[str, Any], receipt_tick_start: int
    ) -> None:
        self.received_foods += int(fragment["food_count"])
        self.received_food_tick_positions.extend(
            _food_receipt_tick_positions(fragment, receipt_tick_start)
        )
        if bool(fragment["terminated"][-1] or fragment["truncated"][-1]):
            self.received_episodes.append(
                _received_episode(fragment, receipt_tick_start)
            )

    def add_trace(self, trace: dict[str, Any]) -> None:
        validate_credit_trace(trace, self.config)
        self.traces.append(trace)
        self.transitions += trace["length"]
        self.foods += trace["food_count"]
        self.logic_ticks += _trace_logic_ticks(trace)
        self.event_traces += int(trace["contains_food_event"])
        self.safety_traces += int(trace["credit_boundary"] == "safety_cap")
        self.resume_traces += int(trace["credit_boundary"] == "resume")
        if (
            self.minimum_transitions is None
            and self.transitions >= self.config.batch_min_transitions
        ):
            self.minimum_traces = len(self.traces)
            self.minimum_transitions = self.transitions
            self.minimum_logic_ticks = self.logic_ticks
            self.quota_required_after_minimum = (
                self.config.batch_food_target > 0
                and self.foods < self.config.batch_food_target
            )
        if trace["credit_boundary"] in ("terminated", "truncated"):
            self.trained_episodes.append(_trained_episode(trace))

    def metrics(self, assembler: CreditTraceAssembler) -> dict[str, Any]:
        assert self.minimum_traces is not None
        assert self.minimum_transitions is not None
        assert self.minimum_logic_ticks is not None
        target_met = self.foods >= self.config.batch_food_target
        quota_wait_traces = len(self.traces) - self.minimum_traces
        quota_wait_transitions = self.transitions - self.minimum_transitions
        quota_wait_logic_ticks = self.logic_ticks - self.minimum_logic_ticks
        return {
            **self.received_action_audit.metrics(),
            "fresh_credit_traces": len(self.traces),
            "fresh_fragments": sum(trace["segment_count"] for trace in self.traces),
            "fresh_transitions": self.transitions,
            "fresh_logic_ticks": self.logic_ticks,
            "fresh_received_fragments": self.received_fragments,
            "fresh_received_transitions": self.received_transitions,
            "fresh_received_logic_ticks": self.received_logic_ticks,
            # Compatibility alias for the learner population. Formal progress
            # curves use the explicitly named receipt-side numerator.
            "fresh_food_count": self.foods,
            "fresh_trained_food_count": self.foods,
            "fresh_received_food_count": self.received_foods,
            "fresh_received_food_tick_positions": self.received_food_tick_positions,
            "fresh_event_trace_count": self.event_traces,
            "fresh_non_event_trace_count": len(self.traces) - self.event_traces,
            "fresh_safety_trace_count": self.safety_traces,
            "fresh_resume_trace_count": self.resume_traces,
            "fresh_food_target": self.config.batch_food_target,
            "fresh_food_target_met": target_met,
            "fresh_food_quota_required_after_minimum": (
                self.quota_required_after_minimum
            ),
            "fresh_food_quota_binding": self.quota_required_after_minimum,
            "fresh_food_quota_added_data": quota_wait_traces > 0,
            "fresh_food_quota_wait_traces": quota_wait_traces,
            "fresh_food_quota_wait_transitions": quota_wait_transitions,
            "fresh_food_quota_wait_logic_ticks": quota_wait_logic_ticks,
            "fresh_hard_limit_reached": (
                self.transitions >= self.config.batch_max_transitions and not target_met
            ),
            "pending_credit_fragments": assembler.pending_fragments,
            "pending_credit_transitions": assembler.pending_transitions,
            "pending_credit_logic_ticks": assembler.pending_logic_ticks,
            "pending_credit_bytes": assembler.pending_bytes,
            "ready_credit_transitions": assembler.ready_transitions,
            "ready_credit_logic_ticks": assembler.ready_logic_ticks,
            "cumulative_received_fragments": (assembler.cumulative_received_fragments),
            "cumulative_received_transitions": (
                assembler.cumulative_received_transitions
            ),
            "cumulative_received_logic_ticks": (
                assembler.cumulative_received_logic_ticks
            ),
            "cumulative_resume_flush_traces": assembler.resume_flush_traces,
            "cumulative_resume_flush_transitions": (assembler.resume_flush_transitions),
            "fresh_episodes": self.trained_episodes,
            "fresh_trained_episodes": self.trained_episodes,
            "fresh_received_episodes": self.received_episodes,
        }


def _wait_for_actor_fragment(
    output_queue,
    stop_event: Event,
    actor_processes: list[Any] | None,
) -> dict[str, Any]:
    while True:
        _raise_if_actor_exited(actor_processes, stop_event)
        try:
            payload = output_queue.get(timeout=5)
        except queue.Empty:
            if stop_event.is_set():
                raise RuntimeError("actors stopped while the learner waited") from None
            continue
        if payload.get("kind") == "actor_error":
            stop_event.set()
            raise RuntimeError(
                f"actor {payload['actor_id']} failed:\n{payload['traceback']}"
            )
        return payload


def _receive_fragment(
    payload: dict[str, Any],
    collection: _FreshCollection,
    assembler: CreditTraceAssembler,
    config: VTraceConfig,
) -> None:
    validate_fragment(payload, config)
    collection.observe_received_transport(payload)
    receipt_tick_start = assembler.cumulative_received_logic_ticks
    completed = assembler.append(payload)
    # Transfer ownership to checkpointable state before inspecting actor exit
    # status or receipt metadata; a process may die immediately after queue.get.
    assembler.retain_ready(completed)
    collection.observe_received_outcomes(payload, receipt_tick_start)


def collect_fresh_credit_traces(
    output_queue,
    stop_event: Event,
    config: VTraceConfig,
    assembler: CreditTraceAssembler,
    *,
    actor_processes: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Consume FIFO fragments and return only completed credit traces.

    Every valid fragment is either returned inside a completed trace or remains
    in the checkpointable bounded assembler.  No consumed fragment is dropped.
    """
    collection = _FreshCollection(config)
    while collection.needs_more():
        _raise_if_actor_exited(actor_processes, stop_event)
        trace = assembler.pop_ready()
        if trace is None:
            payload = _wait_for_actor_fragment(
                output_queue, stop_event, actor_processes
            )
            _receive_fragment(payload, collection, assembler, config)
            _raise_if_actor_exited(actor_processes, stop_event)
            trace = assembler.pop_ready()
            if trace is None:
                continue
        collection.add_trace(trace)
    return collection.traces, collection.metrics(assembler)

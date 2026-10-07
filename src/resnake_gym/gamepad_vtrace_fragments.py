"""Actor-fragment schema and causal validation for gamepad V-trace."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from torch import Tensor

import resnake_gym.gamepad as gamepad
from resnake_gym.gamepad_vtrace_contract import (
    FRAGMENT_FORMAT,
    VTraceConfig,
    _action_variable_count,
    action_semantics,
)

_DPAD5_REPORTS = gamepad.dpad5_reports()


def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Keep float32 exactly.  Quantizing body_progress would make the learner's
    # state differ from the state under which behavior_logp was recorded.
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def _stack_fragment_observations(
    observations: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def _new_buffer(
    *,
    actor_id: int,
    env_id: int,
    episode_id: int,
    initial_hidden: Tensor,
    score: int,
    action_head: str = "dpad5",
    burn_context=(),
    run_generation: int = 0,
    fragment_sequence: int = 0,
    decision_start: int = 0,
) -> dict[str, Any]:
    burn_context = list(burn_context)
    burn_h0 = (
        np.asarray(burn_context[0][1], dtype=np.float32).copy()
        if burn_context
        else initial_hidden.detach().cpu().numpy().astype(np.float32)
    )
    semantics = action_semantics(action_head)
    return {
        "actor_id": actor_id,
        "env_id": env_id,
        "episode_id": episode_id,
        "run_generation": run_generation,
        "fragment_sequence": fragment_sequence,
        "decision_start": decision_start,
        "score_start": int(score),
        "action_head": action_head,
        "action_encoding": semantics["action_encoding"],
        "initial_hidden": initial_hidden.detach().cpu().numpy().astype(np.float32),
        "burn_h0": burn_h0,
        "burn_observations": [
            _copy_observation(observation) for observation, _ in burn_context
        ],
        "observations": [],
        "policy_actions": [],
        "requested_reports": [],
        "behavior_log_probs": [],
        "behavior_slot_log_probs": [],
        "rewards": [],
        "discounts": [],
        "terminated": [],
        "truncated": [],
        "policy_versions": [],
        "ticks_advanced": [],
        "controller_ticks": [],
        "capture_ticks": [],
        "command_origin_ticks": [],
        "command_arrival_ticks": [],
        "execution_ticks": [],
        "tick_rewards": [],
        "executed_reports": [],
        "executed_sources": [],
        "submitted_sequences": [],
        "food_events": [],
        "food_event_transition_indices": [],
        "scores": [],
        "perturbation_events": 0,
        "expired_reports": 0,
    }


def _finish_fragment(
    buffer: dict[str, Any],
    bootstrap_observation: dict[str, np.ndarray],
    *,
    won: bool,
    termination_reason: str | None,
) -> dict[str, Any]:
    length = len(buffer["rewards"])
    if length < 1:
        raise ValueError("cannot finish an empty actor fragment")
    return {
        "kind": "fragment",
        "format": FRAGMENT_FORMAT,
        "fragment_id": uuid.uuid4().hex,
        "actor_id": buffer["actor_id"],
        "env_id": buffer["env_id"],
        "episode_id": buffer["episode_id"],
        "run_generation": buffer["run_generation"],
        "fragment_sequence": buffer["fragment_sequence"],
        "decision_start": buffer["decision_start"],
        "decision_end": buffer["decision_start"] + length,
        "action_head": buffer["action_head"],
        "action_encoding": buffer["action_encoding"],
        "action_variables_per_decision": (
            1
            if buffer["action_head"] == "held_dpad5"
            else int(np.asarray(buffer["policy_actions"][0]).size)
        ),
        "score_start": buffer["score_start"],
        "score_end": int(buffer["scores"][-1]),
        "won": bool(won),
        "termination_reason": termination_reason,
        "length": length,
        "initial_hidden": buffer["initial_hidden"],
        "burn_h0": buffer["burn_h0"],
        "burn_length": len(buffer["burn_observations"]),
        "burn_observations": (
            _stack_fragment_observations(buffer["burn_observations"])
            if buffer["burn_observations"]
            else {}
        ),
        "observations": _stack_fragment_observations(buffer["observations"]),
        "bootstrap_observation": _copy_observation(bootstrap_observation),
        "policy_actions": np.stack(buffer["policy_actions"]).astype(np.int8),
        "requested_reports": np.stack(buffer["requested_reports"]).astype(np.float32),
        "behavior_log_probs": np.asarray(
            buffer["behavior_log_probs"], dtype=np.float32
        ),
        "behavior_slot_log_probs": np.stack(buffer["behavior_slot_log_probs"]).astype(
            np.float32
        ),
        "rewards": np.asarray(buffer["rewards"], dtype=np.float32),
        "discounts": np.asarray(buffer["discounts"], dtype=np.float32),
        "terminated": np.asarray(buffer["terminated"], dtype=np.bool_),
        "truncated": np.asarray(buffer["truncated"], dtype=np.bool_),
        "policy_versions": np.asarray(buffer["policy_versions"], dtype=np.int64),
        "ticks_advanced": np.asarray(buffer["ticks_advanced"], dtype=np.int16),
        "controller_ticks": np.asarray(buffer["controller_ticks"], dtype=np.int64),
        "capture_ticks": np.asarray(buffer["capture_ticks"], dtype=np.int64),
        "command_origin_ticks": np.asarray(
            buffer["command_origin_ticks"], dtype=np.int64
        ),
        "command_arrival_ticks": np.asarray(
            buffer["command_arrival_ticks"], dtype=np.int64
        ),
        "execution_ticks": list(buffer["execution_ticks"]),
        "tick_rewards": list(buffer["tick_rewards"]),
        "executed_reports": list(buffer["executed_reports"]),
        "executed_sources": list(buffer["executed_sources"]),
        "submitted_sequences": np.asarray(
            buffer["submitted_sequences"], dtype=np.int64
        ),
        "timebase": "simulation_tick",
        "food_events": list(buffer["food_events"]),
        "food_count": len(buffer["food_events"]),
        "contains_food_event": bool(buffer["food_events"]),
        "food_event_transition_indices": np.asarray(
            buffer["food_event_transition_indices"], dtype=np.int32
        ),
        "scores": np.asarray(buffer["scores"], dtype=np.int32),
        "perturbation_events": int(buffer["perturbation_events"]),
        "expired_reports": int(buffer["expired_reports"]),
    }


def _append_transition(
    buffer: dict[str, Any],
    *,
    observation: dict[str, np.ndarray],
    policy_action: Tensor,
    requested_report: np.ndarray,
    behavior_log_prob: float,
    behavior_slot_log_probs: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    policy_version: int,
) -> None:
    buffer["observations"].append(_copy_observation(observation))
    buffer["policy_actions"].append(
        policy_action.detach().cpu().numpy().astype(np.int8)
    )
    buffer["requested_reports"].append(np.asarray(requested_report).copy())
    buffer["behavior_log_probs"].append(float(behavior_log_prob))
    behavior_slot_log_probs = np.asarray(behavior_slot_log_probs, dtype=np.float32)
    if buffer["action_head"] == "held_dpad5" and behavior_slot_log_probs.shape == (5,):
        behavior_slot_log_probs = behavior_slot_log_probs[None]
    buffer["behavior_slot_log_probs"].append(behavior_slot_log_probs.copy())
    buffer["rewards"].append(float(reward))
    buffer["discounts"].append(0.0 if terminated else float(info["bootstrap_discount"]))
    buffer["terminated"].append(bool(terminated))
    buffer["truncated"].append(bool(truncated))
    buffer["policy_versions"].append(int(policy_version))
    buffer["ticks_advanced"].append(int(info["ticks_advanced"]))
    buffer["controller_ticks"].append(int(info["controller_tick"]))
    buffer["capture_ticks"].append(int(info["capture_tick"]))
    buffer["command_origin_ticks"].append(int(info["command_origin_tick"]))
    buffer["command_arrival_ticks"].append(int(info["command_arrival_tick"]))
    buffer["execution_ticks"].append(list(info["execution_ticks"]))
    buffer["tick_rewards"].append(list(info["tick_rewards"]))
    buffer["executed_reports"].append(
        np.asarray(info["executed_reports"], dtype=np.float32).copy()
    )
    buffer["executed_sources"].append(
        np.asarray(info["executed_sources"], dtype=np.int64).copy()
    )
    buffer["submitted_sequences"].append(int(info["submitted_sequence"]))
    food_events = list(info["food_events"])
    buffer["food_events"].extend(food_events)
    buffer["food_event_transition_indices"].extend(
        [len(buffer["rewards"]) - 1] * len(food_events)
    )
    buffer["scores"].append(int(info["score"]))
    buffer["perturbation_events"] += len(info["perturbation_events"])
    buffer["expired_reports"] += int(info["expired_reports"])


def _fragment_command_table(
    fragments: list[dict[str, Any]],
) -> dict[int, tuple[int, int, np.ndarray]]:
    """Index command metadata that is actually present in audited fragments."""
    commands: dict[int, tuple[int, int, np.ndarray]] = {}
    for fragment in fragments:
        for sequence, origin, arrival, reports in zip(
            fragment["submitted_sequences"],
            fragment["command_origin_ticks"],
            fragment["command_arrival_ticks"],
            fragment["requested_reports"],
            strict=True,
        ):
            sequence = int(sequence)
            if sequence in commands:
                raise ValueError("fragment command sequence is duplicated")
            commands[sequence] = (int(origin), int(arrival), reports)
    return commands


def _validate_execution_causality(
    fragment: dict[str, Any],
    config: VTraceConfig,
    known_commands: dict[int, tuple[int, int, np.ndarray]],
) -> None:
    """Audit sources without rejecting valid commands from an older fragment.

    Positive sources older than ``known_commands`` remain legal when the local
    timing bounds cannot prove that they expired.  Whenever their command
    metadata is available, arrival, target slot, priority and report contents
    are checked exactly against :class:`AsyncGamepad`.
    """
    for index, execution_ticks in enumerate(fragment["execution_ticks"]):
        reports = np.asarray(fragment["executed_reports"][index])
        sources = np.asarray(fragment["executed_sources"][index])
        submitted = int(fragment["submitted_sequences"][index])
        origin = int(fragment["command_origin_ticks"][index])
        for _offset, (tick, source, report) in enumerate(
            zip(execution_ticks, sources, reports, strict=True)
        ):
            tick = int(tick)
            source = int(source)
            if source == -1:
                if not np.array_equal(report, _DPAD5_REPORTS[0]):
                    raise ValueError("neutral execution source has a nonneutral report")
            else:
                # Even without metadata for an older fragment, decision_min
                # gives a sound lower bound on the command's age.
                minimum_age = (submitted - source) * config.decision_min
                if minimum_age + (tick - origin) >= config.chunk_length:
                    raise ValueError("executed command source is already expired")
                if source in known_commands:
                    source_origin, source_arrival, source_reports = known_commands[
                        source
                    ]
                    source_slot = tick - source_origin
                    if not 0 <= source_slot < config.chunk_length:
                        raise ValueError("executed command source is outside its chunk")
                    if source_arrival > tick:
                        raise ValueError("executed command source has not arrived")
                    if not np.array_equal(report, source_reports[source_slot]):
                        raise ValueError(
                            "executed report does not match its known command source"
                        )

            eligible_known = [
                sequence
                for sequence, (
                    source_origin,
                    source_arrival,
                    _,
                ) in known_commands.items()
                if source_origin <= tick < source_origin + config.chunk_length
                and source_arrival <= tick
            ]
            if eligible_known and source != max(eligible_known):
                raise ValueError(
                    "executed command source violates known command priority"
                )


def _validate_fragment_schema(
    fragment: dict[str, Any], config: VTraceConfig
) -> tuple[int, int]:
    """Validate scalar identity, range, and action-contract metadata."""
    if fragment.get("kind") != "fragment":
        raise ValueError("expected an actor fragment")
    if fragment.get("format") != FRAGMENT_FORMAT:
        raise ValueError("fragment format mismatch")
    if not isinstance(fragment.get("fragment_id"), str) or not fragment["fragment_id"]:
        raise ValueError("fragment id is invalid")
    if (
        type(fragment.get("run_generation")) is not int
        or fragment["run_generation"] < 0
    ):
        raise ValueError("fragment run generation is invalid")
    for key, upper_bound in (
        ("actor_id", config.actor_processes),
        ("env_id", config.envs_per_actor),
    ):
        if type(fragment.get(key)) is not int or not 0 <= fragment[key] < upper_bound:
            raise ValueError(f"fragment {key} is invalid")
    if type(fragment.get("episode_id")) is not int or fragment["episode_id"] < 0:
        raise ValueError("fragment episode id is invalid")
    for key in ("fragment_sequence", "decision_start", "decision_end"):
        if type(fragment.get(key)) is not int or fragment[key] < 0:
            raise ValueError(f"fragment {key} is invalid")
    length = fragment.get("length")
    if type(length) is not int or not 1 <= length <= config.unroll_length:
        raise ValueError("invalid fragment length")
    if fragment["decision_end"] != fragment["decision_start"] + length:
        raise ValueError("fragment decision range does not match its length")
    if fragment.get("timebase") != "simulation_tick":
        raise ValueError("fragment timebase is invalid")
    action_variables = _action_variable_count(config)
    semantics = action_semantics(config.action_head)
    if fragment.get("action_head") != config.action_head:
        raise ValueError("fragment action head mismatch")
    if fragment.get("action_encoding") != semantics["action_encoding"]:
        raise ValueError("fragment action encoding mismatch")
    if fragment.get("action_variables_per_decision") != action_variables:
        raise ValueError("fragment action variable count mismatch")
    return length, action_variables


def _validate_fragment_shapes(
    fragment: dict[str, Any],
    config: VTraceConfig,
    *,
    length: int,
    action_variables: int,
) -> None:
    """Validate tensor, recurrent-state, burn-in, and audit row shapes."""
    policy_action_shape = (
        (length,)
        if config.action_head == "held_dpad5"
        else (length, config.chunk_length)
    )
    if fragment["policy_actions"].shape != policy_action_shape:
        raise ValueError("fragment policy action shape mismatch")
    if fragment["requested_reports"].shape != (
        length,
        config.chunk_length,
        20,
    ):
        raise ValueError("fragment full-report shape mismatch")
    if fragment["behavior_slot_log_probs"].shape != (
        length,
        action_variables,
        5,
    ):
        raise ValueError("fragment behavior distribution shape mismatch")
    for key in (
        "behavior_log_probs",
        "rewards",
        "discounts",
        "terminated",
        "truncated",
        "policy_versions",
        "ticks_advanced",
        "controller_ticks",
        "capture_ticks",
        "command_origin_ticks",
        "command_arrival_ticks",
        "submitted_sequences",
        "scores",
    ):
        if fragment[key].shape != (length,):
            raise ValueError(f"fragment {key} shape mismatch")
    if fragment["initial_hidden"].shape != (config.dim,):
        raise ValueError("fragment recurrent state shape mismatch")
    if fragment["burn_h0"].shape != (config.dim,):
        raise ValueError("fragment burn-in state shape mismatch")
    burn_length = fragment.get("burn_length")
    if type(burn_length) is not int or not 0 <= burn_length <= config.recurrent_burn_in:
        raise ValueError("invalid fragment burn-in length")
    if burn_length:
        if not fragment["burn_observations"]:
            raise ValueError("burn-in observations are missing")
        for key, values in fragment["burn_observations"].items():
            if values.shape[0] != burn_length:
                raise ValueError(f"fragment burn observation {key} length mismatch")
    elif fragment["burn_observations"]:
        raise ValueError("zero burn-in length must have no burn observations")
    if len(fragment["execution_ticks"]) != length:
        raise ValueError("fragment execution timestamp length mismatch")
    for key in ("tick_rewards", "executed_reports", "executed_sources"):
        if len(fragment[key]) != length:
            raise ValueError(f"fragment {key} length mismatch")


def _validate_fragment_behavior(fragment: dict[str, Any], config: VTraceConfig) -> None:
    """Validate behavior probabilities, sampled actions, and policy version."""
    if not np.isfinite(fragment["behavior_log_probs"]).all():
        raise ValueError("fragment contains nonfinite behavior log probability")
    if not np.isfinite(fragment["behavior_slot_log_probs"]).all():
        raise ValueError("fragment contains nonfinite behavior distribution")
    if np.any(fragment["discounts"] < 0) or np.any(fragment["discounts"] > 1):
        raise ValueError("fragment discount is outside [0, 1]")
    if np.any(fragment["policy_actions"] < 0) or np.any(
        fragment["policy_actions"] >= 5
    ):
        raise ValueError("fragment dpad5 category is outside [0, 5)")
    behavior_probabilities = np.exp(fragment["behavior_slot_log_probs"])
    if not np.allclose(behavior_probabilities.sum(-1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("fragment behavior log probabilities are not normalized")
    action_indices = fragment["policy_actions"]
    if config.action_head == "held_dpad5":
        action_indices = action_indices[:, None]
    selected_slot_log_probs = np.take_along_axis(
        fragment["behavior_slot_log_probs"],
        action_indices[..., None],
        axis=-1,
    )[..., 0].sum(-1)
    if not np.allclose(
        selected_slot_log_probs,
        fragment["behavior_log_probs"],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("fragment behavior joint log probability is inconsistent")
    versions = np.unique(fragment["policy_versions"])
    if len(versions) != 1:
        raise ValueError("fragment must contain exactly one behavior policy version")
    if (
        not np.issubdtype(fragment["policy_versions"].dtype, np.integer)
        or int(versions[0]) < 0
    ):
        raise ValueError("fragment behavior policy version must be non-negative")
    if np.any(fragment["ticks_advanced"] < 1) or np.any(
        fragment["ticks_advanced"] > config.decision_max
    ):
        raise ValueError("fragment decision duration is outside configured bounds")


def _validate_fragment_timeline(
    fragment: dict[str, Any], config: VTraceConfig, *, length: int
) -> None:
    """Validate decision sequence and simulation-tick chronology."""
    integer_timeline_fields = (
        "ticks_advanced",
        "controller_ticks",
        "capture_ticks",
        "command_origin_ticks",
        "command_arrival_ticks",
        "submitted_sequences",
    )
    if any(
        not np.issubdtype(fragment[key].dtype, np.integer)
        for key in integer_timeline_fields
    ):
        raise ValueError("fragment causal timeline must contain integers")
    expected_sequences = np.arange(
        fragment["decision_start"] + 1,
        fragment["decision_end"] + 1,
        dtype=np.int64,
    )
    if not np.array_equal(fragment["submitted_sequences"], expected_sequences):
        raise ValueError("fragment submitted sequence is not decision-contiguous")
    origins = fragment["command_origin_ticks"]
    arrivals = fragment["command_arrival_ticks"]
    controllers = fragment["controller_ticks"]
    captures = fragment["capture_ticks"]
    if np.any(origins < 0) or np.any(controllers <= origins):
        raise ValueError("fragment command/controller timeline is invalid")
    delays = arrivals - origins
    if np.any(delays < 0) or np.any(delays > config.command_delay_max):
        raise ValueError("fragment command arrival delay is outside configured bounds")
    if np.any(captures < 0) or np.any(captures > controllers):
        raise ValueError("fragment capture tick is outside the controller timeline")
    if np.any(np.diff(captures) < 0):
        raise ValueError("fragment capture ticks are not monotonic")
    if length > 1 and not np.array_equal(origins[1:], controllers[:-1]):
        raise ValueError("fragment command origins are not decision-contiguous")
    if fragment["decision_start"] == 0 and int(origins[0]) != 0:
        raise ValueError("episode command timeline does not start at zero")


def _validate_fragment_episode_metadata(
    fragment: dict[str, Any], *, length: int
) -> None:
    """Validate boundary, score, and food-event metadata."""
    if np.any(fragment["terminated"] & fragment["truncated"]):
        raise ValueError("a transition cannot be both terminated and truncated")
    if np.any(fragment["terminated"][:-1]) or np.any(fragment["truncated"][:-1]):
        raise ValueError("episode boundary may only appear at fragment end")
    ended = bool(fragment["terminated"][-1] or fragment["truncated"][-1])
    reason = fragment["termination_reason"]
    if ended != (isinstance(reason, str) and bool(reason)):
        raise ValueError("fragment boundary and termination reason disagree")
    if fragment["won"] and not bool(fragment["terminated"][-1]):
        raise ValueError("a win must be a true terminal transition")
    if bool(fragment["won"]) != (reason == "board_filled"):
        raise ValueError("win flag and board-filled reason disagree")
    if fragment["truncated"][-1] and reason != "time_limit":
        raise ValueError("a truncated fragment must end for the time limit")
    if fragment["score_end"] != int(fragment["scores"][-1]):
        raise ValueError("fragment score_end does not match its score trace")
    score_path = np.concatenate(
        (np.asarray([fragment["score_start"]]), fragment["scores"])
    )
    if np.any(np.diff(score_path) < 0):
        raise ValueError("fragment score cannot decrease")
    food_count = fragment.get("food_count")
    if type(food_count) is not int or food_count != len(fragment["food_events"]):
        raise ValueError("fragment food count does not match its events")
    contains_food_event = fragment.get("contains_food_event")
    if type(contains_food_event) is not bool or contains_food_event != bool(food_count):
        raise ValueError("fragment contains_food_event does not match its events")
    event_transition_indices = fragment.get("food_event_transition_indices")
    if not isinstance(event_transition_indices, np.ndarray) or (
        event_transition_indices.shape != (food_count,)
    ):
        raise ValueError("fragment food event transition indices are invalid")
    if food_count and not np.all(event_transition_indices == length - 1):
        raise ValueError("a food event fragment must end at its event transition")
    if fragment["score_end"] - fragment["score_start"] != food_count:
        raise ValueError("fragment score change does not match its food events")


def _validate_fragment_action_semantics(
    fragment: dict[str, Any], config: VTraceConfig
) -> None:
    """Reconstruct discounts and reports from their causal source fields."""
    expected_discounts = np.power(
        config.gamma, fragment["ticks_advanced"], dtype=np.float64
    )
    expected_discounts = np.where(fragment["terminated"], 0.0, expected_discounts)
    if not np.allclose(fragment["discounts"], expected_discounts, rtol=1e-5, atol=1e-7):
        raise ValueError("fragment discount does not equal gamma ** ticks")
    expected_reports = np.zeros_like(fragment["requested_reports"])
    report_actions = fragment["policy_actions"]
    if config.action_head == "held_dpad5":
        report_actions = np.repeat(report_actions[:, None], config.chunk_length, axis=1)
    for category, button in ((1, 0), (2, 1), (3, 2), (4, 3)):
        expected_reports[..., button] = report_actions == category
    if not np.array_equal(fragment["requested_reports"], expected_reports):
        raise ValueError("requested full report does not match dpad5 category")


def _validate_fragment_execution_audit(
    fragment: dict[str, Any], config: VTraceConfig
) -> None:
    """Validate per-tick execution rows and their command causality."""
    known_commands = _fragment_command_table([fragment])
    for index, ticks in enumerate(fragment["ticks_advanced"]):
        ticks = int(ticks)
        tick_rewards = fragment["tick_rewards"][index]
        executed_reports = np.asarray(fragment["executed_reports"][index])
        executed_sources = np.asarray(fragment["executed_sources"][index])
        execution_ticks = fragment["execution_ticks"][index]
        if not (
            len(tick_rewards)
            == len(executed_reports)
            == len(executed_sources)
            == len(execution_ticks)
            == ticks
        ):
            raise ValueError("executed audit rows must match ticks_advanced")
        if executed_reports.shape != (ticks, 20):
            raise ValueError("executed report must have shape [K, 20]")
        if not np.isfinite(executed_reports).all():
            raise ValueError("executed report contains nonfinite values")
        legal_reports = np.any(
            np.all(
                executed_reports[:, None, :] == _DPAD5_REPORTS[None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        if not np.all(legal_reports):
            raise ValueError("executed report is outside the dpad5 report set")
        if executed_sources.shape != (ticks,) or not np.issubdtype(
            executed_sources.dtype, np.integer
        ):
            raise ValueError("executed source must have integer shape [K]")
        submitted = int(fragment["submitted_sequences"][index])
        if np.any(
            (executed_sources < -1)
            | (executed_sources == 0)
            | (executed_sources > submitted)
        ):
            raise ValueError("executed source is not a submitted command or neutral")
        origin = int(fragment["command_origin_ticks"][index])
        controller = int(fragment["controller_ticks"][index])
        if execution_ticks != list(range(origin, controller)):
            raise ValueError("execution ticks do not match command/controller timeline")
        if controller - origin != ticks:
            raise ValueError("execution/controller ticks are inconsistent")
        expected_reward = sum(
            config.gamma**offset * float(reward)
            for offset, reward in enumerate(tick_rewards)
        )
        if not np.isclose(
            fragment["rewards"][index], expected_reward, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("fragment reward does not match discounted tick rewards")
    _validate_execution_causality(fragment, config, known_commands)


def _validate_fragment_observations(fragment: dict[str, Any], *, length: int) -> None:
    """Validate the transition dimension of each observation component."""
    for key, values in fragment["observations"].items():
        if values.shape[0] != length:
            raise ValueError(f"fragment observation {key} length mismatch")


def validate_fragment(fragment: dict[str, Any], config: VTraceConfig) -> None:
    """Validate one serialized actor fragment without changing its contents."""
    length, action_variables = _validate_fragment_schema(fragment, config)
    _validate_fragment_shapes(
        fragment,
        config,
        length=length,
        action_variables=action_variables,
    )
    _validate_fragment_behavior(fragment, config)
    _validate_fragment_timeline(fragment, config, length=length)
    _validate_fragment_episode_metadata(fragment, length=length)
    _validate_fragment_action_semantics(fragment, config)
    _validate_fragment_execution_audit(fragment, config)
    _validate_fragment_observations(fragment, length=length)

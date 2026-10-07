"""Distributed recurrent V-trace for autonomous gamepad control.

The environment boundary remains a complete ``[L, 20]`` normalized gamepad
report.  The policy's score-function action is one of the versioned ``dpad5``
categorical latents described in :mod:`resnake_gym.models`; no board-dependent
action mask, teacher, demonstration, or collision repair is used here.

Actors never stop or reset a live game at a learner-update boundary.  They emit
episode-local fragments and retain their recurrent state across fragment cuts.
The decision containing a food event also closes its fragment so the event and
its causal prefix stay together; the next fragment continues the same game and
GRU state.
The learner may wait beyond its minimum amount of fresh experience for food
events, but consumes every FIFO fragment it observes.  Consequently the event
quota changes batch size, not which fresh transitions are accepted.
"""

from __future__ import annotations

from resnake_gym.gamepad_vtrace_actor import (
    actor_worker as actor_worker,
)
from resnake_gym.gamepad_vtrace_actor import (
    load_published_parameters as load_published_parameters,
)
from resnake_gym.gamepad_vtrace_actor import (
    parameter_count as parameter_count,
)
from resnake_gym.gamepad_vtrace_actor import (
    publish_parameters as publish_parameters,
)
from resnake_gym.gamepad_vtrace_contract import (
    ACTION_ENCODING as ACTION_ENCODING,
)
from resnake_gym.gamepad_vtrace_contract import (
    CHECKPOINT_FORMAT as CHECKPOINT_FORMAT,
)
from resnake_gym.gamepad_vtrace_contract import (
    COLLECTION_SEMANTICS_VERSION as COLLECTION_SEMANTICS_VERSION,
)
from resnake_gym.gamepad_vtrace_contract import (
    CREDIT_ASSEMBLER_FORMAT as CREDIT_ASSEMBLER_FORMAT,
)
from resnake_gym.gamepad_vtrace_contract import (
    CREDIT_TRACE_FORMAT as CREDIT_TRACE_FORMAT,
)
from resnake_gym.gamepad_vtrace_contract import (
    FRAGMENT_FORMAT as FRAGMENT_FORMAT,
)
from resnake_gym.gamepad_vtrace_contract import (
    RECURRENT_STATE_VERSION as RECURRENT_STATE_VERSION,
)
from resnake_gym.gamepad_vtrace_contract import (
    TRAINING_OBJECTIVE_VERSION as TRAINING_OBJECTIVE_VERSION,
)
from resnake_gym.gamepad_vtrace_contract import (
    VTraceConfig as VTraceConfig,
)
from resnake_gym.gamepad_vtrace_contract import (
    checkpoint_metadata as checkpoint_metadata,
)
from resnake_gym.gamepad_vtrace_contract import (
    validate_checkpoint as validate_checkpoint,
)
from resnake_gym.gamepad_vtrace_credit import (
    CreditTraceAssembler as CreditTraceAssembler,
)
from resnake_gym.gamepad_vtrace_credit import (
    _make_credit_trace as _make_credit_trace,
)
from resnake_gym.gamepad_vtrace_credit import (
    collect_fresh_credit_traces as collect_fresh_credit_traces,
)
from resnake_gym.gamepad_vtrace_credit import (
    validate_credit_trace as validate_credit_trace,
)
from resnake_gym.gamepad_vtrace_fragments import (
    _append_transition as _append_transition,
)
from resnake_gym.gamepad_vtrace_fragments import (
    _copy_observation as _copy_observation,
)
from resnake_gym.gamepad_vtrace_fragments import (
    _finish_fragment as _finish_fragment,
)
from resnake_gym.gamepad_vtrace_fragments import (
    _new_buffer as _new_buffer,
)
from resnake_gym.gamepad_vtrace_fragments import (
    validate_fragment as validate_fragment,
)
from resnake_gym.gamepad_vtrace_learner import (
    GamepadVTraceLearner as GamepadVTraceLearner,
)

__all__ = [
    "ACTION_ENCODING",
    "CHECKPOINT_FORMAT",
    "COLLECTION_SEMANTICS_VERSION",
    "CREDIT_ASSEMBLER_FORMAT",
    "CREDIT_TRACE_FORMAT",
    "CreditTraceAssembler",
    "FRAGMENT_FORMAT",
    "GamepadVTraceLearner",
    "RECURRENT_STATE_VERSION",
    "TRAINING_OBJECTIVE_VERSION",
    "VTraceConfig",
    "actor_worker",
    "checkpoint_metadata",
    "collect_fresh_credit_traces",
    "load_published_parameters",
    "parameter_count",
    "publish_parameters",
    "validate_checkpoint",
    "validate_credit_trace",
    "validate_fragment",
]

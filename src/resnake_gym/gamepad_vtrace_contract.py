"""Configuration and version contracts for gamepad V-trace training."""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass
from typing import Any

CHECKPOINT_FORMAT = "gamepad-vtrace-v3"
TRAINING_OBJECTIVE_VERSION = "snake-dpad5-event-trace-vtrace-v2"
COLLECTION_SEMANTICS_VERSION = "fifo-stitched-credit-trace-v3"
RECURRENT_STATE_VERSION = "stored-state-burnin-window-tbptt-v5"
ACTION_ENCODING = "xinput-dpad5-categorical-v1"
HELD_TRAINING_OBJECTIVE_VERSION = "snake-held-dpad5-event-trace-vtrace-h1-v2"
HELD_ACTION_ENCODING = "xinput-held-dpad5-categorical-h1-v1"
FRAGMENT_FORMAT = "gamepad-vtrace-fragment-v5"
CREDIT_TRACE_FORMAT = "gamepad-vtrace-credit-trace-v1"
CREDIT_ASSEMBLER_FORMAT = "gamepad-vtrace-credit-assembler-v2"


def action_semantics(action_head: str) -> dict[str, str]:
    """Return action-local versions without changing shared V-trace formats."""
    if action_head == "dpad5":
        return {
            "training_objective": TRAINING_OBJECTIVE_VERSION,
            "action_encoding": ACTION_ENCODING,
        }
    if action_head == "held_dpad5":
        return {
            "training_objective": HELD_TRAINING_OBJECTIVE_VERSION,
            "action_encoding": HELD_ACTION_ENCODING,
        }
    raise ValueError(f"unknown V-trace action head: {action_head}")


def _action_variable_count(config: VTraceConfig) -> int:
    return 1 if config.action_head == "held_dpad5" else config.chunk_length


@dataclass
class VTraceConfig:
    """Environment, actor and learner settings with explicit time semantics."""

    width: int = 31
    height: int = 20
    logic_fps: float = 10.0
    chunk_length: int = 8
    decision_min: int = 2
    decision_max: int = 5
    observation_delay_max: int = 3
    command_delay_max: int = 2
    drop_probability: float = 0.1
    max_logic_steps: int = 200_000
    dim: int = 64
    gamma: float = 0.9999
    learning_rate: float = 1e-4
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    rho_bar: float = 1.0
    c_bar: float = 1.0
    pg_rho_bar: float = 1.0
    seed: int = 2026
    decoder: str = "parallel"
    perturbation_min: int = 20
    perturbation_max: int = 40
    random_rotation: bool = False
    solver_ms: float = 50.0
    shaping_scale: float = 0.25
    death_cost: float = 3.0
    spatial_pool: bool = True
    chunk_rho: float | None = None
    action_head: str = "dpad5"
    actor_processes: int = 8
    envs_per_actor: int = 2
    actor_sync_steps: int = 128
    unroll_length: int = 128
    recurrent_burn_in: int = 64
    bptt_window: int = 128
    credit_trace_max_transitions: int = 2048
    batch_min_transitions: int = 2048
    batch_max_transitions: int = 8192
    batch_food_target: int = 4
    queue_capacity: int = 16
    # Runtime compatibility for old v3 callers.  InitVar keeps these values
    # out of dataclass fields and checkpoint metadata; __post_init__ shadows
    # the class defaults on the instance so legacy attribute reads still see
    # the values that were supplied.  New replay experiments should use
    # ExperimentalReplayConfig from the experimental package instead.
    replay_capacity: InitVar[int] = 128
    replay_retention: InitVar[str] = "reservoir"
    replay_batches_per_update: InitVar[int] = 0
    replay_batch_fragments: InitVar[int] = 16
    replay_warmup_fragments: InitVar[int] = 16
    replay_event_fraction: InitVar[float] = 0.5
    replay_high_progress_fraction: InitVar[float] = 0.1
    replay_is_beta: InitVar[float] = 0.4
    initial_length: int = 3

    def __post_init__(
        self,
        replay_capacity: int,
        replay_retention: str,
        replay_batches_per_update: int,
        replay_batch_fragments: int,
        replay_warmup_fragments: int,
        replay_event_fraction: float,
        replay_high_progress_fraction: float,
        replay_is_beta: float,
    ):
        positive = (
            self.width,
            self.height,
            self.initial_length,
            self.chunk_length,
            self.max_logic_steps,
            self.dim,
            self.actor_processes,
            self.envs_per_actor,
            self.actor_sync_steps,
            self.unroll_length,
            self.bptt_window,
            self.credit_trace_max_transitions,
            self.batch_min_transitions,
            self.batch_max_transitions,
            self.queue_capacity,
            replay_capacity,
            replay_batch_fragments,
            replay_warmup_fragments,
        )
        if any(value < 1 for value in positive):
            raise ValueError("V-trace counts and dimensions must be positive")
        if self.initial_length > self.width:
            raise ValueError("initial_length cannot exceed width")
        if self.batch_food_target < 0:
            raise ValueError("batch_food_target cannot be negative")
        if self.recurrent_burn_in < 0:
            raise ValueError("recurrent_burn_in cannot be negative")
        if self.recurrent_burn_in > self.unroll_length:
            raise ValueError("recurrent_burn_in cannot exceed unroll_length")
        if self.bptt_window > 128:
            raise ValueError("bptt_window cannot exceed 128 transitions")
        if self.unroll_length > self.credit_trace_max_transitions:
            raise ValueError(
                "unroll_length cannot exceed the credit trace safety bound"
            )
        if self.batch_min_transitions > self.batch_max_transitions:
            raise ValueError("minimum batch size cannot exceed its hard limit")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if self.action_head not in ("dpad5", "held_dpad5"):
            raise ValueError("V-trace requires dpad5 or held_dpad5")
        if self.chunk_rho is not None:
            raise ValueError("V-trace dpad5 action heads do not support chunk_rho")
        if self.decoder not in ("parallel", "gru"):
            raise ValueError("decoder must be parallel or gru")
        for name in ("rho_bar", "c_bar", "pg_rho_bar"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.c_bar > self.rho_bar:
            raise ValueError("c_bar cannot exceed rho_bar")
        if replay_retention not in ("reservoir", "fifo"):
            raise ValueError("replay_retention must be reservoir or fifo")
        if replay_warmup_fragments > replay_capacity:
            raise ValueError("replay warmup cannot exceed replay capacity")
        if replay_batches_per_update != 0:
            raise ValueError(
                "event-trace V-trace currently requires replay_batches_per_update=0; "
                "isolated fragment replay would truncate credit assignment"
            )
        if not 0 <= replay_high_progress_fraction <= replay_event_fraction <= 1:
            raise ValueError("invalid replay event/progress fractions")
        if not 0 <= replay_is_beta <= 1:
            raise ValueError("replay_is_beta must be in [0, 1]")
        legacy_replay_values = {
            "replay_capacity": replay_capacity,
            "replay_retention": replay_retention,
            "replay_batches_per_update": replay_batches_per_update,
            "replay_batch_fragments": replay_batch_fragments,
            "replay_warmup_fragments": replay_warmup_fragments,
            "replay_event_fraction": replay_event_fraction,
            "replay_high_progress_fraction": replay_high_progress_fraction,
            "replay_is_beta": replay_is_beta,
        }
        for name, value in legacy_replay_values.items():
            setattr(self, name, value)
        self._experimental_replay_config = {
            "capacity": replay_capacity,
            "retention": replay_retention,
            "batch_fragments": replay_batch_fragments,
            "event_fraction": replay_event_fraction,
            "high_progress_fraction": replay_high_progress_fraction,
            "is_beta": replay_is_beta,
            "seed": self.seed,
        }


def checkpoint_metadata(config: VTraceConfig) -> dict[str, Any]:
    """Versioned semantics shared by full and inference checkpoints."""
    semantics = action_semantics(config.action_head)
    return {
        "format": CHECKPOINT_FORMAT,
        "training_objective": semantics["training_objective"],
        "collection_semantics": COLLECTION_SEMANTICS_VERSION,
        "recurrent_state": RECURRENT_STATE_VERSION,
        "action_encoding": semantics["action_encoding"],
        "scene_encoding": "scene-grid-v2",
        "time_encoding": "policy-time-v2",
        "reward_version": "food-efficiency-pbrs-v1",
        "config": asdict(config),
    }


def validate_checkpoint(payload: dict[str, Any]) -> None:
    """Reject checkpoints whose actor, recurrence, or reward meaning differs."""
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict) or "action_head" not in config_payload:
        raise ValueError("V-trace checkpoint has no explicit action head")
    try:
        expected = checkpoint_metadata(VTraceConfig(**config_payload))
    except (TypeError, ValueError) as error:
        raise ValueError("V-trace checkpoint config is invalid") from error
    for key in (
        "format",
        "training_objective",
        "collection_semantics",
        "recurrent_state",
        "action_encoding",
        "scene_encoding",
        "time_encoding",
        "reward_version",
    ):
        if payload.get(key) != expected[key]:
            raise ValueError(f"V-trace checkpoint {key} mismatch")

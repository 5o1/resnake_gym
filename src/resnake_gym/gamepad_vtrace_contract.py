"""Configuration and version contracts for gamepad V-trace training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CHECKPOINT_FORMAT = "gamepad-vtrace-v3"
TRAINING_OBJECTIVE_VERSION = "snake-dpad5-event-trace-vtrace-v2"
COLLECTION_SEMANTICS_VERSION = "fifo-stitched-credit-trace-v3"
RECURRENT_STATE_VERSION = "stored-state-burnin-window-tbptt-v5"
ACTION_ENCODING = "xinput-dpad5-categorical-v1"
FRAGMENT_FORMAT = "gamepad-vtrace-fragment-v5"
CREDIT_TRACE_FORMAT = "gamepad-vtrace-credit-trace-v1"
CREDIT_ASSEMBLER_FORMAT = "gamepad-vtrace-credit-assembler-v2"


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
    # Serialized in the original gamepad-vtrace-v3 contract.  Keep these
    # fields so existing dpad5 checkpoints remain loadable; the retired H1
    # diagnostic is rejected explicitly below.
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
    initial_length: int = 3

    def __post_init__(self):
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
        if self.action_head != "dpad5":
            raise ValueError("V-trace supports the dpad5 action head only")
        if self.chunk_rho is not None:
            raise ValueError("V-trace dpad5 does not support chunk_rho")
        if self.decoder not in ("parallel", "gru"):
            raise ValueError("decoder must be parallel or gru")
        for name in ("rho_bar", "c_bar", "pg_rho_bar"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.c_bar > self.rho_bar:
            raise ValueError("c_bar cannot exceed rho_bar")


def checkpoint_metadata(config: VTraceConfig) -> dict[str, Any]:
    """Versioned semantics shared by full and inference checkpoints."""
    return {
        "format": CHECKPOINT_FORMAT,
        "training_objective": TRAINING_OBJECTIVE_VERSION,
        "collection_semantics": COLLECTION_SEMANTICS_VERSION,
        "recurrent_state": RECURRENT_STATE_VERSION,
        "action_encoding": ACTION_ENCODING,
        "scene_encoding": "scene-grid-v2",
        "time_encoding": "policy-time-v2",
        "reward_version": "food-efficiency-pbrs-v1",
        "config": asdict(config),
    }


def validate_checkpoint(payload: dict[str, Any]) -> None:
    """Reject checkpoints whose actor, recurrence, or reward meaning differs."""
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("V-trace checkpoint has no valid config")
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

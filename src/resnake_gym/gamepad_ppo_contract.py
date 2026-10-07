"""Stable configuration and checkpoint contract for the recurrent PPO policy.

Inference and evaluation need to reconstruct and validate a PPO checkpoint, but
they should not import the trainer implementation merely to do so.  This module
contains that serialized boundary and deliberately contains no optimization or
rollout code.
"""

from dataclasses import dataclass
from typing import Any

TRAINING_OBJECTIVE_VERSION = "snake-causal-marginal-v1"
STRUCTURED_TRAINING_OBJECTIVE_VERSION = "snake-dpad5-categorical-v1"
COLLECTION_SEMANTICS_VERSION = "food-quota-v1"
RECURRENT_UPDATE_VERSION = "stored-state-burnin-v1"
REPLAY_OBJECTIVE_VERSION = "success-snapshot-sil-v2"
CHECKPOINT_FORMAT = "gamepad-ppo-v3"
STRUCTURED_CHECKPOINT_FORMAT = "gamepad-ppo-v4"


def checkpoint_semantics(action_head: str) -> dict[str, str]:
    """Return the exact checkpoint semantics for one policy action head."""
    if action_head == "raw":
        return {
            "format": CHECKPOINT_FORMAT,
            "training_objective": TRAINING_OBJECTIVE_VERSION,
            "action_encoding": "xinput-normalized-v1",
        }
    if action_head == "dpad5":
        return {
            "format": STRUCTURED_CHECKPOINT_FORMAT,
            "training_objective": STRUCTURED_TRAINING_OBJECTIVE_VERSION,
            "action_encoding": "xinput-dpad5-categorical-v1",
        }
    raise ValueError(f"unknown action head: {action_head}")


def validate_policy_checkpoint(payload: dict[str, Any]) -> None:
    """Reject policies whose forward or training semantics would be changed."""
    action_head = payload.get("config", {}).get("action_head", "raw")
    semantics = checkpoint_semantics(action_head)
    if payload.get("format") != semantics["format"]:
        raise ValueError(f"requires {semantics['format']} checkpoint")
    if (
        action_head != "raw"
        and payload.get("action_encoding") != semantics["action_encoding"]
    ):
        raise ValueError("action_encoding mismatch")
    expected_versions = {
        "training_objective": semantics["training_objective"],
        "collection_semantics": COLLECTION_SEMANTICS_VERSION,
        "recurrent_update": RECURRENT_UPDATE_VERSION,
        "replay_objective": REPLAY_OBJECTIVE_VERSION,
    }
    for key, expected in expected_versions.items():
        if payload.get(key) != expected:
            raise ValueError(f"{key} mismatch; use its original policy implementation")


@dataclass
class PPOConfig:
    width: int = 31
    height: int = 20
    logic_fps: float = 10.0
    chunk_length: int = 8
    decision_min: int = 2
    decision_max: int = 5
    observation_delay_max: int = 3
    command_delay_max: int = 2
    drop_probability: float = 0.1
    max_logic_steps: int = 3000
    num_envs: int = 4
    rollout_steps: int = 256
    recurrent_burn_in: int = 64
    recurrent_unroll: int = 128
    epochs: int = 3
    dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95  # per decision, not per game tick
    clip: float = 0.2
    entropy_coefficient: float = 0.001  # latent-action entropy, explicitly
    target_kl: float = 0.03
    seed: int = 2026
    decoder: str = "parallel"
    perturbation_min: int = 20
    perturbation_max: int = 40
    random_rotation: bool = False
    solver_ms: float = 50.0
    shaping_scale: float = 0.25
    death_cost: float = 3.0
    collection_mode: str = "fixed"
    minimum_steps: int = 128
    event_target: int = 4
    sil_updates: int = 0
    sil_weight: float = 0.1
    sil_batch_size: int = 1
    sil_priority_alpha: float = 0.0
    replay_capacity: int = 256
    replay_success_fraction: float = 0.5
    spatial_pool: bool = False
    chunk_rho: float | None = None
    action_head: str = "raw"
    initial_length: int = 3


__all__ = [
    "CHECKPOINT_FORMAT",
    "COLLECTION_SEMANTICS_VERSION",
    "PPOConfig",
    "RECURRENT_UPDATE_VERSION",
    "REPLAY_OBJECTIVE_VERSION",
    "STRUCTURED_CHECKPOINT_FORMAT",
    "STRUCTURED_TRAINING_OBJECTIVE_VERSION",
    "TRAINING_OBJECTIVE_VERSION",
    "checkpoint_semantics",
    "validate_policy_checkpoint",
]

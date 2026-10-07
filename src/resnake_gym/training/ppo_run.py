"""Run-level orchestration for the recurrent PPO/SIL trainer.

The command-line entry point owns argument parsing only.  This module owns the
artifact transaction: output initialization, compatible resume, metadata,
metrics, and checkpoint publication.  Policy collection and optimization stay
in :mod:`resnake_gym.gamepad_ppo`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from resnake_gym.gamepad_ppo import (
    COLLECTION_SEMANTICS_VERSION,
    RECURRENT_UPDATE_VERSION,
    REPLAY_OBJECTIVE_VERSION,
    GamepadPPO,
    PPOConfig,
    checkpoint_semantics,
)
from resnake_gym.provenance import python_source_hashes

PPO_CONFIG_ARGUMENT_FIELDS = (
    "sil_updates",
    "sil_batch_size",
    "sil_priority_alpha",
    "replay_capacity",
    "learning_rate",
    "entropy_coefficient",
    "spatial_pool",
    "chunk_rho",
    "action_head",
    "max_logic_steps",
    "seed",
    "width",
    "height",
    "initial_length",
    "dim",
    "num_envs",
    "rollout_steps",
    "recurrent_burn_in",
    "recurrent_unroll",
    "decoder",
    "solver_ms",
    "gamma",
    "gae_lambda",
    "shaping_scale",
    "death_cost",
    "collection_mode",
    "minimum_steps",
    "event_target",
    "replay_success_fraction",
)

RESUME_COMPATIBILITY_FIELDS = (
    "dim",
    "chunk_length",
    "decoder",
    "gamma",
    "shaping_scale",
    "death_cost",
    "chunk_rho",
    "action_head",
    "spatial_pool",
    "initial_length",
    "recurrent_burn_in",
    "recurrent_unroll",
    "replay_success_fraction",
    "entropy_coefficient",
)


@dataclass(frozen=True, slots=True)
class PPORunOptions:
    """Process and artifact options that are not part of ``PPOConfig``."""

    output: Path
    resume: Path | None
    updates: int
    save_every: int
    device: str
    threads: int
    source_root: Path


def build_ppo_config(arguments: Mapping[str, Any]) -> PPOConfig:
    """Translate the stable CLI names into an algorithm configuration."""
    return PPOConfig(**{name: arguments[name] for name in PPO_CONFIG_ARGUMENT_FIELDS})


def lightweight_policy_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove optimizer and all replay-only state from an inference snapshot."""
    excluded = {"optimizer", "replay_episodes", "replay_success_snapshots"}
    result = {key: value for key, value in payload.items() if key not in excluded}
    result["note"] = "inference-only model checkpoint; no optimizer or replay state"
    return result


source_hashes = python_source_hashes


def validate_resume_config(payload: Mapping[str, Any], config: PPOConfig) -> None:
    """Reject a resume that would change learned-policy semantics."""
    defaults = PPOConfig()
    restored = payload["config"]
    for key in RESUME_COMPATIBILITY_FIELDS:
        if restored.get(key, getattr(defaults, key)) != getattr(config, key):
            raise ValueError(f"resume config mismatch: {key}")


def restore_trainer(
    trainer: GamepadPPO,
    config: PPOConfig,
    resume: Path,
) -> None:
    """Restore trainer state and apply the invocation's optimizer rate."""
    payload = torch.load(resume, map_location="cpu", weights_only=True)
    validate_resume_config(payload, config)
    trainer.restore(payload)
    for group in trainer.optimizer.param_groups:
        group["lr"] = config.learning_rate


def build_run_metadata(
    config: PPOConfig,
    options: PPORunOptions,
) -> dict[str, Any]:
    """Build the existing config.json artifact schema."""
    semantics = checkpoint_semantics(config.action_head)
    metadata = {
        "config": asdict(config),
        "torch": torch.__version__,
        "device": options.device,
        "experiment": "autonomous_ppo_sil_learning",
        "resume": str(options.resume) if options.resume else None,
        "demonstrations": False,
        "rule_teacher": False,
        "wall_clock_realtime_test": False,
        "perturbations_implemented": True,
        "solver_mode": "sync",
        "time_encoding": "policy-time-v2",
        "checkpoint_format": semantics["format"],
        "action_encoding": semantics["action_encoding"],
        "training_objective": semantics["training_objective"],
        "collection_semantics": COLLECTION_SEMANTICS_VERSION,
        "recurrent_update": RECURRENT_UPDATE_VERSION,
        "replay_objective": REPLAY_OBJECTIVE_VERSION,
    }
    metadata["source_sha256"] = source_hashes(options.source_root)
    return metadata


def write_run_metadata(
    config: PPOConfig,
    options: PPORunOptions,
) -> None:
    """Write metadata after trainer creation and optional restore."""
    metadata = build_run_metadata(config, options)
    (options.output / "config.json").write_text(json.dumps(metadata, indent=2))


def save_periodic_checkpoint(
    trainer: GamepadPPO,
    output: Path,
    update: int,
) -> None:
    """Publish the resumable checkpoint, then its inference-only snapshot."""
    target = output / "checkpoint.pt"
    temporary = output / "checkpoint.tmp"
    payload = trainer.checkpoint()
    torch.save(payload, temporary)
    temporary.replace(target)
    torch.save(
        lightweight_policy_checkpoint(payload),
        output / f"policy-{update:06d}.pt",
    )


def run_ppo_updates(trainer: GamepadPPO, options: PPORunOptions) -> None:
    """Run updates in order and preserve the metrics/checkpoint artifact schema."""
    with (options.output / "metrics.jsonl").open("w") as log:
        for update in range(options.updates):
            metrics = trainer.update()
            metrics["update"] = update + 1
            line = json.dumps(metrics)
            log.write(line + "\n")
            log.flush()
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in metrics.items()
                        if key not in ("episodes", "food_events", "fragments")
                    }
                ),
                flush=True,
            )
            if (update + 1) % options.save_every == 0:
                save_periodic_checkpoint(trainer, options.output, update + 1)
    torch.save(trainer.checkpoint(), options.output / "checkpoint.pt")


def run_ppo_training(config: PPOConfig, options: PPORunOptions) -> None:
    """Initialize, optionally restore, run, and close one PPO invocation."""
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(options.threads)
    trainer = GamepadPPO(config, options.device, archive=options.output / "episodes")
    try:
        if options.resume:
            restore_trainer(trainer, config, options.resume)
        write_run_metadata(config, options)
        run_ppo_updates(trainer, options)
    finally:
        trainer.close()


__all__ = [
    "PPO_CONFIG_ARGUMENT_FIELDS",
    "PPORunOptions",
    "RESUME_COMPATIBILITY_FIELDS",
    "build_ppo_config",
    "build_run_metadata",
    "lightweight_policy_checkpoint",
    "restore_trainer",
    "run_ppo_training",
    "run_ppo_updates",
    "save_periodic_checkpoint",
    "source_hashes",
    "validate_resume_config",
    "write_run_metadata",
]

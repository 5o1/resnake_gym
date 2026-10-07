"""Complete checkpoint evaluation and immutable result publication."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from resnake_gym.artifact_io import sha256_file
from resnake_gym.evaluation_protocol import (
    DETERMINISTIC_SOLVER_BUDGET,
    frozen_gamepad_evaluation_protocol,
    frozen_gamepad_evaluation_request,
    protocol_sha256,
)
from resnake_gym.gamepad_runtime import make_env, stack_observations
from resnake_gym.models import (
    distribution_from_observation,
    reports_from_policy_action,
    sample_policy_action,
)
from resnake_gym.policy_checkpoint import load_policy_model


@dataclass(frozen=True, slots=True)
class PolicyEvaluationOptions:
    """Inputs that identify one standalone checkpoint evaluation."""

    checkpoint: Path
    output: Path
    sizes: tuple[str, ...]
    episodes: int
    seed: int
    device: str
    stochastic: bool


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Create one JSON result without exposing partial contents."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _rollout_episode(env, model, config, options, episode: int):
    torch.manual_seed(options.seed + episode)
    observation, _ = env.reset(seed=options.seed + episode)
    hidden = None
    terminated = truncated = False
    observed_ages = []
    timings = []
    while not (terminated or truncated):
        inputs = stack_observations([observation], options.device)
        if str(options.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        distribution, _, hidden = distribution_from_observation(
            model,
            inputs,
            hidden,
        )
        action = sample_policy_action(distribution, stochastic=options.stochastic)
        report = (
            reports_from_policy_action(
                distribution,
                action,
                continuous_buttons=config.chunk_rho is not None,
            )[0]
            .cpu()
            .numpy()
        )
        timings.append(time.perf_counter() - started)
        observed_ages.append(float(observation["timing"][0]))
        observation, _, terminated, truncated, info = env.step(report)
    return info, observed_ages, timings


def _evaluate_sizes(model, config, options: PolicyEvaluationOptions):
    results = []
    timings = []
    with torch.no_grad():
        for size in options.sizes:
            config.width, config.height = map(int, size.split("x"))
            env = make_env(config, deterministic_solver=True)
            try:
                for episode in range(options.episodes):
                    info, observed_ages, episode_timings = _rollout_episode(
                        env,
                        model,
                        config,
                        options,
                        episode,
                    )
                    timings.extend(episode_timings)
                    results.append(
                        {
                            "size": size,
                            "seed": options.seed + episode,
                            "score": info["score"],
                            "won": info["won"],
                            "ticks": info["logic_steps"],
                            "termination": info["termination_reason"],
                            "mean_observation_age_seconds": float(
                                np.mean(observed_ages)
                            ),
                        }
                    )
            finally:
                env.close()
    return results, timings


def _source_training_iteration(checkpoint: dict[str, Any], path: Path):
    iteration = checkpoint.get("training_iteration")
    if iteration is None:
        match = re.fullmatch(r"policy-(\d+)", path.stem)
        if match is not None:
            iteration = int(match.group(1))
    return int(iteration) if iteration is not None else None


def _source_fields(checkpoint: dict[str, Any], config, path: Path) -> dict[str, Any]:
    policy_version = checkpoint.get("policy_version")
    return {
        "source_training_iteration": _source_training_iteration(checkpoint, path),
        "source_policy_version": (
            int(policy_version) if policy_version is not None else None
        ),
        "source_logic_ticks": int(checkpoint.get("logic_ticks", 0)),
        "source_logic_ticks_semantics": "trained",
        "source_trained_logic_ticks": int(
            checkpoint.get("trained_logic_ticks", checkpoint.get("logic_ticks", 0))
        ),
        "source_trained_transitions": int(
            checkpoint.get("trained_transitions", checkpoint.get("transitions", 0))
        ),
        "source_received_logic_ticks": (
            int(checkpoint["received_logic_ticks"])
            if "received_logic_ticks" in checkpoint
            else None
        ),
        "source_received_transitions": (
            int(checkpoint["received_transitions"])
            if "received_transitions" in checkpoint
            else None
        ),
        "source_action_head": config.action_head,
        "source_action_encoding": checkpoint.get("action_encoding"),
        "source_training_objective": checkpoint.get("training_objective"),
        "source_checkpoint_format": checkpoint.get("format"),
        "source_collection_semantics": checkpoint.get("collection_semantics"),
        "source_recurrent_state": checkpoint.get("recurrent_state"),
        "source_recurrent_update": checkpoint.get("recurrent_update"),
        "source_replay_objective": checkpoint.get("replay_objective"),
        "source_credit_trace_max_transitions": getattr(
            config,
            "credit_trace_max_transitions",
            None,
        ),
        "source_bptt_window": getattr(config, "bptt_window", None),
        "source_action_variables_per_decision": (
            config.chunk_length if config.action_head == "dpad5" else None
        ),
    }


def build_evaluation_result(
    checkpoint: dict[str, Any],
    config,
    options: PolicyEvaluationOptions,
    *,
    checkpoint_sha256: str,
    results: list[dict[str, Any]],
    timings: list[float],
) -> dict[str, Any]:
    """Assemble the versioned result without performing environment I/O."""
    request = frozen_gamepad_evaluation_request(
        sizes=options.sizes,
        episodes_per_size=options.episodes,
        base_seed=options.seed,
        stochastic=options.stochastic,
    )
    protocol = frozen_gamepad_evaluation_protocol(
        config,
        sizes=options.sizes,
        episodes_per_size=options.episodes,
        base_seed=options.seed,
        stochastic=options.stochastic,
    )
    return {
        "checkpoint": str(options.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_request": request,
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": protocol_sha256(protocol),
        **_source_fields(checkpoint, config, options.checkpoint),
        "episodes": results,
        "deterministic_policy": not options.stochastic,
        "same_weights_across_sizes": True,
        "deterministic_solver_budget": DETERMINISTIC_SOLVER_BUDGET,
        "wall_clock_realtime_test": False,
        "latency_scope": (
            "forward plus output CPU transfer; excludes input transfer, no warmup"
        ),
        "latency_seconds_p50_p95": np.quantile(timings, [0.5, 0.95]).tolist(),
        "note": (
            "Perturbed simulation, no robot. "
            "Not a closed-loop capability acceptance result."
        ),
    }


def run_policy_evaluation(options: PolicyEvaluationOptions) -> dict[str, Any]:
    """Evaluate one immutable checkpoint and atomically publish the result."""
    if options.episodes < 1 or not options.sizes:
        raise ValueError("evaluation requires positive episodes and at least one size")
    checkpoint_sha256 = sha256_file(options.checkpoint)
    checkpoint, config, model = load_policy_model(
        options.checkpoint,
        options.device,
    )
    results, timings = _evaluate_sizes(model, config, options)
    if sha256_file(options.checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during evaluation")
    output = build_evaluation_result(
        checkpoint,
        config,
        options,
        checkpoint_sha256=checkpoint_sha256,
        results=results,
        timings=timings,
    )
    atomic_json(options.output, output)
    return output


__all__ = [
    "PolicyEvaluationOptions",
    "atomic_json",
    "build_evaluation_result",
    "run_policy_evaluation",
    "sha256_file",
]

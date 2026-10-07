"""End-to-end orchestration for the production V-trace trainer.

The command-line entry point owns argument parsing only.  This module owns the
stateful training transaction: output initialization, matched learner/runtime
resume, actor lifecycle, one optimizer iteration, metrics, and checkpoints.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from resnake_gym.gamepad_vtrace_contract import (
    VTraceConfig,
    checkpoint_metadata,
)
from resnake_gym.gamepad_vtrace_credit import CreditTraceAssembler
from resnake_gym.gamepad_vtrace_learner import GamepadVTraceLearner
from resnake_gym.provenance import python_source_hashes
from resnake_gym.training.vtrace_actor_pool import VTraceActorPool
from resnake_gym.training.vtrace_checkpoint import (
    assert_credit_accounting,
    load_matching_runtime,
    save_checkpoint,
)
from resnake_gym.training.vtrace_metrics import (
    METRICS_STATE_FORMAT,
    RunMetricsState,
)
from resnake_gym.training.vtrace_reporting import (
    CollectionPopulations,
    build_iteration_metrics,
)

VTRACE_CONFIG_ARGUMENT_FIELDS = (
    "width",
    "height",
    "initial_length",
    "dim",
    "gamma",
    "learning_rate",
    "entropy_coefficient",
    "rho_bar",
    "c_bar",
    "pg_rho_bar",
    "max_logic_steps",
    "actor_processes",
    "envs_per_actor",
    "actor_sync_steps",
    "unroll_length",
    "recurrent_burn_in",
    "bptt_window",
    "credit_trace_max_transitions",
    "batch_min_transitions",
    "batch_max_transitions",
    "batch_food_target",
    "queue_capacity",
    "shaping_scale",
    "death_cost",
    "solver_ms",
    "decoder",
    "action_head",
    "seed",
)


@dataclass(frozen=True, slots=True)
class VTraceRunOptions:
    """Process- and artifact-level options outside the learned policy contract."""

    output: Path
    resume: Path | None
    updates: int
    save_every: int
    device: str
    threads: int
    max_fresh_logic_ticks: int | None
    source_root: Path


@dataclass(slots=True)
class VTraceRunState:
    """All mutable state that survives across optimizer iterations."""

    learner: GamepadVTraceLearner
    assembler: CreditTraceAssembler
    metrics: RunMetricsState
    start_iteration: int
    run_generation: int


def build_vtrace_config(arguments: Mapping[str, Any]) -> VTraceConfig:
    """Translate versioned CLI names into the algorithm configuration."""
    values = {name: arguments[name] for name in VTRACE_CONFIG_ARGUMENT_FIELDS}
    values["spatial_pool"] = not bool(arguments["no_spatial_pool"])
    return VTraceConfig(**values)


source_hashes = python_source_hashes


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish metadata without exposing a partially written JSON document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def save_best_trace(trace: Mapping[str, Any], output: Path) -> None:
    """Atomically replace the best progress trace artifact."""
    temporary = output / "best-progress-trace.tmp"
    torch.save(trace, temporary)
    temporary.replace(output / "best-progress-trace.pt")


def build_run_metadata(
    config: VTraceConfig,
    options: VTraceRunOptions,
    *,
    run_generation: int,
) -> dict[str, Any]:
    """Build the stable config.json schema for a training invocation."""
    metadata = {
        **checkpoint_metadata(config),
        "torch": torch.__version__,
        "device": options.device,
        "experiment": "autonomous_recurrent_impala_vtrace",
        "demonstrations": False,
        "rule_teacher": False,
        "state_dependent_action_mask": False,
        "fresh_fifo_discards": False,
        "fragment_replay_training_enabled": False,
        "runtime_state_checkpointed": True,
        "credit_assembler_checkpointed": True,
        "resume": str(options.resume) if options.resume else None,
        "run_generation": run_generation,
        "training_budget": {
            "primary_axis": "cumulative_received_logic_ticks",
            "max_fresh_logic_ticks": options.max_fresh_logic_ticks,
            "updates_safety_cap": options.updates,
            "overshoot": "at_most_one_complete_fresh_collection_batch",
        },
        "metric_semantics": METRICS_STATE_FORMAT,
    }
    metadata["source_sha256"] = source_hashes(options.source_root)
    return metadata


def _restore_assembler(
    assembler: CreditTraceAssembler,
    learner: GamepadVTraceLearner,
    resume: Path,
    resume_payload: Mapping[str, Any],
    *,
    max_fresh_logic_ticks: int | None,
) -> None:
    runtime_checkpoint_id = resume_payload.get(
        "runtime_checkpoint_id",
        resume_payload.get("replay_checkpoint_id"),
    )
    if not isinstance(runtime_checkpoint_id, str) or not runtime_checkpoint_id:
        raise ValueError("V-trace checkpoint has no runtime snapshot identity")
    # Runtime sidecars are local trusted artifacts and contain NumPy arrays.
    # Legacy replay sidecars remain readable, but only their assembler state
    # is restored; isolated fragment replay is not part of this algorithm.
    runtime_state = load_matching_runtime(resume, runtime_checkpoint_id)
    try:
        assembler.load_state_dict(runtime_state["credit_assembler"])
    except KeyError as error:
        raise ValueError("V-trace checkpoint has no credit assembler state") from error
    assembler.flush_for_new_generation()
    assert_credit_accounting(learner, assembler)
    if (
        max_fresh_logic_ticks is not None
        and max_fresh_logic_ticks <= assembler.cumulative_received_logic_ticks
    ):
        raise ValueError(
            "--max-fresh-logic-ticks must exceed the resumed received tick count"
        )


def _restore_best_trace(
    resume: Path,
    output: Path,
    run_metrics: RunMetricsState,
) -> None:
    if run_metrics.best_score < 0:
        return
    source_best = resume.parent / "best-progress-trace.pt"
    if not source_best.is_file():
        raise FileNotFoundError(
            "resumed best score has no sibling best-progress-trace.pt"
        )
    temporary_best = output / "best-progress-trace.tmp"
    shutil.copyfile(source_best, temporary_best)
    temporary_best.replace(output / "best-progress-trace.pt")


def initialize_vtrace_run(
    config: VTraceConfig,
    options: VTraceRunOptions,
) -> VTraceRunState:
    """Create or restore every persistent object before actors are started."""
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(options.threads)
    learner = GamepadVTraceLearner(config, options.device)
    start_iteration = 0
    resume_payload: dict[str, Any] | None = None
    if options.resume:
        resume_payload = torch.load(
            options.resume,
            map_location="cpu",
            weights_only=True,
        )
        learner.restore(resume_payload)
        start_iteration = int(resume_payload.get("training_iteration", 0))
        for group in learner.optimizer.param_groups:
            group["lr"] = config.learning_rate
        if options.updates <= start_iteration:
            raise ValueError("--updates must exceed the resumed training iteration")

    run_generation = (
        int(resume_payload.get("run_generation", 0)) + 1
        if resume_payload is not None
        else 0
    )
    assembler = CreditTraceAssembler(config)
    if options.resume:
        assert resume_payload is not None
        _restore_assembler(
            assembler,
            learner,
            options.resume,
            resume_payload,
            max_fresh_logic_ticks=options.max_fresh_logic_ticks,
        )

    metadata = build_run_metadata(
        config,
        options,
        run_generation=run_generation,
    )
    write_json_atomic(options.output / "config.json", metadata)

    run_metrics = (
        RunMetricsState.restore(resume_payload.get("script_state", {}))
        if resume_payload is not None
        else RunMetricsState()
    )
    if options.resume:
        _restore_best_trace(options.resume, options.output, run_metrics)
    return VTraceRunState(
        learner=learner,
        assembler=assembler,
        metrics=run_metrics,
        start_iteration=start_iteration,
        run_generation=run_generation,
    )


def run_vtrace_iteration(
    state: VTraceRunState,
    actor_pool: VTraceActorPool,
    options: VTraceRunOptions,
    *,
    iteration: int,
    started: float,
) -> dict[str, Any]:
    """Collect, optimize, publish, account, and report one fresh batch."""
    before = time.monotonic()
    traces, raw_collection = actor_pool.collect(state.assembler)
    fresh_metrics = state.learner.update(traces, source="fresh")
    for trace in traces:
        if trace["score_end"] > state.metrics.best_score:
            state.metrics.best_score = int(trace["score_end"])
            save_best_trace(trace, options.output)
    published_version = actor_pool.publish(policy_version=state.learner.update_count)
    # The primary training axis is receipt-side environment ticks.  Keep its
    # event population separate from traces selected for this optimizer step:
    # ready traces can be trained one update later without representing newly
    # received experience.
    populations = CollectionPopulations.split(raw_collection)
    collection = populations.metrics
    state.metrics.observe(
        populations.received_episodes,
        received_food_count=collection["fresh_received_food_count"],
        trained_food_count=collection["fresh_trained_food_count"],
    )
    elapsed = time.monotonic() - before
    conservation_error = assert_credit_accounting(state.learner, state.assembler)
    return build_iteration_metrics(
        iteration=iteration,
        collection=collection,
        learner_metrics=fresh_metrics,
        populations=populations,
        run_metrics=state.metrics,
        learner=state.learner,
        assembler=state.assembler,
        published_policy_version=published_version,
        wall_seconds=elapsed,
        total_wall_seconds=time.monotonic() - started,
        max_fresh_logic_ticks=options.max_fresh_logic_ticks,
        conservation_error=conservation_error,
    )


def run_vtrace_training(config: VTraceConfig, options: VTraceRunOptions) -> None:
    """Run the production trainer through its update or fresh-tick budget."""
    state = initialize_vtrace_run(config, options)
    actor_pool = VTraceActorPool(
        config,
        state.learner.model,
        run_generation=state.run_generation,
    )
    actor_pool.start(policy_version=state.learner.update_count)

    started = time.monotonic()
    last_iteration = state.start_iteration
    try:
        with (options.output / "metrics.jsonl").open("w") as log:
            for iteration in range(state.start_iteration + 1, options.updates + 1):
                last_iteration = iteration
                metrics = run_vtrace_iteration(
                    state,
                    actor_pool,
                    options,
                    iteration=iteration,
                    started=started,
                )
                line = json.dumps(metrics)
                log.write(line + "\n")
                log.flush()
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in metrics.items()
                            if key not in ("episodes", "trained_episodes")
                        }
                    ),
                    flush=True,
                )
                if iteration % options.save_every == 0:
                    save_checkpoint(
                        state.learner,
                        state.assembler,
                        options.output,
                        iteration=iteration,
                        script_state=state.metrics.checkpoint_payload(),
                        run_generation=state.run_generation,
                    )
                if (
                    options.max_fresh_logic_ticks is not None
                    and state.assembler.cumulative_received_logic_ticks
                    >= options.max_fresh_logic_ticks
                ):
                    break
        save_checkpoint(
            state.learner,
            state.assembler,
            options.output,
            iteration=last_iteration,
            script_state=state.metrics.checkpoint_payload(),
            run_generation=state.run_generation,
        )
    finally:
        actor_pool.close()


__all__ = [
    "VTRACE_CONFIG_ARGUMENT_FIELDS",
    "VTraceRunOptions",
    "VTraceRunState",
    "build_run_metadata",
    "build_vtrace_config",
    "initialize_vtrace_run",
    "run_vtrace_iteration",
    "run_vtrace_training",
    "save_best_trace",
    "source_hashes",
    "write_json_atomic",
]

"""Matched atomic publication of learner and assembler V-trace checkpoints.

The rename protocol survives a Python-process interruption at each publication
boundary.  It does not fsync file contents or the containing directory, so it
must not be described as durable across a host crash or power loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

RUNTIME_STATE_FORMAT = "gamepad-vtrace-runtime-v1"


class LearnerState(Protocol):
    logic_ticks: int
    transitions: int
    update_count: int

    def checkpoint(self) -> dict[str, Any]: ...


class AssemblerState(Protocol):
    cumulative_received_logic_ticks: int
    cumulative_received_transitions: int
    pending_logic_ticks: int
    pending_transitions: int
    ready_logic_ticks: int
    ready_transitions: int

    def state_dict(self) -> dict[str, Any]: ...


def credit_accounting_errors(
    learner: LearnerState, assembler: AssemblerState
) -> tuple[int, int]:
    """Return receipt minus trained/pending/ready conservation residuals."""
    return (
        assembler.cumulative_received_logic_ticks
        - learner.logic_ticks
        - assembler.pending_logic_ticks
        - assembler.ready_logic_ticks,
        assembler.cumulative_received_transitions
        - learner.transitions
        - assembler.pending_transitions
        - assembler.ready_transitions,
    )


def assert_credit_accounting(
    learner: LearnerState, assembler: AssemblerState
) -> tuple[int, int]:
    """Fail before publishing a checkpoint that has lost received experience."""
    tick_error, transition_error = credit_accounting_errors(learner, assembler)
    if tick_error != 0 or transition_error != 0:
        raise RuntimeError(
            "credit accounting invariant failed: "
            f"ticks={tick_error}, transitions={transition_error}"
        )
    return tick_error, transition_error


def save_checkpoint(
    learner: LearnerState,
    assembler: AssemblerState,
    output: Path,
    *,
    iteration: int,
    script_state: dict[str, Any],
    run_generation: int,
) -> str:
    """Publish a matched learner/runtime pair with atomic rename boundaries.

    Fragment replay is not part of the current algorithm.  The sidecar stores
    only assembler state needed to conserve already received experience.  This
    protects process-level interruption; it is not a power-loss durability
    guarantee because the files and directory are not explicitly fsynced.
    """
    assert_credit_accounting(learner, assembler)
    checkpoint_id = (
        f"generation-{run_generation}-iteration-{iteration}-"
        f"update-{learner.update_count}-received-"
        f"{assembler.cumulative_received_logic_ticks}"
    )
    payload = learner.checkpoint()
    payload.update(
        {
            "trained_logic_ticks": learner.logic_ticks,
            "trained_transitions": learner.transitions,
            "received_logic_ticks": assembler.cumulative_received_logic_ticks,
            "received_transitions": assembler.cumulative_received_transitions,
            "training_iteration": iteration,
            "script_state": script_state,
            "run_generation": run_generation,
            "policy_version": learner.update_count,
            "runtime_checkpoint_id": checkpoint_id,
        }
    )
    checkpoint_temporary = output / "checkpoint.tmp"
    torch.save(payload, checkpoint_temporary)

    inference = {key: value for key, value in payload.items() if key != "optimizer"}
    inference["note"] = (
        "inference-only V-trace model; no optimizer or runtime assembler state"
    )
    policy_temporary = output / "policy.tmp"
    torch.save(inference, policy_temporary)

    runtime_payload = {
        "format": RUNTIME_STATE_FORMAT,
        "credit_assembler": assembler.state_dict(),
        "checkpoint_id": checkpoint_id,
        "training_iteration": iteration,
        "learner_update": learner.update_count,
        "run_generation": run_generation,
    }
    runtime_temporary = output / "runtime.tmp"
    torch.save(runtime_payload, runtime_temporary)

    runtime_path = output / "runtime.pt"
    previous_runtime = output / "runtime.previous.pt"
    if runtime_path.exists():
        runtime_path.replace(previous_runtime)
    runtime_temporary.replace(runtime_path)
    policy_temporary.replace(output / f"policy-{iteration:06d}.pt")
    checkpoint_temporary.replace(output / "checkpoint.pt")
    previous_runtime.unlink(missing_ok=True)
    return checkpoint_id


def load_matching_runtime(
    resume_checkpoint: Path, checkpoint_id: str
) -> dict[str, Any]:
    """Load the matching assembler sidecar, including legacy replay sidecars."""
    checked = []
    candidates = (
        ("runtime.pt", True),
        ("runtime.previous.pt", True),
        ("replay.pt", False),
        ("replay.previous.pt", False),
    )
    for name, current in candidates:
        path = resume_checkpoint.parent / name
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        checked.append((name, state.get("checkpoint_id")))
        if state.get("checkpoint_id") != checkpoint_id:
            continue
        if current and state.get("format") != RUNTIME_STATE_FORMAT:
            raise ValueError("V-trace runtime state format mismatch")
        if not isinstance(state.get("credit_assembler"), dict):
            raise ValueError("V-trace runtime has no credit assembler state")
        return state
    raise ValueError(
        "no runtime snapshot matches the learner checkpoint: " + repr(checked)
    )


__all__ = [
    "RUNTIME_STATE_FORMAT",
    "assert_credit_accounting",
    "credit_accounting_errors",
    "load_matching_runtime",
    "save_checkpoint",
]

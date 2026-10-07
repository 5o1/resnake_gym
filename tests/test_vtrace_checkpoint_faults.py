"""Failure-injection tests for learner/runtime checkpoint publication."""

from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.training.vtrace_checkpoint import (  # noqa: E402
    load_matching_runtime,
    save_checkpoint,
)


@dataclass
class _Learner:
    logic_ticks: int = 7
    transitions: int = 5
    update_count: int = 3

    def checkpoint(self):
        return {"model": {"weight": torch.tensor([1.0])}, "optimizer": {}}


@dataclass
class _Assembler:
    cumulative_received_logic_ticks: int = 11
    cumulative_received_transitions: int = 8
    pending_logic_ticks: int = 2
    pending_transitions: int = 1
    ready_logic_ticks: int = 2
    ready_transitions: int = 2

    def state_dict(self):
        return {"format": "assembler-fixture", "pending": []}


def _assert_published_pair_matches(output: Path) -> None:
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    runtime = load_matching_runtime(
        output / "checkpoint.pt", checkpoint["runtime_checkpoint_id"]
    )
    assert runtime["checkpoint_id"] == checkpoint["runtime_checkpoint_id"]


@pytest.mark.parametrize("failed_replace", [1, 2, 3, 4])
def test_each_atomic_replace_failure_keeps_a_matching_resume_pair(
    tmp_path, monkeypatch, failed_replace
):
    learner = _Learner()
    assembler = _Assembler()
    save_checkpoint(
        learner,
        assembler,
        tmp_path,
        iteration=1,
        script_state={},
        run_generation=0,
    )
    learner.update_count = 4
    original_replace = Path.replace
    calls = 0

    def injected_replace(path, target):
        nonlocal calls
        calls += 1
        if calls == failed_replace:
            raise OSError(f"injected replace failure {failed_replace}")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", injected_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        save_checkpoint(
            learner,
            assembler,
            tmp_path,
            iteration=2,
            script_state={},
            run_generation=0,
        )

    _assert_published_pair_matches(tmp_path)


def test_failure_after_new_checkpoint_publication_keeps_new_pair(tmp_path, monkeypatch):
    learner = _Learner()
    assembler = _Assembler()
    save_checkpoint(
        learner,
        assembler,
        tmp_path,
        iteration=1,
        script_state={},
        run_generation=0,
    )
    learner.update_count = 4
    original_unlink = Path.unlink

    def injected_unlink(path, *args, **kwargs):
        if path.name == "runtime.previous.pt":
            raise OSError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", injected_unlink)
    with pytest.raises(OSError, match="injected cleanup failure"):
        save_checkpoint(
            learner,
            assembler,
            tmp_path,
            iteration=2,
            script_state={},
            run_generation=0,
        )

    _assert_published_pair_matches(tmp_path)

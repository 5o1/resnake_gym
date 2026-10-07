from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.training.vtrace_checkpoint import (  # noqa: E402
    RUNTIME_STATE_FORMAT,
    assert_credit_accounting,
    load_matching_runtime,
    save_checkpoint,
)


@dataclass
class FakeLearner:
    logic_ticks: int = 7
    transitions: int = 5
    update_count: int = 3

    def checkpoint(self):
        return {"model": {"weight": torch.tensor([1.0])}, "optimizer": {}}


@dataclass
class FakeAssembler:
    cumulative_received_logic_ticks: int = 11
    cumulative_received_transitions: int = 8
    pending_logic_ticks: int = 2
    pending_transitions: int = 1
    ready_logic_ticks: int = 2
    ready_transitions: int = 2

    def state_dict(self):
        return {"format": "assembler-fixture", "pending": []}


def test_checkpoint_uses_matched_runtime_sidecar(tmp_path):
    checkpoint_id = save_checkpoint(
        FakeLearner(),
        FakeAssembler(),
        tmp_path,
        iteration=4,
        script_state={"metrics_state_format": "fixture"},
        run_generation=2,
    )

    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    runtime = torch.load(tmp_path / "runtime.pt", weights_only=False)
    policy = torch.load(tmp_path / "policy-000004.pt", weights_only=True)

    assert checkpoint["runtime_checkpoint_id"] == checkpoint_id
    assert runtime["format"] == RUNTIME_STATE_FORMAT
    assert runtime["checkpoint_id"] == checkpoint_id
    assert set(runtime) == {
        "format",
        "credit_assembler",
        "checkpoint_id",
        "training_iteration",
        "learner_update",
        "run_generation",
    }
    assert "optimizer" not in policy
    assert load_matching_runtime(tmp_path / "checkpoint.pt", checkpoint_id) == runtime


def test_checkpoint_rejects_broken_credit_accounting():
    assembler = FakeAssembler(cumulative_received_logic_ticks=12)
    with pytest.raises(RuntimeError, match="credit accounting invariant"):
        assert_credit_accounting(FakeLearner(), assembler)

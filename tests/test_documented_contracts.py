from pathlib import Path

from resnake_gym.gamepad_vtrace import (
    CHECKPOINT_FORMAT,
    COLLECTION_SEMANTICS_VERSION,
    CREDIT_ASSEMBLER_FORMAT,
    CREDIT_TRACE_FORMAT,
    FRAGMENT_FORMAT,
    RECURRENT_STATE_VERSION,
)

ROOT = Path(__file__).parents[1]


def test_vtrace_contract_versions_are_present_in_public_api_documentation():
    api = (ROOT / "docs/gamepad-api.html").read_text()
    for version in (
        CHECKPOINT_FORMAT,
        COLLECTION_SEMANTICS_VERSION,
        CREDIT_ASSEMBLER_FORMAT,
        CREDIT_TRACE_FORMAT,
        FRAGMENT_FORMAT,
        RECURRENT_STATE_VERSION,
    ):
        assert version in api


def test_vtrace_contract_versions_are_present_in_training_design():
    design = (ROOT / "docs/vtrace-progress-replay-design.md").read_text()
    for version in (
        CHECKPOINT_FORMAT,
        COLLECTION_SEMANTICS_VERSION,
        CREDIT_ASSEMBLER_FORMAT,
        CREDIT_TRACE_FORMAT,
        FRAGMENT_FORMAT,
        RECURRENT_STATE_VERSION,
    ):
        assert version in design

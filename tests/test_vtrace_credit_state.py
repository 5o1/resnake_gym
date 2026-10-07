"""Transactional restore checks for the V-trace credit assembler."""

import copy

import pytest

from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.gamepad_vtrace_credit import CreditTraceAssembler


def test_invalid_checkpoint_does_not_partially_mutate_live_assembler():
    assembler = CreditTraceAssembler(VTraceConfig())
    assembler.resume_flush_traces = 2
    assembler.resume_flush_transitions = 3
    assembler.cumulative_received_fragments = 5
    assembler.cumulative_received_transitions = 7
    assembler.cumulative_received_logic_ticks = 11
    before = copy.deepcopy(assembler.state_dict())
    invalid = copy.deepcopy(before)
    invalid["cumulative_received_transitions"] = -1

    with pytest.raises(ValueError, match="cumulative counters"):
        assembler.load_state_dict(invalid)

    assert assembler.state_dict() == before

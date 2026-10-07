import numpy as np
import pytest

from resnake_gym.action_audit import ActionExecutionAudit


def _report(category):
    value = np.zeros(20, np.float32)
    if category:
        value[category - 1] = 1
    return value


def _fragment(action_head):
    actions = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], np.int64)
    if action_head == "held_dpad5":
        actions = np.array([1, 4], np.int64)
    requested = np.zeros((2, 4, 20), np.float32)
    expanded = (
        np.repeat(actions[:, None], 4, axis=1)
        if action_head == "held_dpad5"
        else actions
    )
    for row in range(2):
        for slot in range(4):
            requested[row, slot] = _report(int(expanded[row, slot]))
    log_probs = np.log(np.full((2, 1 if action_head == "held_dpad5" else 4, 5), 0.2))
    return {
        "action_head": action_head,
        "length": 2,
        "policy_actions": actions,
        "behavior_slot_log_probs": log_probs,
        "requested_reports": requested,
        "ticks_advanced": np.array([2, 3]),
        "command_origin_ticks": np.array([0, 2]),
        "submitted_sequences": np.array([1, 2]),
        "execution_ticks": [[0, 1], [2, 3, 4]],
        "executed_sources": [np.array([1, 1]), np.array([1, 2, 2])],
        "executed_reports": [
            np.stack((requested[0, 0], requested[0, 1])),
            np.stack((requested[0, 2], requested[1, 1], requested[1, 2])),
        ],
    }


def test_j8_audit_keeps_additive_execution_and_nll_counts():
    audit = ActionExecutionAudit("dpad5", 4)
    audit.add_fragment(_fragment("dpad5"))
    metrics = audit.metrics()

    assert metrics["received_action_transition_count"] == 2
    assert metrics["received_action_slot_0_same_transition_execution_numerator"] == 1
    assert metrics["received_action_slot_1_same_transition_execution_numerator"] == 2
    assert metrics["received_action_slot_2_same_transition_execution_numerator"] == 1
    assert metrics["received_action_slot_3_same_transition_execution_numerator"] == 0
    assert metrics["received_action_slot_2_opportunity_execution_denominator"] == 1
    assert "received_action_slot_3_opportunity_execution_rate" not in metrics
    assert metrics["received_action_zero_current_source_rate"] == 0
    assert metrics["received_action_current_source_report_match_rate"] == 1
    assert metrics[
        "received_action_j8_unexecuted_chosen_nll_fraction"
    ] == pytest.approx(0.5)
    assert not any("h1_zero" in key for key in metrics)


def test_h1_audit_scores_one_probability_variable_not_repeated_slots():
    fragment = _fragment("held_dpad5")
    fragment["executed_sources"][1] = np.array([1, 1, -1])
    fragment["executed_reports"][1] = np.stack(
        (fragment["requested_reports"][0, 2], _report(0), _report(0))
    )
    audit = ActionExecutionAudit("held_dpad5", 4)

    audit.add_fragment(fragment)
    metrics = audit.metrics()

    assert metrics["received_action_zero_current_source_rate"] == 0.5
    assert metrics[
        "received_action_h1_zero_execution_chosen_nll_fraction"
    ] == pytest.approx(0.5)
    assert metrics["received_action_fallback_tick_numerator"] == 1
    assert metrics["received_action_executed_neutral_tick_numerator"] == 2
    assert not any("j8_unexecuted" in key for key in metrics)


def test_current_source_mismatch_is_counted_and_zero_denominators_are_omitted():
    fragment = _fragment("dpad5")
    fragment["executed_reports"][0][0] = _report(4)
    audit = ActionExecutionAudit("dpad5", 4)
    audit.add_fragment(fragment)

    metrics = audit.metrics()

    assert metrics["received_action_current_source_report_match_numerator"] == 3
    assert metrics["received_action_current_source_report_match_denominator"] == 4
    assert metrics["received_action_current_source_report_match_rate"] == 0.75

    empty = ActionExecutionAudit("held_dpad5", 4).metrics()
    assert "received_action_current_source_report_match_rate" not in empty
    assert "received_action_executed_neutral_tick_rate" not in empty

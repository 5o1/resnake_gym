"""Audit raw real-time trace without confusing scheduled and measured time."""

import numpy as np

from resnake_gym.gamepad import digital_report
from resnake_gym.protocol import GamepadCommand


def audit_timing(rows):
    errors, commands, ticks = [], {}, []
    summaries = [row for row in rows if row["type"] == "summary"]
    summary = summaries[-1] if summaries else None
    for row in rows:
        if row["type"] == "command":
            try:
                packet = GamepadCommand.from_wire(row["packet"])
                if packet.submitted_ns > row["received_ns"]:
                    errors.append("command_received_before_submission")
                if row["result"]["accepted"]:
                    commands[packet.sequence] = (packet, row["received_ns"])
            except (TypeError, ValueError):
                errors.append("invalid_command_packet")
        elif row["type"] == "tick":
            ticks.append(row)
    if not ticks or summary is None:
        errors.append("missing_ticks_or_summary")
    if [row["tick"] for row in ticks] != list(range(len(ticks))):
        errors.append("noncontiguous_logic_ticks")
    ages, lateness, step_cost = [], [], []
    inference = [
        p.inference_completed_ns - p.inference_started_ns for p, _ in commands.values()
    ]
    transport = [received - p.submitted_ns for p, received in commands.values()]
    previous_capture = 0
    for row in ticks:
        if not previous_capture <= row["applied_ns"] <= row["state_captured_ns"]:
            errors.append("nonmonotonic_game_execution")
        previous_capture = row["state_captured_ns"]
        if row["applied_ns"] < row["scheduled_ns"]:
            errors.append("executed_before_deadline")
        if summary:
            expected = summary["epoch_ns"] + (row["tick"] + 1) * summary["period_ns"]
            if (
                row["scheduled_ns"] != expected
                or row["clock_id"] != summary["clock_id"]
            ):
                errors.append("clock_or_deadline_mismatch")
        lateness.append(row["applied_ns"] - row["scheduled_ns"])
        step_cost.append(row["state_captured_ns"] - row["applied_ns"])
        if row["sequence"] is not None:
            if row["sequence"] not in commands:
                errors.append("execution_without_command")
                continue
            packet, received = commands[row["sequence"]]
            index = row["chunk_index"]
            if type(index) is not int or not 0 <= index < len(packet.reports):
                errors.append("invalid_chunk_index")
                continue
            if packet.origin_tick + index != row["tick"]:
                errors.append("late_prefix_was_shifted")
            if not np.array_equal(
                np.asarray(row["report"], np.float32),
                digital_report(packet.reports[index]),
            ):
                errors.append("executed_report_differs_from_command")
            if packet.clock_id != row["clock_id"] or received > row["applied_ns"]:
                errors.append("invalid_command_execution_clock")
            if row["command_received_ns"] != received:
                errors.append("receipt_timestamp_changed")
            if row["observation_captured_ns"] != packet.observation_captured_ns:
                errors.append("capture_timestamp_changed")
            ages.append(row["applied_ns"] - packet.observation_captured_ns)
    if summary and summary["stats"]["ticks"] != len(ticks):
        errors.append("summary_tick_count_mismatch")

    def quantiles(values):
        return (
            dict(
                zip(
                    ("p50", "p95", "p99", "max"),
                    np.quantile(values, [0.5, 0.95, 0.99, 1]).tolist(),
                    strict=True,
                )
            )
            if values
            else None
        )

    elapsed = (
        (ticks[-1]["state_captured_ns"] - summary["epoch_ns"])
        if ticks and summary
        else None
    )
    return {
        "valid_timeline": not errors,
        "errors": sorted(set(errors)),
        "ticks": len(ticks),
        "executed_command_reports": len(ages),
        "effective_logic_fps": len(ticks) * 1e9 / elapsed
        if elapsed and elapsed > 0
        else None,
        "start_lateness_ns": quantiles(lateness),
        "game_step_cost_ns": quantiles(step_cost),
        "observation_to_application_ns": quantiles(ages),
        "inference_duration_ns": quantiles(inference),
        "submit_to_game_receipt_ns": quantiles(transport),
        "late_over_one_period": sum(value >= summary["period_ns"] for value in lateness)
        if summary
        else None,
        "note": (
            "Receipt delay includes IPC and polling, not isolated network delay. "
            "Ages include chunk horizon. Inference/transport quantiles count "
            "each accepted command once; ages count each executed report."
        ),
    }

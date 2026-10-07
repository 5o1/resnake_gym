"""Training summaries and cross-run comparisons for V-trace v3."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .artifacts import Audit, _episode_summary, _finite_issues
from .protocol import FORMAL_ARMS, FORMAL_CONTRASTS, SCORE_THRESHOLDS


@dataclass
class _TrainingSeries:
    """Mutable reduction state for one ordered metrics JSONL series."""

    finite_issues: list[str] = field(default_factory=list)
    food_positions: list[int] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    previous_iteration: int = -1
    previous_received_ticks: int | None = None
    previous_received_transitions: int | None = None
    previous_food_count: int | None = None
    initial_ticks: int | None = None
    initial_transitions: int | None = None
    initial_food: int | None = None


@dataclass
class _CreditTraceSeries:
    """Aggregates trace diagnostics without coupling them to row traversal."""

    count: int = 0
    over_128_count: int = 0
    cap_hit_count: int = 0
    artificial_boundary_count: int = 0
    bootstrap_boundary_count: int = 0
    weighted_length: float = 0.0
    length_max: int = 0
    updates_with_over_128: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "length_mean_weighted": self.weighted_length / self.count
            if self.count
            else None,
            "length_max": self.length_max if self.count else None,
            "over_128_count": self.over_128_count,
            "over_128_rate": self.over_128_count / self.count if self.count else None,
            "over_128_activated": self.over_128_count > 0,
            "updates_with_over_128": self.updates_with_over_128,
            "cap_hit_count": self.cap_hit_count,
            "artificial_boundary_count": self.artificial_boundary_count,
            "bootstrap_boundary_count": self.bootstrap_boundary_count,
        }


def _audit_iteration(
    row: dict[str, Any], prefix: str, audit: Audit, series: _TrainingSeries
) -> None:
    iteration = row.get("training_iteration")
    audit.check(
        f"{prefix}.training_iteration",
        isinstance(iteration, int) and iteration > series.previous_iteration,
    )
    if isinstance(iteration, int):
        series.previous_iteration = iteration


def _audit_axes(
    row: dict[str, Any], prefix: str, audit: Audit, series: _TrainingSeries
) -> tuple[int, int, int] | None:
    """Validate receipt axes and return ticks, transitions, and batch start tick."""

    received_ticks = row.get("cumulative_received_logic_ticks")
    fresh_ticks = row.get("fresh_received_logic_ticks")
    received_transitions = row.get("cumulative_received_transitions")
    fresh_transitions = row.get("fresh_received_transitions")
    if not audit.check(
        f"{prefix}.received_tick_counters",
        isinstance(received_ticks, int)
        and isinstance(fresh_ticks, int)
        and received_ticks >= 0
        and fresh_ticks > 0,
    ):
        return None
    if not audit.check(
        f"{prefix}.received_transition_counters",
        isinstance(received_transitions, int)
        and isinstance(fresh_transitions, int)
        and received_transitions >= 0
        and fresh_transitions > 0,
    ):
        return None
    assert isinstance(received_ticks, int)
    assert isinstance(fresh_ticks, int)
    assert isinstance(received_transitions, int)
    assert isinstance(fresh_transitions, int)
    start_tick = _advance_axes(
        prefix,
        audit,
        series,
        received_ticks=received_ticks,
        fresh_ticks=fresh_ticks,
        received_transitions=received_transitions,
        fresh_transitions=fresh_transitions,
    )
    _audit_axis_aliases(
        row,
        prefix,
        audit,
        received_ticks=received_ticks,
        received_transitions=received_transitions,
    )
    return received_ticks, received_transitions, start_tick


def _advance_axes(
    prefix: str,
    audit: Audit,
    series: _TrainingSeries,
    *,
    received_ticks: int,
    fresh_ticks: int,
    received_transitions: int,
    fresh_transitions: int,
) -> int:
    if series.previous_received_ticks is None:
        series.initial_ticks = received_ticks - fresh_ticks
        series.initial_transitions = received_transitions - fresh_transitions
        audit.check(
            f"{prefix}.nonnegative_initial_tick_axis", series.initial_ticks >= 0
        )
        audit.check(
            f"{prefix}.nonnegative_initial_transition_axis",
            series.initial_transitions >= 0,
        )
    else:
        assert series.previous_received_transitions is not None
        audit.check(
            f"{prefix}.received_tick_increment",
            received_ticks - series.previous_received_ticks == fresh_ticks,
        )
        audit.check(
            f"{prefix}.received_transition_increment",
            received_transitions - series.previous_received_transitions
            == fresh_transitions,
        )
    start_tick = received_ticks - fresh_ticks
    series.previous_received_ticks = received_ticks
    series.previous_received_transitions = received_transitions
    return start_tick


def _audit_axis_aliases(
    row: dict[str, Any],
    prefix: str,
    audit: Audit,
    *,
    received_ticks: int,
    received_transitions: int,
) -> None:
    audit.check(
        f"{prefix}.received_tick_alias",
        row.get("cumulative_fresh_logic_ticks") == received_ticks,
    )
    audit.check(
        f"{prefix}.received_transition_alias",
        row.get("cumulative_fresh_transitions") == received_transitions,
    )


def _audit_food_positions(
    positions: Any,
    food_count: Any,
    prefix: str,
    audit: Audit,
    series: _TrainingSeries,
    *,
    start_tick: int,
    received_ticks: int,
) -> None:
    positions_valid = (
        isinstance(positions, list)
        and isinstance(food_count, int)
        and all(isinstance(position, int) for position in positions)
    )
    audit.check(f"{prefix}.food_positions_typed", positions_valid)
    if not positions_valid:
        return
    assert isinstance(positions, list)
    assert isinstance(food_count, int)
    audit.check(f"{prefix}.food_positions_count", len(positions) == food_count)
    audit.check(
        f"{prefix}.food_positions_in_receipt_batch",
        all(start_tick < position <= received_ticks for position in positions),
    )
    audit.check(
        f"{prefix}.food_positions_strictly_ordered",
        all(
            left < right for left, right in zip(positions, positions[1:], strict=False)
        ),
    )
    if series.food_positions and positions:
        audit.check(
            f"{prefix}.food_positions_global_order",
            series.food_positions[-1] < positions[0],
        )
    series.food_positions.extend(positions)


def _audit_food_counter(
    row: dict[str, Any],
    food_count: Any,
    prefix: str,
    audit: Audit,
    series: _TrainingSeries,
) -> None:
    cumulative_food = row.get("cumulative_received_food_count")
    if not isinstance(cumulative_food, int) or not isinstance(food_count, int):
        audit.check(f"{prefix}.received_food_counter", False)
        return
    if series.previous_food_count is None:
        series.initial_food = cumulative_food - food_count
        audit.check(f"{prefix}.nonnegative_initial_food", series.initial_food >= 0)
    else:
        audit.check(
            f"{prefix}.received_food_increment",
            cumulative_food - series.previous_food_count == food_count,
        )
    series.previous_food_count = cumulative_food


def _audit_food_series(
    row: dict[str, Any],
    prefix: str,
    audit: Audit,
    series: _TrainingSeries,
    *,
    start_tick: int,
    received_ticks: int,
) -> None:
    positions = row.get("fresh_received_food_tick_positions")
    food_count = row.get("fresh_received_food_count")
    _audit_food_positions(
        positions,
        food_count,
        prefix,
        audit,
        series,
        start_tick=start_tick,
        received_ticks=received_ticks,
    )
    _audit_food_counter(row, food_count, prefix, audit, series)


_CONSERVATION_FIELDS = (
    "cumulative_trained_logic_ticks",
    "pending_credit_logic_ticks",
    "ready_credit_logic_ticks",
    "cumulative_trained_transitions",
    "pending_credit_transitions",
    "ready_credit_transitions",
)


def _audit_credit_conservation(
    row: dict[str, Any],
    prefix: str,
    audit: Audit,
    *,
    received_ticks: int,
    received_transitions: int,
) -> None:
    fields_valid = all(isinstance(row.get(key), int) for key in _CONSERVATION_FIELDS)
    audit.check(f"{prefix}.conservation_fields", fields_valid)
    if not fields_valid:
        return
    tick_error = (
        received_ticks
        - row["cumulative_trained_logic_ticks"]
        - row["pending_credit_logic_ticks"]
        - row["ready_credit_logic_ticks"]
    )
    transition_error = (
        received_transitions
        - row["cumulative_trained_transitions"]
        - row["pending_credit_transitions"]
        - row["ready_credit_transitions"]
    )
    audit.check(f"{prefix}.tick_conservation_recomputed", tick_error == 0)
    audit.check(f"{prefix}.transition_conservation_recomputed", transition_error == 0)
    audit.check(
        f"{prefix}.tick_conservation_reported",
        row.get("received_trained_pending_ready_tick_error") == 0,
    )
    audit.check(
        f"{prefix}.transition_conservation_reported",
        row.get("received_trained_pending_ready_transition_error") == 0,
    )
    audit.check(
        f"{prefix}.conservation_flag",
        row.get("received_trained_pending_ready_conserved") is True,
    )


def _audit_received_episodes(
    row: dict[str, Any], prefix: str, audit: Audit, series: _TrainingSeries
) -> None:
    received_episodes = row.get("episodes")
    if not audit.check(
        f"{prefix}.received_episodes_list", isinstance(received_episodes, list)
    ):
        return
    assert isinstance(received_episodes, list)
    for episode_index, episode in enumerate(received_episodes):
        valid_episode = (
            isinstance(episode, dict)
            and isinstance(episode.get("score"), int)
            and isinstance(episode.get("won"), bool)
            and isinstance(episode.get("ticks", episode.get("controller_tick")), int)
        )
        audit.check(f"{prefix}.episodes[{episode_index}]", valid_episode)
        if valid_episode:
            series.episodes.append(episode)


def _accumulate_trace_boundary_counts(
    row: dict[str, Any],
    prefix: str,
    audit: Audit,
    traces: _CreditTraceSeries,
    batch_trace_count: int,
) -> None:
    destinations = (
        ("credit_trace_cap_hit_count", "cap_hit_count"),
        ("credit_artificial_boundary_count", "artificial_boundary_count"),
        ("credit_bootstrap_boundary_count", "bootstrap_boundary_count"),
    )
    for field_name, destination in destinations:
        value = row.get(field_name)
        valid = isinstance(value, int) and 0 <= value <= batch_trace_count
        audit.check(f"{prefix}.{field_name}", valid)
        if valid:
            setattr(traces, destination, getattr(traces, destination) + value)


def _audit_credit_traces(
    row: dict[str, Any], prefix: str, audit: Audit, traces: _CreditTraceSeries
) -> None:
    batch_trace_count = row.get("credit_trace_count")
    batch_over = row.get("credit_trace_over_128_count")
    if not isinstance(batch_trace_count, int) or batch_trace_count <= 0:
        audit.check(f"{prefix}.credit_trace_count", False)
        return
    traces.count += batch_trace_count
    batch_over_valid = (
        isinstance(batch_over, int) and 0 <= batch_over <= batch_trace_count
    )
    audit.check(f"{prefix}.trace_over_128_count", batch_over_valid)
    if batch_over_valid:
        traces.over_128_count += batch_over
        traces.updates_with_over_128 += int(batch_over > 0)
        audit.check(
            f"{prefix}.trace_over_128_rate",
            math.isclose(
                float(row.get("credit_trace_over_128_rate", math.nan)),
                batch_over / batch_trace_count,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ),
        )
    mean_length = row.get("credit_trace_len_mean")
    if isinstance(mean_length, (int, float)) and math.isfinite(mean_length):
        traces.weighted_length += float(mean_length) * batch_trace_count
    else:
        audit.check(f"{prefix}.trace_len_mean", False)
    batch_max = row.get("credit_trace_len_max")
    if isinstance(batch_max, int):
        traces.length_max = max(traces.length_max, batch_max)
    else:
        audit.check(f"{prefix}.trace_len_max", False)
    _accumulate_trace_boundary_counts(row, prefix, audit, traces, batch_trace_count)


def _audit_training_row(
    row: dict[str, Any],
    row_index: int,
    audit: Audit,
    series: _TrainingSeries,
    traces: _CreditTraceSeries,
) -> None:
    prefix = f"metrics[{row_index}]"
    series.finite_issues.extend(_finite_issues(row, prefix))
    _audit_iteration(row, prefix, audit, series)
    axes = _audit_axes(row, prefix, audit, series)
    if axes is None:
        return
    received_ticks, received_transitions, start_tick = axes
    _audit_food_series(
        row,
        prefix,
        audit,
        series,
        start_tick=start_tick,
        received_ticks=received_ticks,
    )
    _audit_credit_conservation(
        row,
        prefix,
        audit,
        received_ticks=received_ticks,
        received_transitions=received_transitions,
    )
    _audit_received_episodes(row, prefix, audit, series)
    _audit_credit_traces(row, prefix, audit, traces)


def _audit_budget(final: dict[str, Any], audit: Audit, *, expected_budget: int) -> Any:
    final_ticks = final.get("cumulative_received_logic_ticks")
    audit.check(
        "budget.reached",
        isinstance(final_ticks, int) and final_ticks >= expected_budget,
        f"expected at least {expected_budget}, got {final_ticks!r}",
    )
    audit.check(
        "budget.target_recorded_in_metrics",
        final.get("max_fresh_logic_ticks") == expected_budget,
    )
    if isinstance(final_ticks, int):
        expected_overshoot = max(0, final_ticks - expected_budget)
        audit.check(
            "budget.overshoot",
            final.get("fresh_logic_tick_budget_overshoot") == expected_overshoot,
        )
        audit.check(
            "budget.reached_flag",
            final.get("fresh_logic_tick_budget_reached")
            is (final_ticks >= expected_budget),
        )
    return final_ticks


def _audit_exact_food_history(audit: Audit, series: _TrainingSeries) -> None:
    if series.initial_ticks == 0:
        audit.check("food.exact_history_available", True)
        if series.previous_food_count is not None:
            audit.check(
                "food.cumulative_matches_positions",
                series.previous_food_count == len(series.food_positions),
            )
    else:
        audit.check(
            "food.exact_history_available",
            False,
            f"metrics begin after {series.initial_ticks} received ticks",
        )


def _audit_episode_totals(
    final: dict[str, Any], audit: Audit, series: _TrainingSeries
) -> None:
    if series.initial_ticks != 0:
        return
    audit.check(
        "episodes.cumulative_count",
        final.get("cumulative_received_completed_episodes") == len(series.episodes),
    )
    audit.check(
        "episodes.cumulative_score_sum",
        final.get("cumulative_received_score_sum")
        == sum(int(episode["score"]) for episode in series.episodes),
    )
    final_thresholds = final.get("cumulative_received_score_thresholds")
    audit.check(
        "episodes.cumulative_thresholds_object", isinstance(final_thresholds, dict)
    )
    if isinstance(final_thresholds, dict):
        for threshold in SCORE_THRESHOLDS:
            expected_count = sum(
                int(episode["score"]) >= threshold for episode in series.episodes
            )
            actual = final_thresholds.get(
                str(threshold), final_thresholds.get(threshold)
            )
            audit.check(
                f"episodes.cumulative_score_ge_{threshold}", actual == expected_count
            )
    expected_terminations = Counter(
        str(episode.get("termination")) for episode in series.episodes
    )
    final_terminations = final.get("cumulative_received_termination_counts")
    audit.check(
        "episodes.cumulative_terminations_object",
        isinstance(final_terminations, dict),
    )
    if isinstance(final_terminations, dict):
        all_reasons = set(expected_terminations) | set(final_terminations)
        audit.check(
            "episodes.cumulative_terminations",
            all(
                int(final_terminations.get(reason, 0))
                == expected_terminations.get(reason, 0)
                for reason in all_reasons
            ),
        )


def _food_window(
    positions: list[int],
    *,
    name: str,
    lower_exclusive: int,
    upper_inclusive: int,
    final_ticks: Any,
    exact_history: bool,
) -> dict[str, Any]:
    count = sum(lower_exclusive < position <= upper_inclusive for position in positions)
    available_ticks = 0
    if isinstance(final_ticks, int):
        available_ticks = max(
            0,
            min(final_ticks, upper_inclusive) - lower_exclusive,
        )
    window_ticks = upper_inclusive - lower_exclusive
    complete = exact_history and available_ticks == window_ticks
    return {
        "name": name,
        "interval": f"({lower_exclusive}, {upper_inclusive}]",
        "food_events": count,
        "window_ticks": window_ticks,
        "observed_ticks": available_ticks,
        "complete": complete,
        "food_per_10k_received_logic_ticks": (
            10_000 * count / window_ticks if complete else None
        ),
    }


def _food_windows(
    series: _TrainingSeries, final_ticks: Any, audit: Audit
) -> dict[str, dict[str, Any]]:
    common = {
        "positions": series.food_positions,
        "final_ticks": final_ticks,
        "exact_history": series.initial_ticks == 0,
    }
    early = _food_window(
        name="0-250k", lower_exclusive=0, upper_inclusive=250_000, **common
    )
    late = _food_window(
        name="750k-1M",
        lower_exclusive=750_000,
        upper_inclusive=1_000_000,
        **common,
    )
    audit.check("food.early_window_complete", early["complete"])
    audit.check("food.late_window_complete", late["complete"])
    return {"early_0_250k": early, "late_750k_1m": late}


def _assemble_training_summary(
    rows: list[dict[str, Any]],
    series: _TrainingSeries,
    traces: _CreditTraceSeries,
    *,
    expected_budget: int,
    final_ticks: Any,
    windows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    final = rows[-1]
    return {
        "metric_rows": len(rows),
        "first_training_iteration": rows[0].get("training_iteration"),
        "final_training_iteration": final.get("training_iteration"),
        "initial_received_logic_ticks": series.initial_ticks,
        "final_received_logic_ticks": final_ticks,
        "initial_received_transitions": series.initial_transitions,
        "final_received_transitions": final.get("cumulative_received_transitions"),
        "budget": {
            "target_received_logic_ticks": expected_budget,
            "reached": isinstance(final_ticks, int) and final_ticks >= expected_budget,
            "overshoot": max(0, final_ticks - expected_budget)
            if isinstance(final_ticks, int)
            else None,
        },
        "food": {
            "events_in_metrics": len(series.food_positions),
            "initial_cumulative_food_events": series.initial_food,
            "final_cumulative_food_events": series.previous_food_count,
            "windows": windows,
        },
        "received_episode_outcomes": _episode_summary(series.episodes),
        "credit_traces": traces.summary(),
        "numeric_finiteness": {
            "all_finite": not series.finite_issues,
            "nonfinite_paths": series.finite_issues,
        },
    }


def _training_summary(
    rows: list[dict[str, Any]], audit: Audit, *, expected_budget: int
) -> dict[str, Any]:
    """Validate and reduce an ordered metrics series into the public JSON schema."""

    series = _TrainingSeries()
    traces = _CreditTraceSeries()
    for row_index, row in enumerate(rows):
        _audit_training_row(row, row_index, audit, series, traces)

    audit.check(
        "metrics.numeric_finiteness",
        not series.finite_issues,
        ", ".join(series.finite_issues[:20]),
    )
    audit.check(
        "metrics.global_food_positions_unique",
        len(series.food_positions) == len(set(series.food_positions)),
    )
    final = rows[-1]
    final_ticks = _audit_budget(final, audit, expected_budget=expected_budget)
    _audit_exact_food_history(audit, series)
    _audit_episode_totals(final, audit, series)
    windows = _food_windows(series, final_ticks, audit)
    return _assemble_training_summary(
        rows,
        series,
        traces,
        expected_budget=expected_budget,
        final_ticks=final_ticks,
        windows=windows,
    )


def _preferred_validation_mean(run: dict[str, Any]) -> float | None:
    groups = run.get("validation", {}).get("groups", {})
    for name in ("deterministic_31x20", "stochastic_31x20"):
        value = groups.get(name, {}).get("score", {}).get("mean")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def compare_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return raw, non-inferential comparisons for named run directories."""

    source_digests = {
        name: run.get("config", {}).get("source_manifest_sha256")
        for name, run in runs.items()
    }
    nonnull_digests = {value for value in source_digests.values() if value is not None}
    exact_six = set(runs) == set(FORMAL_ARMS)
    contrasts = []
    if exact_six:
        for name, treatment, control in FORMAL_CONTRASTS:
            treatment_run = runs[treatment]
            control_run = runs[control]
            treatment_food = treatment_run["training"]["food"]["windows"]
            control_food = control_run["training"]["food"]["windows"]
            treatment_early = treatment_food["early_0_250k"][
                "food_per_10k_received_logic_ticks"
            ]
            treatment_late = treatment_food["late_750k_1m"][
                "food_per_10k_received_logic_ticks"
            ]
            control_early = control_food["early_0_250k"][
                "food_per_10k_received_logic_ticks"
            ]
            control_late = control_food["late_750k_1m"][
                "food_per_10k_received_logic_ticks"
            ]
            contrasts.append(
                {
                    "name": name,
                    "treatment": treatment,
                    "control": control,
                    "late_food_per_10k_delta": (
                        treatment_late - control_late
                        if treatment_late is not None and control_late is not None
                        else None
                    ),
                    "within_run_food_change_delta": (
                        (treatment_late - treatment_early)
                        - (control_late - control_early)
                        if None
                        not in (
                            treatment_early,
                            treatment_late,
                            control_early,
                            control_late,
                        )
                        else None
                    ),
                    "preferred_validation_mean_score_delta": (
                        _preferred_validation_mean(treatment_run)
                        - _preferred_validation_mean(control_run)
                        if _preferred_validation_mean(treatment_run) is not None
                        and _preferred_validation_mean(control_run) is not None
                        else None
                    ),
                }
            )
    return {
        "run_names": list(runs),
        "all_runs_valid": all(
            run.get("integrity", {}).get("valid") is True for run in runs.values()
        ),
        "source_manifest_sha256_by_run": source_digests,
        "same_source_manifest": len(nonnull_digests) == 1
        and len(nonnull_digests) == len(set(source_digests.values())),
        "formal_six_arm_set_complete": exact_six,
        "formal_six_arm_missing": sorted(set(FORMAL_ARMS) - set(runs)),
        "formal_six_arm_unexpected": sorted(set(runs) - set(FORMAL_ARMS)),
        "contrasts": contrasts,
        "interpretation": (
            "Descriptive audit only; deltas are not evidence of learning or a "
            "causal effect without the pre-registered multi-seed analysis."
        ),
    }

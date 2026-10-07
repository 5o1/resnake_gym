"""Per-iteration population splitting and V-trace metric construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from resnake_gym.training.vtrace_metrics import RunMetricsState, episode_metrics


class LearnerProgress(Protocol):
    logic_ticks: int
    update_count: int


class AssemblerProgress(Protocol):
    cumulative_received_logic_ticks: int
    cumulative_received_transitions: int


@dataclass(frozen=True, slots=True)
class CollectionPopulations:
    """Collector metrics with received and trained episode sets made explicit."""

    metrics: dict[str, Any]
    received_episodes: list[dict[str, Any]]
    trained_episodes: list[dict[str, Any]]

    @classmethod
    def split(cls, collection: Mapping[str, Any]) -> CollectionPopulations:
        """Remove episode payloads while checking the legacy trained alias."""
        metrics = dict(collection)
        legacy_trained = metrics.pop("fresh_episodes")
        trained = metrics.pop("fresh_trained_episodes")
        received = metrics.pop("fresh_received_episodes")
        if legacy_trained != trained:
            raise RuntimeError("trained episode compatibility alias diverged")
        return cls(
            metrics=metrics,
            received_episodes=list(received),
            trained_episodes=list(trained),
        )


def build_iteration_metrics(
    *,
    iteration: int,
    collection: Mapping[str, Any],
    learner_metrics: Mapping[str, Any],
    populations: CollectionPopulations,
    run_metrics: RunMetricsState,
    learner: LearnerProgress,
    assembler: AssemblerProgress,
    published_policy_version: int,
    wall_seconds: float,
    total_wall_seconds: float,
    max_fresh_logic_ticks: int | None,
    conservation_error: tuple[int, int],
) -> dict[str, Any]:
    """Build the stable JSON metric schema for one optimizer iteration.

    ``run_metrics.observe`` remains an explicit caller responsibility.  This
    function is deliberately pure so rendering or retrying a log record cannot
    double-count received experience.
    """
    received_episodes = populations.received_episodes
    trained_episodes = populations.trained_episodes
    received_ticks = int(collection["fresh_received_logic_ticks"])
    trained_ticks = int(collection["fresh_logic_ticks"])
    cumulative_received_ticks = assembler.cumulative_received_logic_ticks
    tick_error, transition_error = conservation_error
    metrics = {
        "training_iteration": iteration,
        **collection,
        **learner_metrics,
        # Unprefixed outcome metrics name the primary received-tick
        # population. Explicit copies prevent cross-population ratios.
        **episode_metrics(received_episodes),
        **episode_metrics(received_episodes, prefix="received_"),
        **episode_metrics(trained_episodes, prefix="trained_"),
        "published_policy_version": published_policy_version,
        "wall_seconds": wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "received_logic_ticks_per_second": received_ticks / max(wall_seconds, 1e-9),
        "fresh_logic_ticks_per_second": received_ticks / max(wall_seconds, 1e-9),
        "trained_logic_ticks_per_second": trained_ticks / max(wall_seconds, 1e-9),
        **run_metrics.cumulative_metrics(
            received_logic_ticks=cumulative_received_ticks,
            trained_logic_ticks=learner.logic_ticks,
        ),
        "best_fragment_score": run_metrics.best_score,
        "best_trace_score": run_metrics.best_score,
        "max_fresh_logic_ticks": max_fresh_logic_ticks,
        "fresh_logic_tick_budget_reached": (
            max_fresh_logic_ticks is not None
            and cumulative_received_ticks >= max_fresh_logic_ticks
        ),
        "fresh_logic_tick_budget_overshoot": (
            max(0, cumulative_received_ticks - max_fresh_logic_ticks)
            if max_fresh_logic_ticks is not None
            else 0
        ),
        "episodes": received_episodes,
        "trained_episodes": trained_episodes,
        "cumulative_fresh_logic_ticks": cumulative_received_ticks,
        "cumulative_fresh_transitions": (assembler.cumulative_received_transitions),
        "received_trained_pending_ready_tick_error": tick_error,
        "received_trained_pending_ready_transition_error": transition_error,
        "received_trained_pending_ready_conserved": True,
    }
    if received_ticks:
        received_food_rate = (
            10_000 * int(collection["fresh_received_food_count"]) / received_ticks
        )
        metrics["food_per_10k_logic_ticks"] = received_food_rate
        metrics["received_food_per_10k_logic_ticks"] = received_food_rate
    if trained_ticks:
        metrics["trained_food_per_10k_logic_ticks"] = (
            10_000 * int(collection["fresh_trained_food_count"]) / trained_ticks
        )
    metrics["optimizer_updates"] = learner.update_count
    for threshold, count in run_metrics.received_score_thresholds.items():
        metrics[f"score_ge_{threshold}_count"] = sum(
            episode["score"] >= int(threshold) for episode in received_episodes
        )
        metrics[f"cumulative_score_ge_{threshold}_count"] = count
        metrics[f"cumulative_score_ge_{threshold}_rate"] = (
            count / run_metrics.received_episodes
            if run_metrics.received_episodes
            else 0.0
        )
    return metrics


__all__ = ["CollectionPopulations", "build_iteration_metrics"]

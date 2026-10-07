"""Metric definitions and persistent counters for V-trace training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

METRICS_STATE_FORMAT = "received-vs-trained-observability-v1"
TERMINATION_REASONS = (
    "wall_collision",
    "self_collision",
    "obstacle_collision",
    "time_limit",
    "board_filled",
)
DEATH_REASONS = frozenset(("wall_collision", "self_collision", "obstacle_collision"))
SCORE_THRESHOLDS = (1, 2, 5, 10, 25, 50, 100, 250)


def percentile(values: Iterable[int | float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without requiring NumPy."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a percentile of no values")
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def episode_metrics(
    episodes: Iterable[Mapping[str, Any]], *, prefix: str = ""
) -> dict[str, int | float]:
    """Summarize one explicitly named received or trained population."""
    episode_rows = list(episodes)
    result: dict[str, int | float] = {
        f"{prefix}completed_episodes": len(episode_rows),
        f"{prefix}strict_wins": sum(bool(row["won"]) for row in episode_rows),
    }
    if not episode_rows:
        return result

    scores = [int(row["score"]) for row in episode_rows]
    lifetimes = [
        int(row.get("ticks", row.get("controller_tick", 0))) for row in episode_rows
    ]
    result.update(
        {
            f"{prefix}episode_score_mean": sum(scores) / len(scores),
            f"{prefix}episode_score_max": max(scores),
            f"{prefix}episode_score_p50": percentile(scores, 50),
            f"{prefix}episode_score_p90": percentile(scores, 90),
            f"{prefix}episode_score_p99": percentile(scores, 99),
            f"{prefix}episode_lifetime_ticks_mean": sum(lifetimes) / len(lifetimes),
            f"{prefix}episode_lifetime_ticks_p50": percentile(lifetimes, 50),
            f"{prefix}episode_lifetime_ticks_p90": percentile(lifetimes, 90),
            f"{prefix}episode_lifetime_ticks_p99": percentile(lifetimes, 99),
            f"{prefix}episode_lifetime_ticks_max": max(lifetimes),
            f"{prefix}win_rate": result[f"{prefix}strict_wins"] / len(episode_rows),
        }
    )
    observed_reasons = {
        str(row.get("termination"))
        for row in episode_rows
        if row.get("termination") is not None
    }
    for reason in sorted(set(TERMINATION_REASONS) | observed_reasons):
        count = sum(row.get("termination") == reason for row in episode_rows)
        result[f"{prefix}termination_{reason}_count"] = count
        result[f"{prefix}termination_{reason}_rate"] = count / len(episode_rows)
    deaths = sum(row.get("termination") in DEATH_REASONS for row in episode_rows)
    result[f"{prefix}death_count"] = deaths
    result[f"{prefix}death_rate"] = deaths / len(episode_rows)
    return result


@dataclass(slots=True)
class RunMetricsState:
    """Counters that must survive a trainer restart.

    The state distinguishes receipt-side environment progress from the subset
    already consumed by the learner.  Legacy aliases are accepted and emitted
    only at the checkpoint boundary, not throughout the training loop.
    """

    format: ClassVar[str] = METRICS_STATE_FORMAT
    best_score: int = -1
    received_episodes: int = 0
    received_wins: int = 0
    received_food_count: int = 0
    trained_food_count: int = 0
    received_score_sum: int = 0
    received_lifetime_ticks: int = 0
    received_termination_counts: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in TERMINATION_REASONS}
    )
    received_score_thresholds: dict[str, int] = field(
        default_factory=lambda: {str(value): 0 for value in SCORE_THRESHOLDS}
    )

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> RunMetricsState:
        """Restore current state while accepting pre-explicit naming aliases."""
        if payload.get("metrics_state_format") != cls.format:
            raise ValueError(
                "resume checkpoint predates explicit received/trained metric state"
            )
        terminations = {reason: 0 for reason in TERMINATION_REASONS}
        terminations.update(
            {
                str(reason): int(count)
                for reason, count in payload.get(
                    "cumulative_received_termination_counts", {}
                ).items()
            }
        )
        thresholds = payload.get(
            "cumulative_received_score_thresholds",
            payload.get(
                "cumulative_score_thresholds",
                {str(value): 0 for value in SCORE_THRESHOLDS},
            ),
        )
        return cls(
            best_score=int(payload.get("best_score", -1)),
            received_episodes=int(
                payload.get(
                    "cumulative_received_episodes",
                    payload.get("cumulative_episodes", 0),
                )
            ),
            received_wins=int(
                payload.get(
                    "cumulative_received_wins",
                    payload.get("cumulative_wins", 0),
                )
            ),
            received_food_count=int(payload.get("cumulative_received_food_count", 0)),
            trained_food_count=int(payload.get("cumulative_trained_food_count", 0)),
            received_score_sum=int(payload.get("cumulative_received_score_sum", 0)),
            received_lifetime_ticks=int(
                payload.get("cumulative_received_lifetime_ticks", 0)
            ),
            received_termination_counts=terminations,
            received_score_thresholds={
                str(threshold): int(count) for threshold, count in thresholds.items()
            },
        )

    def observe(
        self,
        received_episodes: Iterable[Mapping[str, Any]],
        *,
        received_food_count: int,
        trained_food_count: int,
    ) -> None:
        """Accumulate one collector batch exactly once."""
        episodes = list(received_episodes)
        self.received_episodes += len(episodes)
        self.received_wins += sum(bool(row["won"]) for row in episodes)
        self.received_food_count += int(received_food_count)
        self.trained_food_count += int(trained_food_count)
        self.received_score_sum += sum(int(row["score"]) for row in episodes)
        self.received_lifetime_ticks += sum(
            int(row.get("ticks", row.get("controller_tick", 0))) for row in episodes
        )
        for row in episodes:
            reason = str(row.get("termination"))
            self.received_termination_counts[reason] = (
                self.received_termination_counts.get(reason, 0) + 1
            )
        for threshold in self.received_score_thresholds:
            self.received_score_thresholds[threshold] += sum(
                int(row["score"]) >= int(threshold) for row in episodes
            )

    def checkpoint_payload(self) -> dict[str, Any]:
        """Serialize current names plus compatibility aliases for old readers."""
        return {
            "metrics_state_format": self.format,
            "best_score": self.best_score,
            "cumulative_episodes": self.received_episodes,
            "cumulative_wins": self.received_wins,
            "cumulative_received_episodes": self.received_episodes,
            "cumulative_received_wins": self.received_wins,
            "cumulative_received_food_count": self.received_food_count,
            "cumulative_trained_food_count": self.trained_food_count,
            "cumulative_received_score_sum": self.received_score_sum,
            "cumulative_received_lifetime_ticks": self.received_lifetime_ticks,
            "cumulative_received_termination_counts": dict(
                self.received_termination_counts
            ),
            "cumulative_score_thresholds": dict(self.received_score_thresholds),
            "cumulative_received_score_thresholds": dict(
                self.received_score_thresholds
            ),
        }

    def cumulative_metrics(
        self, *, received_logic_ticks: int, trained_logic_ticks: int
    ) -> dict[str, int | float | dict[str, int]]:
        """Build flat JSON metrics from the persistent counters."""
        metrics: dict[str, int | float | dict[str, int]] = {
            "cumulative_completed_episodes": self.received_episodes,
            "cumulative_received_completed_episodes": self.received_episodes,
            "cumulative_strict_wins": self.received_wins,
            "cumulative_received_strict_wins": self.received_wins,
            "cumulative_received_food_count": self.received_food_count,
            "cumulative_trained_food_count": self.trained_food_count,
            "cumulative_score_thresholds": dict(self.received_score_thresholds),
            "cumulative_received_score_thresholds": dict(
                self.received_score_thresholds
            ),
            "cumulative_received_score_sum": self.received_score_sum,
            "cumulative_received_lifetime_ticks": self.received_lifetime_ticks,
            "cumulative_received_termination_counts": dict(
                self.received_termination_counts
            ),
        }
        if received_logic_ticks:
            rate = 10_000 * self.received_food_count / received_logic_ticks
            metrics["cumulative_food_per_10k_logic_ticks"] = rate
            metrics["cumulative_received_food_per_10k_logic_ticks"] = rate
        if trained_logic_ticks:
            metrics["cumulative_trained_food_per_10k_logic_ticks"] = (
                10_000 * self.trained_food_count / trained_logic_ticks
            )
        if self.received_episodes:
            metrics["cumulative_received_episode_score_mean"] = (
                self.received_score_sum / self.received_episodes
            )
            metrics["cumulative_received_episode_lifetime_ticks_mean"] = (
                self.received_lifetime_ticks / self.received_episodes
            )
            metrics["cumulative_received_win_rate"] = (
                self.received_wins / self.received_episodes
            )
            for reason, count in sorted(self.received_termination_counts.items()):
                metrics[f"cumulative_received_termination_{reason}_count"] = count
                metrics[f"cumulative_received_termination_{reason}_rate"] = (
                    count / self.received_episodes
                )
        return metrics


__all__ = [
    "DEATH_REASONS",
    "METRICS_STATE_FORMAT",
    "RunMetricsState",
    "SCORE_THRESHOLDS",
    "TERMINATION_REASONS",
    "episode_metrics",
    "percentile",
]

from dataclasses import dataclass

import pytest

from resnake_gym.training.vtrace_metrics import RunMetricsState, episode_metrics
from resnake_gym.training.vtrace_reporting import (
    CollectionPopulations,
    build_iteration_metrics,
)


@dataclass
class FakeLearner:
    logic_ticks: int = 10
    update_count: int = 4


@dataclass
class FakeAssembler:
    cumulative_received_logic_ticks: int = 20
    cumulative_received_transitions: int = 8


def _episode(score, ticks, termination):
    return {
        "score": score,
        "ticks": ticks,
        "won": False,
        "termination": termination,
    }


def _pre_extraction_metric_record(
    *, populations, state, learner, assembler, collection
):
    """Reference the metric block that previously lived in the CLI script."""
    received = populations.received_episodes
    trained = populations.trained_episodes
    result = {
        "training_iteration": 2,
        **collection,
        "loss": 0.5,
        **episode_metrics(received),
        **episode_metrics(received, prefix="received_"),
        **episode_metrics(trained, prefix="trained_"),
        "published_policy_version": 4,
        "wall_seconds": 2.0,
        "total_wall_seconds": 5.0,
        "received_logic_ticks_per_second": (
            collection["fresh_received_logic_ticks"] / 2.0
        ),
        "fresh_logic_ticks_per_second": (
            collection["fresh_received_logic_ticks"] / 2.0
        ),
        "trained_logic_ticks_per_second": collection["fresh_logic_ticks"] / 2.0,
        **state.cumulative_metrics(
            received_logic_ticks=assembler.cumulative_received_logic_ticks,
            trained_logic_ticks=learner.logic_ticks,
        ),
        "best_fragment_score": state.best_score,
        "best_trace_score": state.best_score,
        "max_fresh_logic_ticks": 19,
        "fresh_logic_tick_budget_reached": True,
        "fresh_logic_tick_budget_overshoot": 1,
        "episodes": received,
        "trained_episodes": trained,
        "cumulative_fresh_logic_ticks": assembler.cumulative_received_logic_ticks,
        "cumulative_fresh_transitions": assembler.cumulative_received_transitions,
        "received_trained_pending_ready_tick_error": 0,
        "received_trained_pending_ready_transition_error": 0,
        "received_trained_pending_ready_conserved": True,
    }
    received_food_rate = (
        10_000
        * collection["fresh_received_food_count"]
        / collection["fresh_received_logic_ticks"]
    )
    result["food_per_10k_logic_ticks"] = received_food_rate
    result["received_food_per_10k_logic_ticks"] = received_food_rate
    result["trained_food_per_10k_logic_ticks"] = (
        10_000
        * collection["fresh_trained_food_count"]
        / collection["fresh_logic_ticks"]
    )
    result["optimizer_updates"] = learner.update_count
    for threshold, count in state.received_score_thresholds.items():
        result[f"score_ge_{threshold}_count"] = sum(
            row["score"] >= int(threshold) for row in received
        )
        result[f"cumulative_score_ge_{threshold}_count"] = count
        result[f"cumulative_score_ge_{threshold}_rate"] = (
            count / state.received_episodes if state.received_episodes else 0.0
        )
    return result


def test_collection_populations_separates_received_and_trained_without_mutation():
    trained = [_episode(1, 8, "wall_collision")]
    received = [_episode(3, 12, "time_limit")]
    raw = {
        "fresh_episodes": trained,
        "fresh_trained_episodes": list(trained),
        "fresh_received_episodes": received,
        "fresh_logic_ticks": 10,
        "fresh_received_logic_ticks": 20,
        "fresh_trained_food_count": 1,
        "fresh_received_food_count": 2,
    }

    populations = CollectionPopulations.split(raw)

    assert populations.received_episodes == received
    assert populations.trained_episodes == trained
    assert "fresh_episodes" not in populations.metrics
    assert "fresh_trained_episodes" not in populations.metrics
    assert "fresh_received_episodes" not in populations.metrics
    assert raw["fresh_episodes"] == trained


def test_collection_populations_rejects_divergent_compatibility_alias():
    with pytest.raises(RuntimeError, match="compatibility alias diverged"):
        CollectionPopulations.split(
            {
                "fresh_episodes": [],
                "fresh_trained_episodes": [_episode(0, 1, "time_limit")],
                "fresh_received_episodes": [],
            }
        )


def test_iteration_metrics_preserve_received_trained_and_budget_semantics():
    populations = CollectionPopulations.split(
        {
            "fresh_episodes": [_episode(1, 8, "wall_collision")],
            "fresh_trained_episodes": [_episode(1, 8, "wall_collision")],
            "fresh_received_episodes": [_episode(3, 12, "time_limit")],
            "fresh_logic_ticks": 10,
            "fresh_received_logic_ticks": 20,
            "fresh_trained_food_count": 1,
            "fresh_received_food_count": 2,
            "collector_marker": 7,
        }
    )
    state = RunMetricsState()
    state.observe(
        populations.received_episodes,
        received_food_count=2,
        trained_food_count=1,
    )
    state.best_score = 3
    before = state.checkpoint_payload()

    metrics = build_iteration_metrics(
        iteration=2,
        collection=populations.metrics,
        learner_metrics={"loss": 0.5},
        populations=populations,
        run_metrics=state,
        learner=FakeLearner(),
        assembler=FakeAssembler(),
        published_policy_version=4,
        wall_seconds=2.0,
        total_wall_seconds=5.0,
        max_fresh_logic_ticks=19,
        conservation_error=(0, 0),
    )
    expected = _pre_extraction_metric_record(
        populations=populations,
        state=state,
        learner=FakeLearner(),
        assembler=FakeAssembler(),
        collection=populations.metrics,
    )

    assert state.checkpoint_payload() == before
    assert metrics == expected
    assert metrics["collector_marker"] == 7
    assert metrics["loss"] == 0.5
    assert metrics["completed_episodes"] == 1
    assert metrics["episode_score_mean"] == 3
    assert metrics["received_episode_score_mean"] == 3
    assert metrics["trained_episode_score_mean"] == 1
    assert metrics["received_logic_ticks_per_second"] == 10
    assert metrics["fresh_logic_ticks_per_second"] == 10
    assert metrics["trained_logic_ticks_per_second"] == 5
    assert metrics["food_per_10k_logic_ticks"] == 1000
    assert metrics["received_food_per_10k_logic_ticks"] == 1000
    assert metrics["trained_food_per_10k_logic_ticks"] == 1000
    assert metrics["cumulative_received_food_per_10k_logic_ticks"] == 1000
    assert metrics["cumulative_trained_food_per_10k_logic_ticks"] == 1000
    assert metrics["score_ge_2_count"] == 1
    assert metrics["cumulative_score_ge_2_count"] == 1
    assert metrics["cumulative_score_ge_2_rate"] == 1
    assert metrics["fresh_logic_tick_budget_reached"] is True
    assert metrics["fresh_logic_tick_budget_overshoot"] == 1
    assert metrics["received_trained_pending_ready_conserved"] is True
    assert metrics["episodes"] == populations.received_episodes
    assert metrics["trained_episodes"] == populations.trained_episodes

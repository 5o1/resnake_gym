import pytest

from resnake_gym.training.vtrace_metrics import (
    METRICS_STATE_FORMAT,
    RunMetricsState,
    episode_metrics,
)


def test_episode_metrics_report_quantiles_lifetime_and_death_reasons():
    metrics = episode_metrics(
        [
            {
                "score": 0,
                "ticks": 10,
                "won": False,
                "termination": "wall_collision",
            },
            {
                "score": 3,
                "ticks": 40,
                "won": False,
                "termination": "time_limit",
            },
        ],
        prefix="received_",
    )

    assert metrics["received_completed_episodes"] == 2
    assert metrics["received_episode_score_p50"] == 1.5
    assert metrics["received_episode_score_p90"] == pytest.approx(2.7)
    assert metrics["received_episode_score_p99"] == pytest.approx(2.97)
    assert metrics["received_episode_lifetime_ticks_mean"] == 25
    assert metrics["received_episode_lifetime_ticks_p90"] == pytest.approx(37)
    assert metrics["received_termination_wall_collision_count"] == 1
    assert metrics["received_termination_time_limit_count"] == 1
    assert metrics["received_death_count"] == 1
    assert metrics["received_death_rate"] == 0.5


def test_run_metrics_state_round_trips_and_keeps_populations_separate():
    state = RunMetricsState()
    state.observe(
        [
            {
                "score": 3,
                "ticks": 40,
                "won": False,
                "termination": "time_limit",
            }
        ],
        received_food_count=3,
        trained_food_count=1,
    )
    state.best_score = 3

    payload = state.checkpoint_payload()
    restored = RunMetricsState.restore(payload)

    assert payload["metrics_state_format"] == METRICS_STATE_FORMAT
    assert restored == state
    metrics = restored.cumulative_metrics(
        received_logic_ticks=100,
        trained_logic_ticks=50,
    )
    assert metrics["cumulative_received_food_per_10k_logic_ticks"] == 300
    assert metrics["cumulative_trained_food_per_10k_logic_ticks"] == 200
    assert metrics["cumulative_received_episode_score_mean"] == 3


def test_run_metrics_state_rejects_unknown_nonempty_format():
    with pytest.raises(ValueError, match="predates explicit"):
        RunMetricsState.restore({"metrics_state_format": "unknown"})

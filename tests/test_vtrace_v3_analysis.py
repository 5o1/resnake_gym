import json
import runpy
from pathlib import Path

from resnake_gym.analysis import REPORT_FORMAT, analyze_run, compare_runs
from resnake_gym.analysis.artifacts import Audit
from resnake_gym.analysis.training import _food_window, _training_summary


def _write_minimal_valid_run(run_directory: Path) -> None:
    run_directory.mkdir()
    config = {
        "format": "gamepad-vtrace-v3",
        "collection_semantics": "fifo-stitched-credit-trace-v3",
        "recurrent_state": "stored-state-burnin-window-tbptt-v5",
        "metric_semantics": "received-vs-trained-observability-v1",
        "training_objective": "snake-dpad5-event-trace-vtrace-v2",
        "action_encoding": "xinput-dpad5-categorical-v1",
        "fragment_replay_training_enabled": False,
        "training_budget": {
            "primary_axis": "cumulative_received_logic_ticks",
            "max_fresh_logic_ticks": 1_000_000,
        },
        "source_sha256": {"src/example.py": "0" * 64},
        "config": {
            "seed": 7,
            "width": 31,
            "height": 20,
            "action_head": "dpad5",
            "entropy_coefficient": 0.001,
            "batch_min_transitions": 512,
            "credit_trace_max_transitions": 128,
            "bptt_window": 128,
        },
    }
    row = {
        "training_iteration": 1,
        "cumulative_received_logic_ticks": 1_000_000,
        "fresh_received_logic_ticks": 1_000_000,
        "cumulative_received_transitions": 10,
        "fresh_received_transitions": 10,
        "cumulative_fresh_logic_ticks": 1_000_000,
        "cumulative_fresh_transitions": 10,
        "fresh_received_food_tick_positions": [1, 250_000, 750_001, 1_000_000],
        "fresh_received_food_count": 4,
        "cumulative_received_food_count": 4,
        "cumulative_trained_logic_ticks": 1_000_000,
        "pending_credit_logic_ticks": 0,
        "ready_credit_logic_ticks": 0,
        "cumulative_trained_transitions": 10,
        "pending_credit_transitions": 0,
        "ready_credit_transitions": 0,
        "received_trained_pending_ready_tick_error": 0,
        "received_trained_pending_ready_transition_error": 0,
        "received_trained_pending_ready_conserved": True,
        "episodes": [
            {
                "score": 2,
                "won": False,
                "ticks": 50,
                "termination": "wall_collision",
            }
        ],
        "credit_trace_count": 1,
        "credit_trace_over_128_count": 0,
        "credit_trace_over_128_rate": 0.0,
        "credit_trace_len_mean": 10.0,
        "credit_trace_len_max": 10,
        "credit_trace_cap_hit_count": 0,
        "credit_artificial_boundary_count": 0,
        "credit_bootstrap_boundary_count": 0,
        "max_fresh_logic_ticks": 1_000_000,
        "fresh_logic_tick_budget_overshoot": 0,
        "fresh_logic_tick_budget_reached": True,
        "cumulative_received_completed_episodes": 1,
        "cumulative_received_score_sum": 2,
        "cumulative_received_score_thresholds": {
            "1": 1,
            "2": 1,
            "5": 0,
            "10": 0,
            "25": 0,
            "50": 0,
            "100": 0,
            "250": 0,
        },
        "cumulative_received_termination_counts": {"wall_collision": 1},
    }
    (run_directory / "config.json").write_text(json.dumps(config))
    (run_directory / "metrics.jsonl").write_text(json.dumps(row) + "\n")


def test_analyze_run_minimal_valid_fixture_preserves_schema(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_minimal_valid_run(run_directory)

    result = analyze_run(
        run_directory,
        expected_budget=1_000_000,
        source_root=None,
        require_validation=False,
    )

    assert set(result) == {
        "run",
        "arm",
        "config",
        "training",
        "validation",
        "integrity",
    }
    assert result["integrity"]["valid"] is True
    assert result["training"]["food"]["windows"]["early_0_250k"] == {
        "name": "0-250k",
        "interval": "(0, 250000]",
        "food_events": 2,
        "window_ticks": 250_000,
        "observed_ticks": 250_000,
        "complete": True,
        "food_per_10k_received_logic_ticks": 0.08,
    }
    assert (
        result["training"]["food"]["windows"]["late_750k_1m"][
            "food_per_10k_received_logic_ticks"
        ]
        == 0.08
    )
    assert result["training"]["received_episode_outcomes"]["death"]["count"] == 1
    comparison = compare_runs({"run": result})
    assert comparison["all_runs_valid"] is True
    assert comparison["formal_six_arm_set_complete"] is False
    assert REPORT_FORMAT == "resnake-vtrace-v3-offline-analysis-v1"


def test_analyze_run_reports_unreadable_artifact(tmp_path: Path) -> None:
    run_directory = tmp_path / "broken"
    run_directory.mkdir()
    (run_directory / "config.json").write_text("{}")
    (run_directory / "metrics.jsonl").write_text("{not-json}\n")

    result = analyze_run(run_directory, require_validation=False)

    assert set(result) == {"run", "arm", "integrity"}
    assert result["integrity"]["valid"] is False
    assert result["integrity"]["checks"] == [
        {
            "name": "artifacts.readable",
            "ok": False,
            "detail": (
                f"{run_directory / 'metrics.jsonl'}:1: invalid JSON: "
                "Expecting property name enclosed in double quotes: "
                "line 1 column 2 (char 1)"
            ),
        }
    ]
    assert result["integrity"]["errors"] == [
        result["integrity"]["checks"][0]["name"]
        + ": "
        + result["integrity"]["checks"][0]["detail"]
    ]


def test_analyze_run_rejects_historical_nonzero_replay_config(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "replay"
    _write_minimal_valid_run(run_directory)
    config_path = run_directory / "config.json"
    metadata = json.loads(config_path.read_text())
    metadata["config"]["replay_batches_per_update"] = 1
    config_path.write_text(json.dumps(metadata))

    result = analyze_run(
        run_directory,
        source_root=None,
        require_validation=False,
    )

    assert result["integrity"]["valid"] is False
    assert "config.fragment_replay_disabled" in result["integrity"]["errors"]


def test_compatibility_script_keeps_cli_report_schema(
    tmp_path: Path,
    capsys,
) -> None:
    run_directory = tmp_path / "cli-run"
    _write_minimal_valid_run(run_directory)
    script = Path(__file__).parents[1] / "scripts" / "analyze_vtrace_v3_results.py"
    namespace = runpy.run_path(str(script))

    exit_code = namespace["main"](
        [
            str(run_directory),
            "--skip-local-source-check",
            "--allow-missing-validation",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert namespace["analyze_run"] is analyze_run
    assert set(report) == {"format", "window_semantics", "runs", "comparison"}
    assert report["format"] == REPORT_FORMAT


def test_training_summary_reduces_multiple_rows_without_schema_drift() -> None:
    first = {
        "training_iteration": 1,
        "cumulative_received_logic_ticks": 250_000,
        "fresh_received_logic_ticks": 250_000,
        "cumulative_received_transitions": 4,
        "fresh_received_transitions": 4,
        "cumulative_fresh_logic_ticks": 250_000,
        "cumulative_fresh_transitions": 4,
        "fresh_received_food_tick_positions": [1, 250_000],
        "fresh_received_food_count": 2,
        "cumulative_received_food_count": 2,
        "cumulative_trained_logic_ticks": 250_000,
        "pending_credit_logic_ticks": 0,
        "ready_credit_logic_ticks": 0,
        "cumulative_trained_transitions": 4,
        "pending_credit_transitions": 0,
        "ready_credit_transitions": 0,
        "received_trained_pending_ready_tick_error": 0,
        "received_trained_pending_ready_transition_error": 0,
        "received_trained_pending_ready_conserved": True,
        "episodes": [
            {
                "score": 2,
                "won": False,
                "ticks": 50,
                "termination": "wall_collision",
            }
        ],
        "credit_trace_count": 2,
        "credit_trace_over_128_count": 1,
        "credit_trace_over_128_rate": 0.5,
        "credit_trace_len_mean": 100.0,
        "credit_trace_len_max": 150,
        "credit_trace_cap_hit_count": 1,
        "credit_artificial_boundary_count": 0,
        "credit_bootstrap_boundary_count": 1,
    }
    final = {
        "training_iteration": 2,
        "cumulative_received_logic_ticks": 1_000_000,
        "fresh_received_logic_ticks": 750_000,
        "cumulative_received_transitions": 10,
        "fresh_received_transitions": 6,
        "cumulative_fresh_logic_ticks": 1_000_000,
        "cumulative_fresh_transitions": 10,
        "fresh_received_food_tick_positions": [750_001, 1_000_000],
        "fresh_received_food_count": 2,
        "cumulative_received_food_count": 4,
        "cumulative_trained_logic_ticks": 1_000_000,
        "pending_credit_logic_ticks": 0,
        "ready_credit_logic_ticks": 0,
        "cumulative_trained_transitions": 10,
        "pending_credit_transitions": 0,
        "ready_credit_transitions": 0,
        "received_trained_pending_ready_tick_error": 0,
        "received_trained_pending_ready_transition_error": 0,
        "received_trained_pending_ready_conserved": True,
        "episodes": [{"score": 5, "won": True, "ticks": 100, "termination": "win"}],
        "credit_trace_count": 1,
        "credit_trace_over_128_count": 0,
        "credit_trace_over_128_rate": 0.0,
        "credit_trace_len_mean": 10.0,
        "credit_trace_len_max": 10,
        "credit_trace_cap_hit_count": 0,
        "credit_artificial_boundary_count": 1,
        "credit_bootstrap_boundary_count": 0,
        "max_fresh_logic_ticks": 1_000_000,
        "fresh_logic_tick_budget_overshoot": 0,
        "fresh_logic_tick_budget_reached": True,
        "cumulative_received_completed_episodes": 2,
        "cumulative_received_score_sum": 7,
        "cumulative_received_score_thresholds": {
            "1": 2,
            "2": 2,
            "5": 1,
            "10": 0,
            "25": 0,
            "50": 0,
            "100": 0,
            "250": 0,
        },
        "cumulative_received_termination_counts": {
            "wall_collision": 1,
            "win": 1,
        },
    }
    audit = Audit()

    result = _training_summary([first, final], audit, expected_budget=1_000_000)

    assert result == {
        "metric_rows": 2,
        "first_training_iteration": 1,
        "final_training_iteration": 2,
        "initial_received_logic_ticks": 0,
        "final_received_logic_ticks": 1_000_000,
        "initial_received_transitions": 0,
        "final_received_transitions": 10,
        "budget": {
            "target_received_logic_ticks": 1_000_000,
            "reached": True,
            "overshoot": 0,
        },
        "food": {
            "events_in_metrics": 4,
            "initial_cumulative_food_events": 0,
            "final_cumulative_food_events": 4,
            "windows": {
                "early_0_250k": {
                    "name": "0-250k",
                    "interval": "(0, 250000]",
                    "food_events": 2,
                    "window_ticks": 250_000,
                    "observed_ticks": 250_000,
                    "complete": True,
                    "food_per_10k_received_logic_ticks": 0.08,
                },
                "late_750k_1m": {
                    "name": "750k-1M",
                    "interval": "(750000, 1000000]",
                    "food_events": 2,
                    "window_ticks": 250_000,
                    "observed_ticks": 250_000,
                    "complete": True,
                    "food_per_10k_received_logic_ticks": 0.08,
                },
            },
        },
        "received_episode_outcomes": {
            "episodes": 2,
            "wins": 1,
            "win_rate": 0.5,
            "score": {
                "mean": 3.5,
                "p50": 3.5,
                "p90": 4.7,
                "p99": 4.970000000000001,
                "max": 5,
            },
            "score_thresholds": {
                "1": {"count": 2, "rate": 1.0},
                "2": {"count": 2, "rate": 1.0},
                "5": {"count": 1, "rate": 0.5},
                "10": {"count": 0, "rate": 0.0},
                "25": {"count": 0, "rate": 0.0},
                "50": {"count": 0, "rate": 0.0},
                "100": {"count": 0, "rate": 0.0},
                "250": {"count": 0, "rate": 0.0},
            },
            "lifetime_ticks": {
                "mean": 75.0,
                "p50": 75.0,
                "p90": 95.0,
                "p99": 99.5,
                "max": 100,
            },
            "termination": {
                "wall_collision": {"count": 1, "rate": 0.5},
                "win": {"count": 1, "rate": 0.5},
            },
            "death": {"count": 1, "rate": 0.5},
        },
        "credit_traces": {
            "count": 3,
            "length_mean_weighted": 70.0,
            "length_max": 150,
            "over_128_count": 1,
            "over_128_rate": 1 / 3,
            "over_128_activated": True,
            "updates_with_over_128": 1,
            "cap_hit_count": 1,
            "artificial_boundary_count": 1,
            "bootstrap_boundary_count": 1,
        },
        "numeric_finiteness": {"all_finite": True, "nonfinite_paths": []},
    }
    assert audit.payload()["valid"] is True


def test_food_window_does_not_report_rate_for_partial_history() -> None:
    result = _food_window(
        [800_000],
        name="750k-1M",
        lower_exclusive=750_000,
        upper_inclusive=1_000_000,
        final_ticks=900_000,
        exact_history=True,
    )

    assert result == {
        "name": "750k-1M",
        "interval": "(750000, 1000000]",
        "food_events": 1,
        "window_ticks": 250_000,
        "observed_ticks": 150_000,
        "complete": False,
        "food_per_10k_received_logic_ticks": None,
    }

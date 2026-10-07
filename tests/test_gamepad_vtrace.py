import copy
import queue
import threading
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_runtime import make_env, stack_observations  # noqa: E402
from resnake_gym.gamepad_vtrace import (  # noqa: E402
    CreditTraceAssembler,
    GamepadVTraceLearner,
    VTraceConfig,
    _append_transition,
    _finish_fragment,
    _make_credit_trace,
    _new_buffer,
    actor_worker,
    checkpoint_metadata,
    collect_fresh_credit_traces,
    load_published_parameters,
    parameter_count,
    publish_parameters,
    validate_checkpoint,
    validate_credit_trace,
    validate_fragment,
)
from resnake_gym.models import (  # noqa: E402
    distribution_from_observation,
    policy_action_log_prob,
    reports_from_policy_action,
    sample_policy_action,
)
from resnake_gym.training.vtrace_learner_audit import (  # noqa: E402
    importance_effective_sample_size,
)
from resnake_gym.vtrace import vtrace_from_log_probs  # noqa: E402

EXPECTED_VTRACE_UPDATE_METRICS = {
    "actor_lag_max",
    "actor_lag_mean",
    "behavior_target_joint_kl_max",
    "behavior_target_joint_kl_mean",
    "bptt_backward_calls",
    "bptt_windows_per_update",
    "credit_artificial_boundary_count",
    "credit_artificial_boundary_rate",
    "credit_bootstrap_boundary_count",
    "credit_bootstrap_boundary_rate",
    "credit_trace_cap_hit_count",
    "credit_trace_cap_hit_rate",
    "credit_trace_count",
    "credit_trace_len_max",
    "credit_trace_len_mean",
    "credit_trace_len_p50",
    "credit_trace_len_p90",
    "credit_trace_len_p99",
    "credit_trace_over_128_count",
    "credit_trace_over_128_rate",
    "cumulative_trained_logic_ticks",
    "cumulative_trained_transitions",
    "entropy_action_variables_per_decision",
    "entropy_beta_h",
    "entropy_joint_nats",
    "entropy_normalized_fraction",
    "entropy_per_action_variable_nats",
    "grad_norm",
    "importance_ess",
    "importance_ess_fraction",
    "latent_entropy",
    "learner_fragments",
    "learner_logic_ticks",
    "learner_source",
    "learner_transitions",
    "learner_update",
    "loss",
    "mixed_policy_versions_max",
    "mixed_policy_versions_mean",
    "padding_fraction",
    "policy_loss",
    "recurrent_burn_in_max",
    "recurrent_hidden_drift_max",
    "recurrent_hidden_drift_mean",
    "rho_clipped_fraction",
    "rho_max",
    "rho_mean",
    "sample_behavior_target_log_ratio",
    "segments_per_trace_max",
    "segments_per_trace_mean",
    "target_policy_version",
    "trace_window_count",
    "transport_log_weight_mean",
    "transport_log_weight_min",
    "unique_behavior_versions_max",
    "unique_behavior_versions_mean",
    "unroll_max",
    "unroll_mean",
    "value_loss",
}


def _config(**updates):
    values = dict(
        width=31,
        height=20,
        dim=16,
        decision_min=1,
        decision_max=1,
        observation_delay_max=0,
        command_delay_max=0,
        drop_probability=0.0,
        max_logic_steps=100,
        perturbation_min=100,
        perturbation_max=100,
        spatial_pool=False,
        actor_processes=1,
        envs_per_actor=1,
        actor_sync_steps=2,
        unroll_length=2,
        recurrent_burn_in=2,
        bptt_window=2,
        credit_trace_max_transitions=8,
        batch_min_transitions=1,
        batch_max_transitions=4,
        batch_food_target=0,
        queue_capacity=4,
    )
    values.update(updates)
    return VTraceConfig(**values)


def _collect_fragment(learner, length, *, max_logic_steps=100, version=0):
    config = copy.deepcopy(learner.config)
    config.max_logic_steps = max_logic_steps
    environment = make_env(config)
    observation = environment.reset(seed=91 + length)[0]
    hidden = torch.zeros(1, config.dim, device=learner.device)
    buffer = _new_buffer(
        actor_id=0,
        env_id=0,
        episode_id=0,
        initial_hidden=hidden[0],
        score=0,
    )
    info = None
    try:
        for _ in range(length):
            obs_batch = stack_observations([observation], learner.device)
            with torch.no_grad():
                distribution, _, next_hidden = distribution_from_observation(
                    learner.model, obs_batch, hidden
                )
                action = sample_policy_action(distribution, stochastic=False)
                log_prob = policy_action_log_prob(distribution, action)
                report = reports_from_policy_action(distribution, action)
            next_observation, reward, terminated, truncated, info = environment.step(
                report[0].cpu().numpy()
            )
            _append_transition(
                buffer,
                observation=observation,
                policy_action=action[0],
                requested_report=report[0].cpu().numpy(),
                behavior_log_prob=float(log_prob[0]),
                behavior_slot_log_probs=distribution.logits[0].cpu().numpy(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                policy_version=version,
            )
            observation = next_observation
            hidden = next_hidden
            if terminated or truncated:
                break
        assert info is not None
        return _finish_fragment(
            buffer,
            observation,
            won=info["won"],
            termination_reason=info["termination_reason"],
        )
    finally:
        environment.close()


def _add_one_food(fragment):
    fragment["score_end"] = fragment["score_start"] + 1
    fragment["scores"][-1] = fragment["score_end"]
    fragment["food_count"] = 1
    fragment["food_events"] = [{"tick": int(fragment["controller_ticks"][-1])}]
    fragment["contains_food_event"] = True
    fragment["food_event_transition_indices"] = np.array(
        [fragment["length"] - 1], dtype=np.int32
    )


def _as_trace(fragment, config, *, boundary=None, credit_sequence=0):
    if boundary is None:
        if bool(fragment["terminated"][-1]):
            boundary = "terminated"
        elif bool(fragment["truncated"][-1]):
            boundary = "truncated"
        elif fragment["contains_food_event"]:
            boundary = "food"
        else:
            boundary = "resume"
    return _make_credit_trace(
        [fragment],
        config,
        credit_sequence=credit_sequence,
        boundary=boundary,
    )


def _slice_fragment(fragment, start, end, *, fragment_sequence):
    """Split a valid no-food fragment while preserving its audit timeline."""
    assert 0 <= start < end <= fragment["length"]
    assert not fragment["contains_food_event"]
    result = copy.deepcopy(fragment)
    result["fragment_id"] = f"{fragment['fragment_id']}-{start}-{end}"
    result["fragment_sequence"] = fragment_sequence
    result["decision_start"] = fragment["decision_start"] + start
    result["decision_end"] = fragment["decision_start"] + end
    result["length"] = end - start
    result["score_start"] = (
        fragment["score_start"] if start == 0 else int(fragment["scores"][start - 1])
    )
    result["score_end"] = int(fragment["scores"][end - 1])
    result["bootstrap_observation"] = {
        key: (
            values[end].copy()
            if end < fragment["length"]
            else fragment["bootstrap_observation"][key].copy()
        )
        for key, values in fragment["observations"].items()
    }
    result["observations"] = {
        key: values[start:end].copy()
        for key, values in fragment["observations"].items()
    }
    array_fields = (
        "policy_actions",
        "requested_reports",
        "behavior_log_probs",
        "behavior_slot_log_probs",
        "rewards",
        "discounts",
        "terminated",
        "truncated",
        "policy_versions",
        "ticks_advanced",
        "controller_ticks",
        "capture_ticks",
        "command_origin_ticks",
        "command_arrival_ticks",
        "submitted_sequences",
        "scores",
    )
    for key in array_fields:
        result[key] = fragment[key][start:end].copy()
    for key in (
        "execution_ticks",
        "tick_rewards",
        "executed_reports",
        "executed_sources",
    ):
        result[key] = copy.deepcopy(fragment[key][start:end])
    if end < fragment["length"]:
        result["terminated"][-1] = False
        result["truncated"][-1] = False
        result["won"] = False
        result["termination_reason"] = None
    result["food_events"] = []
    result["food_count"] = 0
    result["contains_food_event"] = False
    result["food_event_transition_indices"] = np.empty(0, dtype=np.int32)
    result["perturbation_events"] = 0
    result["expired_reports"] = 0
    return result


def _split_continuous_fragment(learner, *, length=4, cut=2, version=0):
    fragment = _collect_fragment(
        learner, length, max_logic_steps=length, version=version
    )
    assert fragment["length"] == length
    assert fragment["truncated"][-1]
    return (
        _slice_fragment(fragment, 0, cut, fragment_sequence=0),
        _slice_fragment(fragment, cut, length, fragment_sequence=1),
    )


def _corrupt_first_observation(fragment):
    values = next(iter(fragment["observations"].values()))
    values.reshape(-1)[0] += 1


def _constant_one_step_stream(template, count):
    observation = {
        key: values[0].copy() for key, values in template["observations"].items()
    }
    fragments = []
    for index in range(count):
        fragment = copy.deepcopy(template)
        fragment["fragment_id"] = f"{template['fragment_id']}-{index}"
        fragment["fragment_sequence"] = index
        fragment["decision_start"] = index
        fragment["decision_end"] = index + 1
        fragment["observations"] = {
            key: value[None].copy() for key, value in observation.items()
        }
        fragment["bootstrap_observation"] = copy.deepcopy(observation)
        fragment["execution_ticks"] = [[index]]
        fragment["controller_ticks"][:] = index + 1
        fragment["capture_ticks"][:] = index + 1
        fragment["command_origin_ticks"][:] = index
        fragment["command_arrival_ticks"][:] = index
        fragment["submitted_sequences"][:] = index + 1
        fragment["executed_sources"] = [np.array([index + 1], dtype=np.int64)]
        fragment["executed_reports"] = [fragment["requested_reports"][0, :1].copy()]
        if index + 1 < count:
            fragment["truncated"][:] = False
            fragment["termination_reason"] = None
        fragments.append(fragment)
    return fragments


def test_vtrace_config_enforces_event_bounds():
    with pytest.raises(ValueError, match="hard limit"):
        _config(batch_min_transitions=5, batch_max_transitions=4)
    with pytest.raises(ValueError, match="burn_in"):
        _config(recurrent_burn_in=3, unroll_length=2)
    with pytest.raises(ValueError, match="initial_length"):
        _config(width=4, initial_length=5)


def test_v3_checkpoint_and_fragment_formats_reject_old_credit_ambiguous_data():
    config = _config()
    learner = GamepadVTraceLearner(config)
    checkpoint = checkpoint_metadata(config)
    old_checkpoint = copy.deepcopy(checkpoint)
    old_checkpoint["format"] = "gamepad-vtrace-v2"
    with pytest.raises(ValueError, match="format mismatch"):
        validate_checkpoint(old_checkpoint)

    fragment = _collect_fragment(learner, 1)
    fragment["format"] = "gamepad-vtrace-fragment-v4"
    with pytest.raises(ValueError, match="fragment format mismatch"):
        validate_fragment(fragment, config)


def test_fresh_collector_consumes_fifo_until_food_or_hard_limit():
    config = _config(
        actor_processes=2,
        batch_min_transitions=2,
        batch_max_transitions=4,
        batch_food_target=1,
    )
    learner = GamepadVTraceLearner(config)
    first = _collect_fragment(learner, 2, max_logic_steps=2)
    second = _collect_fragment(learner, 1)
    second["actor_id"] = 1
    _add_one_food(second)
    fragments = queue.Queue()
    fragments.put(first)
    fragments.put(second)

    assembler = CreditTraceAssembler(config)
    result, metrics = collect_fresh_credit_traces(
        fragments, threading.Event(), config, assembler
    )

    assert [trace["fragments"] for trace in result] == [[first], [second]]
    assert metrics["fresh_transitions"] == 3
    assert metrics["fresh_food_count"] == 1
    assert metrics["fresh_event_trace_count"] == 1
    assert metrics["fresh_non_event_trace_count"] == 1
    assert metrics["fresh_food_target_met"]
    assert not metrics["fresh_hard_limit_reached"]
    assert metrics["fresh_food_quota_required_after_minimum"]
    assert metrics["fresh_food_quota_binding"]
    assert metrics["fresh_food_quota_added_data"]
    assert metrics["fresh_food_quota_wait_traces"] == 1
    assert metrics["fresh_food_quota_wait_transitions"] == 1
    assert metrics["fresh_food_quota_wait_logic_ticks"] == 1


def test_collector_reports_nonbinding_food_quota_without_extra_wait():
    config = _config(
        batch_min_transitions=1,
        batch_max_transitions=4,
        batch_food_target=1,
    )
    learner = GamepadVTraceLearner(config)
    event = _collect_fragment(learner, 1)
    _add_one_food(event)
    fragments = queue.Queue()
    fragments.put(event)

    assembler = CreditTraceAssembler(config)
    result, metrics = collect_fresh_credit_traces(
        fragments, threading.Event(), config, assembler
    )

    assert result[0]["fragments"] == [event]
    assert not metrics["fresh_food_quota_required_after_minimum"]
    assert not metrics["fresh_food_quota_binding"]
    assert not metrics["fresh_food_quota_added_data"]
    assert metrics["fresh_food_quota_wait_traces"] == 0
    assert metrics["fresh_food_quota_wait_transitions"] == 0


def test_collector_distinguishes_unmet_quota_from_a_binding_wait():
    config = _config(
        batch_min_transitions=2,
        batch_max_transitions=2,
        batch_food_target=1,
    )
    learner = GamepadVTraceLearner(config)
    no_event = _collect_fragment(learner, 2, max_logic_steps=2)
    fragments = queue.Queue()
    fragments.put(no_event)

    assembler = CreditTraceAssembler(config)
    result, metrics = collect_fresh_credit_traces(
        fragments, threading.Event(), config, assembler
    )

    assert result[0]["fragments"] == [no_event]
    assert metrics["fresh_food_quota_required_after_minimum"]
    assert metrics["fresh_food_quota_binding"]
    assert not metrics["fresh_food_quota_added_data"]
    assert metrics["fresh_food_quota_wait_traces"] == 0
    assert metrics["fresh_food_quota_wait_transitions"] == 0
    assert metrics["fresh_hard_limit_reached"]


def test_received_events_are_not_delayed_when_terminal_trace_waits_in_ready_queue():
    config = _config(
        unroll_length=2,
        credit_trace_max_transitions=2,
        batch_min_transitions=1,
        batch_max_transitions=1,
        batch_food_target=0,
    )
    learner = GamepadVTraceLearner(config)
    first, terminal = _split_continuous_fragment(learner, length=3, cut=1)
    _add_one_food(terminal)
    fragments = queue.Queue()
    fragments.put(first)
    fragments.put(terminal)
    assembler = CreditTraceAssembler(config)

    first_batch, first_metrics = collect_fresh_credit_traces(
        fragments, threading.Event(), config, assembler
    )

    assert [trace["credit_boundary"] for trace in first_batch] == ["safety_cap"]
    assert first_metrics["fresh_received_logic_ticks"] == 3
    assert first_metrics["fresh_logic_ticks"] == 1
    assert first_metrics["fresh_received_food_count"] == 1
    assert first_metrics["fresh_received_food_tick_positions"] == [3]
    assert first_metrics["received_action_transition_count"] == 3
    assert first_metrics[
        "received_action_current_source_report_match_denominator"
    ] == sum(fragment["ticks_advanced"].sum() for fragment in (first, terminal))
    assert first_metrics["fresh_trained_food_count"] == 0
    assert len(first_metrics["fresh_received_episodes"]) == 1
    assert first_metrics["fresh_received_episodes"][0]["score"] == 1
    assert first_metrics["fresh_received_episodes"][0]["truncated"] is True
    assert first_metrics["fresh_received_episodes"][0]["receipt_tick_position"] == 3
    assert first_metrics["fresh_trained_episodes"] == []
    assert assembler.ready_transitions == 2

    second_batch, second_metrics = collect_fresh_credit_traces(
        fragments, threading.Event(), config, assembler
    )

    assert [trace["credit_boundary"] for trace in second_batch] == ["truncated"]
    assert second_metrics["fresh_received_logic_ticks"] == 0
    assert second_metrics["fresh_logic_ticks"] == 2
    assert second_metrics["fresh_received_food_count"] == 0
    assert second_metrics["fresh_received_food_tick_positions"] == []
    assert second_metrics["received_action_transition_count"] == 0
    assert "received_action_current_source_report_match_rate" not in second_metrics
    assert second_metrics["fresh_trained_food_count"] == 1
    assert second_metrics["fresh_received_episodes"] == []
    assert len(second_metrics["fresh_trained_episodes"]) == 1


def test_actor_food_cut_keeps_game_and_recurrent_state_continuous(monkeypatch):
    config = _config(
        actor_sync_steps=10,
        unroll_length=4,
        recurrent_burn_in=2,
    )
    probe = make_env(config)
    observation = probe.reset(seed=404)[0]
    probe.close()

    class FakeEnvironment:
        def __init__(self):
            self.steps = 0
            self.reset_calls = 0

        def reset(self, *, seed=None):
            self.reset_calls += 1
            return copy.deepcopy(observation), {}

        def step(self, report):
            self.steps += 1
            food_events = [{"tick": self.steps}] if self.steps == 1 else []
            terminated = self.steps == 2
            reward = float(bool(food_events))
            executed = np.asarray(report, dtype=np.float32)[:1].copy()
            info = {
                "bootstrap_discount": config.gamma,
                "ticks_advanced": 1,
                "controller_tick": self.steps,
                "capture_tick": self.steps,
                "command_origin_tick": self.steps - 1,
                "command_arrival_tick": self.steps - 1,
                "execution_ticks": [self.steps - 1],
                "tick_rewards": [reward],
                "executed_reports": executed,
                "executed_sources": np.array([self.steps], dtype=np.int64),
                "submitted_sequence": self.steps,
                "food_events": food_events,
                "score": 1,
                "perturbation_events": [],
                "expired_reports": 0,
                "won": False,
                "termination_reason": "wall_collision" if terminated else None,
            }
            return copy.deepcopy(observation), reward, terminated, False, info

        def close(self):
            return None

    environment = FakeEnvironment()
    monkeypatch.setattr(
        "resnake_gym.gamepad_vtrace_actor.make_env", lambda unused: environment
    )
    learner = GamepadVTraceLearner(config)
    shared = (
        torch.nn.utils.parameters_to_vector(learner.model.parameters())
        .detach()
        .cpu()
        .clone()
        .share_memory_()
    )

    class Version:
        value = 0

    stop_event = threading.Event()

    class Sink:
        def __init__(self):
            self.items = []

        def put(self, payload, timeout):
            self.items.append(payload)
            if len(self.items) == 2:
                stop_event.set()

    sink = Sink()
    actor_worker(
        0,
        0,
        asdict(config),
        shared,
        Version(),
        threading.Lock(),
        sink,
        stop_event,
    )

    assert [item["kind"] for item in sink.items] == ["fragment", "fragment"], sink.items
    event_fragment, continuation = sink.items
    validate_fragment(event_fragment, config)
    validate_fragment(continuation, config)
    assert event_fragment["contains_food_event"]
    assert event_fragment["food_event_transition_indices"].tolist() == [0]
    assert continuation["score_start"] == continuation["score_end"] == 1
    assert not continuation["contains_food_event"]
    assert event_fragment["episode_id"] == continuation["episode_id"] == 0
    assert environment.reset_calls == 2  # initial reset plus the true terminal only
    assert sum(item["length"] for item in sink.items) == 2
    assert [int(item["controller_ticks"][0]) for item in sink.items] == [1, 2]
    assert [item["fragment_sequence"] for item in sink.items] == [0, 1]
    assert [(item["decision_start"], item["decision_end"]) for item in sink.items] == [
        (0, 1),
        (1, 2),
    ]
    for key in observation:
        np.testing.assert_array_equal(
            event_fragment["bootstrap_observation"][key],
            continuation["observations"][key][0],
        )
    learner.model.eval()
    with torch.no_grad():
        _, _, expected_hidden = distribution_from_observation(
            learner.model,
            stack_observations([observation], learner.device),
            torch.zeros(1, config.dim, device=learner.device),
        )
    np.testing.assert_allclose(
        continuation["initial_hidden"],
        expected_hidden[0].cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    assembler = CreditTraceAssembler(config)
    food_traces = assembler.append(event_fragment)
    terminal_traces = assembler.append(continuation)
    assert [trace["credit_boundary"] for trace in food_traces] == ["food"]
    assert [trace["credit_boundary"] for trace in terminal_traces] == ["terminated"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: (
                item.update(score_start=0, score_end=0),
                item["scores"].fill(0),
            ),
            "score is discontinuous",
        ),
        (_corrupt_first_observation, "observation is discontinuous"),
        (
            lambda item: item.update(fragment_sequence=3),
            "missing, duplicated, or out of order",
        ),
    ],
)
def test_assembler_audits_continuity_across_food_credit_boundaries(mutation, message):
    config = _config(unroll_length=4, credit_trace_max_transitions=8)
    learner = GamepadVTraceLearner(config)
    first, second = _split_continuous_fragment(learner, length=4, cut=2)
    _add_one_food(first)
    second["score_start"] = 1
    second["score_end"] = 1
    second["scores"].fill(1)
    assembler = CreditTraceAssembler(config)
    assert assembler.append(first)[0]["credit_boundary"] == "food"
    corrupted = copy.deepcopy(second)
    mutation(corrupted)
    validate_fragment(corrupted, config)

    with pytest.raises(ValueError, match=message):
        assembler.append(corrupted)


def test_assembler_rejects_missing_and_duplicate_segments_before_counting():
    config = _config(unroll_length=4, credit_trace_max_transitions=8)
    learner = GamepadVTraceLearner(config)
    first, second = _split_continuous_fragment(learner, length=4, cut=2)
    assembler = CreditTraceAssembler(config)
    assert assembler.append(first) == []
    counters = (
        assembler.cumulative_received_fragments,
        assembler.cumulative_received_transitions,
        assembler.cumulative_received_logic_ticks,
    )
    missing = copy.deepcopy(second)
    missing["fragment_sequence"] += 1
    with pytest.raises(ValueError, match="missing, duplicated, or out of order"):
        assembler.append(missing)
    duplicate = copy.deepcopy(first)
    with pytest.raises(ValueError, match="missing, duplicated, or out of order"):
        assembler.append(duplicate)
    assert counters == (
        assembler.cumulative_received_fragments,
        assembler.cumulative_received_transitions,
        assembler.cumulative_received_logic_ticks,
    )


def test_assembler_checkpoint_preserves_pending_then_resume_flushes_to_ready():
    config = _config(unroll_length=4, credit_trace_max_transitions=8)
    learner = GamepadVTraceLearner(config)
    first, _ = _split_continuous_fragment(learner, length=4, cut=2)
    assembler = CreditTraceAssembler(config)
    assert assembler.append(first) == []
    state = copy.deepcopy(assembler.state_dict())
    missing_counter = copy.deepcopy(state)
    missing_counter.pop("cumulative_received_logic_ticks")
    below_retained = copy.deepcopy(state)
    below_retained["cumulative_received_transitions"] = 1
    wrong_key = copy.deepcopy(state)
    key, stream = wrong_key["streams"].popitem()
    wrong_key["streams"][(key[0], key[1] + 1, key[2])] = stream
    restored = CreditTraceAssembler(config)

    restored.load_state_dict(state)

    assert restored.pending_transitions == 2
    assert restored.pending_logic_ticks == 2
    assert restored.cumulative_received_transitions == 2
    restored.flush_for_new_generation()
    assert restored.pending_transitions == 0
    assert restored.ready_transitions == 2
    assert restored.ready_logic_ticks == 2
    trace = restored.pop_ready()
    assert trace["credit_boundary"] == "resume"
    validate_credit_trace(trace, config)
    assert restored.cumulative_received_transitions == trace["length"]

    with pytest.raises(ValueError, match="counters are missing"):
        CreditTraceAssembler(config).load_state_dict(missing_counter)
    with pytest.raises(ValueError, match="below retained data"):
        CreditTraceAssembler(config).load_state_dict(below_retained)
    with pytest.raises(ValueError, match="stream key"):
        CreditTraceAssembler(config).load_state_dict(wrong_key)


def test_credit_cap_128_and_2048_use_the_same_assembler_path():
    short_config = _config(
        dim=8,
        max_logic_steps=1,
        unroll_length=1,
        recurrent_burn_in=1,
        bptt_window=128,
        credit_trace_max_transitions=128,
    )
    learner = GamepadVTraceLearner(short_config)
    template = _collect_fragment(learner, 1, max_logic_steps=1)
    fragments = _constant_one_step_stream(template, 129)
    short_assembler = CreditTraceAssembler(short_config)
    short_traces = []
    for fragment in fragments:
        short_traces.extend(short_assembler.append(copy.deepcopy(fragment)))

    long_config = copy.deepcopy(short_config)
    long_config.credit_trace_max_transitions = 2048
    long_learner = GamepadVTraceLearner(long_config)
    long_learner.model.load_state_dict(learner.model.state_dict())
    long_assembler = CreditTraceAssembler(long_config)
    long_traces = []
    for fragment in fragments:
        long_traces.extend(long_assembler.append(copy.deepcopy(fragment)))

    assert [(trace["length"], trace["credit_boundary"]) for trace in short_traces] == [
        (128, "safety_cap"),
        (1, "truncated"),
    ]
    assert [(trace["length"], trace["credit_boundary"]) for trace in long_traces] == [
        (129, "truncated")
    ]
    long_metrics = long_learner.update(long_traces)
    assert long_metrics["credit_trace_over_128_count"] == 1
    assert long_metrics["credit_trace_over_128_rate"] == pytest.approx(1.0)
    assert long_metrics["credit_trace_cap_hit_count"] == 0
    assert long_metrics["credit_artificial_boundary_count"] == 0


def test_collector_detects_actor_exit_without_an_error_payload():
    config = _config()
    assembler = CreditTraceAssembler(config)
    stopped = threading.Event()

    class EmptyQueue:
        @staticmethod
        def get(timeout):
            raise queue.Empty

    class DeadActor:
        name = "dead-actor"
        exitcode = -9

    with pytest.raises(RuntimeError, match="exited without a queue error"):
        collect_fresh_credit_traces(
            EmptyQueue(),
            stopped,
            config,
            assembler,
            actor_processes=[DeadActor()],
        )
    assert stopped.is_set()


def test_learner_handles_variable_lengths_and_exact_full_reports():
    config = _config()
    learner = GamepadVTraceLearner(config)
    short = _collect_fragment(learner, 1)
    long = _collect_fragment(learner, 2)

    for fragment in (short, long):
        reports = fragment["requested_reports"]
        assert reports.shape[-1] == 20
        assert np.all(reports[..., 4:] == 0)
        assert np.all((reports[..., :4] > 0.5).sum(-1) <= 1)

    metrics = learner.update([_as_trace(short, config), _as_trace(long, config)])

    assert metrics["learner_transitions"] == 3
    assert metrics["padding_fraction"] == pytest.approx(0.25)
    assert metrics["rho_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["behavior_target_joint_kl_mean"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["bptt_windows_per_update"] == 1
    assert metrics["bptt_backward_calls"] == 1
    assert metrics["trace_window_count"] == 2
    assert np.isfinite(metrics["loss"])


def test_learner_update_preserves_metric_contract_and_one_optimizer_step(
    monkeypatch,
):
    config = _config(
        unroll_length=2,
        bptt_window=1,
        value_coefficient=0.37,
        entropy_coefficient=0.011,
    )
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 2)
    optimizer_steps = 0
    original_step = learner.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(learner.optimizer, "step", counted_step)

    metrics = learner.update([_as_trace(fragment, config)])

    assert set(metrics) == EXPECTED_VTRACE_UPDATE_METRICS
    assert optimizer_steps == 1
    assert metrics["bptt_backward_calls"] == 2
    assert learner.update_count == metrics["learner_update"] == 1
    assert learner.transitions == metrics["learner_transitions"] == 2
    assert metrics["loss"] == pytest.approx(
        metrics["policy_loss"]
        + config.value_coefficient * metrics["value_loss"]
        - config.entropy_coefficient * metrics["latent_entropy"],
        rel=1e-6,
        abs=1e-7,
    )
    assert metrics["entropy_beta_h"] == pytest.approx(
        config.entropy_coefficient * metrics["latent_entropy"]
    )
    assert all(
        np.isfinite(value) for key, value in metrics.items() if key != "learner_source"
    )


def test_vtrace_return_crosses_actor_segments_while_bptt_steps_once(monkeypatch):
    config = _config(
        gamma=0.9,
        unroll_length=4,
        recurrent_burn_in=2,
        bptt_window=2,
        credit_trace_max_transitions=8,
    )
    learner = GamepadVTraceLearner(config)
    first, second = _split_continuous_fragment(learner, length=4, cut=2)
    for fragment in (first, second):
        fragment["rewards"].fill(0)
        fragment["tick_rewards"] = [[0.0] for _ in range(fragment["length"])]
    second["rewards"][-1] = 1.0
    second["tick_rewards"][-1] = [1.0]
    assembler = CreditTraceAssembler(config)
    assert assembler.append(first) == []
    traces = assembler.append(second)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["segment_count"] == 2
    (
        target_log_probs,
        values,
        bootstrap_value,
        mask,
        _,
        _,
        _,
        _,
    ) = learner._target_credit_traces(traces)
    behavior = torch.as_tensor(
        np.concatenate(
            [fragment["behavior_log_probs"] for fragment in trace["fragments"]]
        )[:, None],
        device=learner.device,
    )
    rewards = torch.as_tensor(
        np.concatenate([fragment["rewards"] for fragment in trace["fragments"]])[
            :, None
        ],
        device=learner.device,
    )
    discounts = torch.as_tensor(
        np.concatenate([fragment["discounts"] for fragment in trace["fragments"]])[
            :, None
        ],
        device=learner.device,
    )
    returns = vtrace_from_log_probs(
        target_log_probs,
        behavior,
        discounts,
        rewards,
        values,
        bootstrap_value,
    )
    expected = float(bootstrap_value[0])
    for reward, discount in zip(
        rewards[:, 0].cpu().numpy()[::-1],
        discounts[:, 0].cpu().numpy()[::-1],
        strict=True,
    ):
        expected = float(reward) + float(discount) * expected
    assert bool(mask.all())
    assert float(returns.vs[0, 0]) == pytest.approx(expected, abs=1e-5)

    optimizer_steps = 0
    original_step = learner.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(learner.optimizer, "step", counted_step)
    metrics = learner.update(traces)

    assert optimizer_steps == 1
    assert metrics["bptt_windows_per_update"] == 2
    assert metrics["credit_trace_len_max"] == 4
    assert metrics["segments_per_trace_max"] == 2
    assert metrics["credit_bootstrap_boundary_count"] == 1
    assert np.isfinite(metrics["transport_log_weight_min"])
    assert assembler.cumulative_received_transitions == (
        learner.transitions
        + assembler.pending_transitions
        + assembler.ready_transitions
    )
    assert assembler.cumulative_received_logic_ticks == (
        learner.logic_ticks
        + assembler.pending_logic_ticks
        + assembler.ready_logic_ticks
    )


def test_full_trace_targets_do_not_depend_on_bptt_window():
    short_config = _config(
        unroll_length=4,
        recurrent_burn_in=2,
        bptt_window=1,
        credit_trace_max_transitions=8,
    )
    long_config = copy.deepcopy(short_config)
    long_config.bptt_window = 2
    short_learner = GamepadVTraceLearner(short_config)
    long_learner = GamepadVTraceLearner(long_config)
    long_learner.model.load_state_dict(short_learner.model.state_dict())
    first, second = _split_continuous_fragment(short_learner, length=4, cut=2)
    assembler = CreditTraceAssembler(short_config)
    assert assembler.append(first) == []
    traces = assembler.append(second)

    short_outputs = short_learner._target_credit_traces(traces)
    long_outputs = long_learner._target_credit_traces(traces)

    for left, right in zip(short_outputs[:7], long_outputs[:7], strict=True):
        torch.testing.assert_close(left, right)


def test_mixed_fragment_policy_versions_keep_per_transition_behavior_probabilities():
    config = _config(
        unroll_length=4,
        recurrent_burn_in=2,
        bptt_window=2,
        credit_trace_max_transitions=8,
    )
    learner = GamepadVTraceLearner(config)
    first, second = _split_continuous_fragment(learner, length=4, cut=2)
    second["policy_versions"].fill(1)
    assembler = CreditTraceAssembler(config)
    assert assembler.append(first) == []
    traces = assembler.append(second)
    learner.update_count = 1

    metrics = learner.update(traces)

    assert metrics["target_policy_version"] == 1
    assert metrics["mixed_policy_versions_mean"] == pytest.approx(2.0)
    assert metrics["mixed_policy_versions_max"] == 2
    assert metrics["unique_behavior_versions_max"] == 2
    assert metrics["rho_mean"] == pytest.approx(1.0, abs=1e-5)


def test_entropy_metrics_use_one_variable_per_dpad_chunk_slot():
    config = _config(entropy_coefficient=0.008)
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1)

    metrics = learner.update([_as_trace(fragment, config)])

    assert metrics["entropy_joint_nats"] == pytest.approx(metrics["latent_entropy"])
    assert metrics["entropy_action_variables_per_decision"] == config.chunk_length
    assert metrics[
        "entropy_per_action_variable_nats"
    ] * config.chunk_length == pytest.approx(metrics["entropy_joint_nats"])
    assert metrics["entropy_normalized_fraction"] == pytest.approx(
        metrics["entropy_per_action_variable_nats"] / np.log(5.0)
    )
    assert metrics["entropy_beta_h"] == pytest.approx(
        config.entropy_coefficient * metrics["entropy_joint_nats"]
    )


def test_truncation_bootstraps_its_final_observation_without_crossing_reset():
    config = _config(max_logic_steps=1, unroll_length=1, recurrent_burn_in=1)
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1, max_logic_steps=1)

    assert fragment["truncated"].tolist() == [True]
    assert fragment["terminated"].tolist() == [False]
    assert fragment["discounts"][0] == pytest.approx(config.gamma)
    (
        target_log_probs,
        values,
        bootstrap_value,
        _,
        _,
        _,
        _,
        _,
    ) = learner._target_credit_traces([_as_trace(fragment, config)])
    behavior = torch.as_tensor(
        fragment["behavior_log_probs"][:, None], device=learner.device
    )
    result = vtrace_from_log_probs(
        target_log_probs,
        behavior,
        torch.as_tensor(fragment["discounts"][:, None], device=learner.device),
        torch.as_tensor(fragment["rewards"][:, None], device=learner.device),
        values,
        bootstrap_value,
    )
    expected = fragment["rewards"][0] + fragment["discounts"][0] * float(
        bootstrap_value[0].detach()
    )
    assert float(result.vs[0, 0]) == pytest.approx(expected, abs=1e-5)


def test_importance_ess_is_stable_for_extreme_log_ratios():
    concentrated = importance_effective_sample_size(
        torch.tensor([1_000.0, 100.0], dtype=torch.float32)
    )
    equal = importance_effective_sample_size(
        torch.tensor([1_000.0, 1_000.0], dtype=torch.float32)
    )

    assert torch.isfinite(concentrated)
    assert concentrated == pytest.approx(1.0)
    assert equal == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(food_count=1), "food count"),
        (lambda item: item.update(score_end=1), "score_end"),
        (lambda item: item.update(contains_food_event=True), "contains_food_event"),
        (lambda item: item.update(won=True), "win"),
        (lambda item: item.update(termination_reason="wall_collision"), "boundary"),
    ],
)
def test_fragment_validator_rejects_progress_or_boundary_metadata_drift(
    mutation, message
):
    config = _config()
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1)
    mutation(fragment)

    with pytest.raises(ValueError, match=message):
        validate_fragment(fragment, config)


def test_fragment_validator_rejects_invalid_policy_version_or_duration():
    config = _config()
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1)
    fragment["policy_versions"][0] = -1
    with pytest.raises(ValueError, match="policy version"):
        validate_fragment(fragment, config)

    fragment = _collect_fragment(learner, 1)
    fragment["ticks_advanced"][0] = config.decision_max + 1
    with pytest.raises(ValueError, match="duration"):
        validate_fragment(fragment, config)


def test_fragment_validator_audits_full_behavior_distribution():
    config = _config()
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1)
    fragment["behavior_slot_log_probs"][0, 0, 0] += 0.5

    with pytest.raises(ValueError, match="behavior log probabilities"):
        validate_fragment(fragment, config)


@pytest.mark.parametrize(
    ("invalid_stage", "message"),
    [
        ("schema", "fragment action encoding mismatch"),
        ("shape", "fragment recurrent state shape mismatch"),
        ("behavior", "fragment contains nonfinite behavior distribution"),
        ("timeline", "fragment command arrival delay is outside configured bounds"),
        ("action", "requested full report does not match dpad5 category"),
        ("execution", "executed audit rows must match ticks_advanced"),
        ("observation", r"fragment observation .* length mismatch"),
    ],
)
def test_fragment_validator_reports_each_validation_stage(invalid_stage, message):
    """Keep the staged validator's public failure messages regression-tested."""
    config = _config()
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1)

    if invalid_stage == "schema":
        fragment["action_encoding"] = "invalid"
    elif invalid_stage == "shape":
        fragment["initial_hidden"] = np.zeros(config.dim + 1, dtype=np.float32)
    elif invalid_stage == "behavior":
        fragment["behavior_slot_log_probs"][0, 0, 0] = np.nan
    elif invalid_stage == "timeline":
        fragment["command_arrival_ticks"][0] += 1
    elif invalid_stage == "action":
        fragment["requested_reports"][0, 0, 0] = (
            1.0 - fragment["requested_reports"][0, 0, 0]
        )
    elif invalid_stage == "execution":
        fragment["tick_rewards"][0].append(0.0)
    else:
        key = next(iter(fragment["observations"]))
        fragment["observations"][key] = fragment["observations"][key][:0]

    with pytest.raises(ValueError, match=message):
        validate_fragment(fragment, config)


def test_future_behavior_version_is_rejected_before_optimizer_mutation():
    config = _config()
    learner = GamepadVTraceLearner(config)
    fragment = _collect_fragment(learner, 1, version=1)
    before = {
        key: value.detach().clone() for key, value in learner.model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="newer"):
        learner.update([_as_trace(fragment, config)])

    assert learner.update_count == 0
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_inference_policy_cannot_partially_restore_vtrace_learner():
    learner = GamepadVTraceLearner(_config())
    payload = learner.checkpoint()
    payload.pop("optimizer")
    payload["artifact_kind"] = "inference-policy"
    before = {
        key: value.detach().clone() for key, value in learner.model.state_dict().items()
    }

    with pytest.raises(ValueError, match="full training-state"):
        learner.restore(payload)

    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_shared_parameter_publication_is_coherent_and_versioned():
    config = _config()
    learner = GamepadVTraceLearner(config)
    replica = GamepadVTraceLearner(config).model
    shared = torch.empty(parameter_count(learner.model)).share_memory_()

    class Version:
        value = 0

    version = Version()
    lock = threading.Lock()
    published = publish_parameters(learner.model, shared, version, lock, 7)
    loaded = load_published_parameters(replica, shared, version, lock, -1)

    assert published == loaded == 7
    for expected, actual in zip(
        learner.model.parameters(), replica.parameters(), strict=True
    ):
        torch.testing.assert_close(expected.cpu(), actual.cpu())

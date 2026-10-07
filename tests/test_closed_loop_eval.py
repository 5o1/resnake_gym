import copy

import numpy as np

from resnake_gym.closed_loop_eval import (
    EvaluationConfig,
    counterfactual_probe,
    evaluate_episode,
    paired_bootstrap_difference,
    recovery_metrics,
    summarize,
)


class Fixture:
    def reset(self, *, seed):
        self.inputs = []

    def act(self, obs):
        self.inputs.append(copy.deepcopy(obs))
        return np.zeros((8, 20), np.float32)


def test_full_feedback_freeze_and_absolute_tick_replay():
    policy = Fixture()
    config = EvaluationConfig(max_logic_steps=12)
    closed = evaluate_episode(policy, config, seed=52, condition="none")
    frozen = evaluate_episode(policy, config, seed=52, condition="none", mode="frozen")
    assert len(policy.inputs) > 1
    for obs in policy.inputs:
        for key, value in obs.items():
            np.testing.assert_array_equal(value, policy.inputs[0][key])
    replay = evaluate_episode(
        policy,
        config,
        seed=52,
        condition="none",
        mode="replay",
        replay=closed["actual_reports"],
    )
    assert not policy.inputs  # replay never calls the policy
    assert closed["action_sha256"] == frozen["action_sha256"] == replay["action_sha256"]
    assert closed["ticks"] == len(closed["actual_reports"])
    assert summarize([closed, frozen, replay])["evidence_gate"] == "insufficient"


def test_recovery_handles_death_cutoff_and_overlapping_events():
    event = {"status": "applied", "tick": 10, "kind": "rotation", "score_at_event": 1}
    assert recovery_metrics([event], [], 15, "max_logic_steps", 10)[0]["censored"]
    dead = recovery_metrics([event], [(11, 2)], 15, "wall_collision", 10)[0]
    assert not dead["censored"] and not dead["survived"] and dead["ate_within_window"]
    events = [event, {**event, "tick": 12}]
    alive = recovery_metrics(events, [(11, 2)], 30, "max_logic_steps", 10)[0]
    assert alive["survived"] and alive["other_event_in_window"]


def test_bootstrap_null_and_positive_control():
    assert paired_bootstrap_difference([1, 2], [1, 2])["ci95"] == [0.0, 0.0]
    assert paired_bootstrap_difference([2, 3], [1, 2])["ci95"] == [1.0, 1.0]


def test_counterfactual_clones_same_memory():
    class Probe:
        def reset(self, *, seed):
            self.memory = 0

        def act(self, obs):
            self.memory += 1
            return np.full((1, 20), self.memory + obs["signal"], np.float32)

        def fork(self):
            return copy.copy(self)

    policy = Probe()
    result = counterfactual_probe(
        policy, [{"signal": 0}], [{"signal": 0}, {"signal": 0}, {"signal": 1}], 0
    )
    assert result["changed"] == [False, True]
    assert policy.memory == 1

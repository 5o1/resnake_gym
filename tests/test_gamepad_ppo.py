import copy
from unittest.mock import Mock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo import (  # noqa: E402
    GamepadPPO,
    PPOConfig,
    _recurrent_windows,
    collection_stop,
    generalized_advantages,
    stack_observations,
    validate_policy_checkpoint,
)
from resnake_gym.models import (  # noqa: E402
    GamepadPolicy,
    distribution_from_observation,
    reports_from_samples,
    sample_log_prob,
)


def test_gae_bootstraps_truncation_but_does_not_cross_reset():
    rewards = torch.tensor([[1.0], [10.0], [100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 8.0)
    discounts = torch.full_like(rewards, 0.25)
    terminated = torch.tensor([[False], [True], [False]])
    truncated = torch.tensor([[True], [False], [False]])
    result = generalized_advantages(
        rewards, values, next_values, discounts, terminated, truncated, 1.0
    )
    torch.testing.assert_close(result, torch.tensor([[3.0], [10.0], [102.0]]))


def test_gae_variable_tick_discount():
    rewards = torch.tensor([[1.0], [2.0]])
    result = generalized_advantages(
        rewards,
        torch.zeros_like(rewards),
        torch.zeros_like(rewards),
        torch.tensor([[0.5**2], [0.5**5]]),
        torch.tensor([[False], [True]]),
        torch.zeros(2, 1, dtype=torch.bool),
        1.0,
    )
    torch.testing.assert_close(result, torch.tensor([[1.5], [2.0]]))


def test_gae_rollout_cutoff_bootstraps_without_crossing_the_next_batch():
    result = generalized_advantages(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[10.0]]),
        torch.tensor([[0.5]]),
        torch.tensor([[False]]),
        torch.tensor([[False]]),
        1.0,
    )
    # The next-state value is used once, while an unavailable future advantage
    # cannot leak backward from the following PPO update.
    torch.testing.assert_close(result, torch.tensor([[4.0]]))


def test_event_collection_uses_only_food_and_has_a_hard_limit():
    c = PPOConfig(
        dim=16,
        num_envs=1,
        rollout_steps=7,
        epochs=1,
        collection_mode="events",
        minimum_steps=3,
        event_target=1,
        max_logic_steps=1,
    )
    assert collection_stop(c, 2, 10, 10) is None
    # Deaths are failure experience, not successful events.
    assert collection_stop(c, 3, 0, 100) is None
    assert collection_stop(c, 3, 1, 100) == "food_target"
    assert collection_stop(c, 7, 0, 100) == "hard_limit"
    trainer = GamepadPPO(c)
    try:
        result = trainer.update()
        # Time-limit truncations must not count as successes or deaths.
        assert result["collection_stop"] == "hard_limit"
        assert result["collection_food_target_met"] is False
        assert result["collection_food_shortfall"] == 1
        assert result["collection_hard_limit_reached"] is True
        assert result["decisions"] == 7
        assert result["rollout_transitions"] == 7
        assert result["completed_episodes"] == 7
        assert result["terminal_events"] == 0
        assert result["truncation_events"] == 7
        assert (
            sum(f["end"] - f["start"] for f in result["fragments"])
            == result["logic_ticks"]
        )
    finally:
        trainer.close()


class _ScriptedEnv:
    """Minimal deterministic environment for collection/replay boundary tests."""

    def __init__(self, observation, script):
        self.observation = {
            key: np.array(value, copy=True) for key, value in observation.items()
        }
        self.script = list(script)
        self.index = 0
        self.tick = 0
        self.score = 0

    def _observation(self):
        return {key: value.copy() for key, value in self.observation.items()}

    def reset(self, **_kwargs):
        self.tick = 0
        self.score = 0
        return self._observation(), {}

    def step(self, _report):
        spec = self.script[self.index]
        self.index += 1
        self.tick += 1
        food_events = []
        for _ in range(spec.get("foods", 0)):
            self.score += 1
            food_events.append({"tick": self.tick})
        terminated = spec.get("terminated", False)
        truncated = spec.get("truncated", False)
        info = {
            "bootstrap_discount": 0.99,
            "tick_rewards": [float(len(food_events) - int(terminated))],
            "execution_ticks": [self.tick - 1],
            "executed_sources": np.array([0]),
            "food_events": food_events,
            "score": self.score,
            "termination_reason": "collision" if terminated else None,
            "controller_tick": self.tick,
            "capture_tick": self.tick,
            "command_origin_tick": self.tick - 1,
            "command_arrival_tick": self.tick - 1,
            "timebase": "simulation_tick",
            "ticks_since_food": 0 if food_events else self.tick,
            "ticks_advanced": 1,
            "expired_reports": 0,
            "perturbation_events": [],
            "logic_steps": self.tick,
            "won": False,
        }
        reward = float(len(food_events) - int(terminated))
        return self._observation(), reward, terminated, truncated, info

    def close(self):
        pass


def test_food_target_waits_past_death_and_replay_keeps_episode_boundaries():
    """A food episode may span rollouts, but must never absorb another episode."""
    torch.set_num_threads(1)
    config = PPOConfig(
        dim=16,
        num_envs=1,
        rollout_steps=4,
        minimum_steps=2,
        event_target=1,
        collection_mode="events",
        epochs=1,
    )
    trainer = GamepadPPO(config)
    original = trainer.envs[0]
    script = [
        {"terminated": True},  # episode 0: failure must not fill the quota
        {},  # episode 1 begins
        {"foods": 1},  # target reached only here; episode remains live
        {"terminated": True},  # episode 1 completes in the next rollout
        {"terminated": True},  # episodes 2--4 remain separate failures
        {"terminated": True},
        {"terminated": True},
    ]
    scripted = _ScriptedEnv(trainer.observations[0], script)
    original.close()
    trainer.envs = [scripted]
    trainer.observations = [scripted.reset()[0]]
    try:
        first = trainer.update()
        assert first["collection_stop"] == "food_target"
        assert first["rollout_decisions_per_env"] == 3
        assert first["terminal_events"] == 1
        assert first["food_count"] == 1
        assert first["collection_food_target_met"] is True
        assert first["collection_hard_limit_reached"] is False

        # The failed episode is complete.  The food-containing episode is still
        # one pending sequence rather than being cut at the rollout boundary.
        archived_ids = [
            [row["episode_id"] for row in episode]
            for episode in trainer.replay.episodes
        ]
        assert archived_ids == [[0]]
        assert [row["episode_id"] for row in trainer.replay.pending[0]] == [1, 1]
        assert any(row["food_events"] for row in trainer.replay.pending[0])

        second = trainer.update()
        assert second["collection_stop"] == "hard_limit"
        assert second["collection_food_target_met"] is False
        assert second["collection_food_shortfall"] == 1
        assert second["terminal_events"] == 4
        assert second["completed_food_episodes"] == 1
        assert second["completed_food_episodes_total"] == 1
        ids = [
            [row["episode_id"] for row in episode]
            for episode in trainer.replay.episodes
        ]
        assert ids == [[0], [1, 1, 1], [2], [3], [4]]
        food_episode_count = sum(
            any(row["food_events"] for row in episode)
            for episode in trainer.replay.episodes
        )
        assert food_episode_count == 1

        payload = copy.deepcopy(trainer.checkpoint())
        restored = GamepadPPO(config)
        try:
            restored.restore(payload)
            assert restored.episode_ids == [5]
            assert restored.replay.count == 5
            assert restored.completed_food_episodes == 1
        finally:
            restored.close()
    finally:
        trainer.close()


def test_nonterminal_food_snapshot_is_sampled_and_restored_without_pending():
    torch.set_num_threads(1)
    config = PPOConfig(
        dim=16,
        num_envs=1,
        rollout_steps=4,
        minimum_steps=1,
        event_target=1,
        collection_mode="events",
        epochs=1,
        sil_updates=1,
        sil_batch_size=1,
    )
    trainer = GamepadPPO(config)
    original = trainer.envs[0]
    scripted = _ScriptedEnv(trainer.observations[0], [{"foods": 1}])
    original.close()
    trainer.envs = [scripted]
    trainer.observations = [scripted.reset()[0]]
    try:
        metrics = trainer.update()
        assert metrics["collection_stop"] == "food_target"
        assert metrics["completed_episodes"] == 0
        assert metrics["replay_episodes"] == 0
        assert metrics["replay_success_snapshots"] == 1
        assert metrics["replay_sampleable_sequences"] == 1
        assert metrics["sil_replay_active_success_snapshots"] == 1
        assert metrics["sil_sampled_success_sequences"] == 1
        assert metrics["sil_sampled_active_success_snapshots"] == 1
        assert metrics["sil_rows"] == 1
        assert len(trainer.replay.pending[0]) == 1
        assert len(trainer.replay.success_snapshots[0]) == 1

        payload = copy.deepcopy(trainer.checkpoint())
        assert "replay_success_snapshots" in payload
        assert "replay_pending" not in payload
        import runpy
        from pathlib import Path

        lightweight = runpy.run_path(
            Path(__file__).parents[1] / "scripts/train_gamepad_ppo.py"
        )["lightweight_policy_checkpoint"](payload)
        assert "model" in lightweight
        assert "optimizer" not in lightweight
        assert "replay_episodes" not in lightweight
        assert "replay_success_snapshots" not in lightweight
        assert lightweight["note"] == (
            "inference-only model checkpoint; no optimizer or replay state"
        )
        restored = GamepadPPO(config)
        try:
            restored.restore(payload)
            assert restored.replay.pending == [[]]
            assert restored.replay.success_snapshots == {}
            retained = [
                episode
                for episode in restored.replay.episodes
                if episode[0].get("sil_retained_success_fragment")
            ]
            assert len(retained) == 1
            assert len(retained[0]) == 1
            assert (
                restored.replay.last_sampling_metrics[
                    "sil_replay_completed_success_episodes"
                ]
                == 0
            )
            assert (
                restored.replay.last_sampling_metrics[
                    "sil_replay_retained_success_fragments"
                ]
                == 1
            )
            assert (
                restored.replay.last_sampling_metrics[
                    "sil_replay_active_success_snapshots"
                ]
                == 0
            )
            # The reset live game starts a fresh pending episode. Its terminal
            # must neither concatenate with nor delete the retained fragment.
            restored.replay.append(
                0,
                {
                    "reward": 0.0,
                    "discount": 0.0,
                    "food_events": [],
                    "score": 0,
                    "episode_id": 999,
                },
                True,
            )
            assert restored.replay.success_snapshots == {}
            assert len(restored.replay.episodes[-1]) == 1
            assert restored.replay.episodes[-1][0]["episode_id"] == 999
            retained = [
                episode
                for episode in restored.replay.episodes
                if episode[0].get("sil_retained_success_fragment")
            ]
            assert len(retained) == 1

            from unittest.mock import patch

            with patch("torch.multinomial", return_value=torch.tensor([0])):
                replay_result = restored.replay.update(
                    restored.model, restored.optimizer, "cpu", batch_size=1
                )
            assert replay_result["sil_rows"] == 1
            assert (
                restored.replay.last_sampling_metrics[
                    "sil_sampled_retained_success_fragments"
                ]
                == 1
            )
            assert (
                restored.replay.last_sampling_metrics[
                    "sil_sampled_active_success_snapshots"
                ]
                == 0
            )
            second_payload = copy.deepcopy(restored.checkpoint())
            restored_again = GamepadPPO(config)
            try:
                restored_again.restore(second_payload)
                assert restored_again.replay.success_snapshots == {}
                assert (
                    sum(
                        bool(episode[0].get("sil_retained_success_fragment", False))
                        for episode in restored_again.replay.episodes
                    )
                    == 1
                )
            finally:
                restored_again.close()
        finally:
            restored.close()
    finally:
        trainer.close()


def test_recurrent_windows_cover_a_300_step_rollout_without_crossing_resets():
    resets = torch.zeros(300, 2, dtype=torch.bool)
    resets[0, 0] = True
    resets[140, 0] = True
    resets[270, 0] = True
    # Environment 1 enters the rollout mid-episode, then resets independently.
    resets[73, 1] = True
    resets[202, 1] = True
    windows = _recurrent_windows(resets, burn_in=64, unroll=128)
    coverage = torch.zeros_like(resets, dtype=torch.int64)
    for window in windows:
        env = window["env"]
        burn_start = window["burn_start"]
        learn_start = window["learn_start"]
        learn_end = window["learn_end"]
        coverage[learn_start:learn_end, env] += 1
        assert learn_end - learn_start <= 128
        assert learn_start - burn_start <= 64
        # A reset at burn_start is allowed; another one would cross episodes.
        assert not resets[burn_start + 1 : learn_end, env].any()
    assert coverage.eq(1).all()


def test_long_rollout_uses_no_grad_burn_in_and_short_gradient_unrolls():
    torch.set_num_threads(1)
    config = PPOConfig(
        width=7,
        height=7,
        chunk_length=1,
        dim=8,
        num_envs=1,
        rollout_steps=300,
        recurrent_burn_in=64,
        recurrent_unroll=128,
        epochs=1,
    )
    trainer = GamepadPPO(config)
    original = trainer.envs[0]
    scripted = _ScriptedEnv(trainer.observations[0], [{} for _ in range(300)])
    original.close()
    trainer.envs = [scripted]
    trainer.observations = [scripted.reset()[0]]
    try:
        metrics = trainer.update()
        assert metrics["rollout_decisions_per_env"] == 300
        assert metrics["recurrent_samples_per_epoch"] == 300
        assert metrics["recurrent_window_count"] == 3
        assert metrics["recurrent_max_learn_steps"] == 128
        assert metrics["recurrent_max_burn_in_steps"] == 64
        assert metrics["recurrent_burn_in_steps_per_epoch"] == 128
        assert metrics["recurrent_windows_crossing_resets"] == 0
        assert metrics["recurrent_burn_in_grad_enabled"] is False
        assert metrics["recurrent_learn_grad_enabled"] is True
    finally:
        trainer.close()


def test_mixed_distribution_log_probability_and_execution_transform():
    buttons = torch.distributions.Bernoulli(logits=torch.zeros(2, 3, 14))
    axes = torch.distributions.Normal(torch.zeros(2, 3, 6), torch.ones(2, 3, 6))
    b, a = buttons.sample(), axes.sample()
    result = sample_log_prob((buttons, axes), b, a)
    expected = buttons.log_prob(b).flatten(1).sum(1) + axes.log_prob(a).flatten(1).sum(
        1
    )
    torch.testing.assert_close(result, expected)
    reports = reports_from_samples(b, a)
    assert reports.shape == (2, 3, 20)
    assert ((reports[..., :14] == 0) | (reports[..., :14] == 1)).all()
    assert reports[..., 14:18].abs().max() <= 1
    assert reports[..., 18:].min() >= 0


def test_ppo_update_changes_weights_with_actual_game_interaction():
    torch.set_num_threads(1)
    trainer = GamepadPPO(PPOConfig(dim=16, rollout_steps=4, num_envs=2, epochs=2))
    try:
        before = trainer.model.actor.weight.detach().clone()
        metrics = trainer.update()
        assert metrics["logic_ticks"] >= 8
        assert metrics["decisions"] == 8
        assert metrics["epochs"] >= 1
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert not torch.equal(before, trainer.model.actor.weight)
        assert trainer.checkpoint()["format"] == "gamepad-ppo-v3"
        metrics = trainer.update()
        assert metrics["decisions"] == 16
    finally:
        trainer.close()


def test_ppo_update_runs_explicit_phases_once_and_preserves_counts():
    torch.set_num_threads(1)
    config = PPOConfig(
        dim=16,
        chunk_length=2,
        rollout_steps=3,
        num_envs=2,
        epochs=1,
    )
    trainer = GamepadPPO(config)
    try:
        with (
            patch.object(
                trainer, "_collect_rollout", wraps=trainer._collect_rollout
            ) as collect,
            patch.object(
                trainer, "_optimize_ppo", wraps=trainer._optimize_ppo
            ) as optimize,
            patch.object(
                trainer, "_optimize_sil", wraps=trainer._optimize_sil
            ) as optimize_sil,
            patch.object(
                trainer, "_refresh_hidden", wraps=trainer._refresh_hidden
            ) as refresh,
            patch.object(
                trainer, "_assemble_metrics", wraps=trainer._assemble_metrics
            ) as assemble,
        ):
            metrics = trainer.update()

        assert collect.call_count == 1
        assert optimize.call_count == 1
        assert optimize_sil.call_count == 1
        assert refresh.call_count == 1
        assert assemble.call_count == 1
        assert metrics["decisions"] == config.num_envs * config.rollout_steps
        assert metrics["rollout_transitions"] == metrics["decisions"]
        assert metrics["recurrent_samples_per_epoch"] == metrics["decisions"]
        assert sum(metrics["sampled_direction_counts"]) == (
            config.num_envs * config.rollout_steps * config.chunk_length
        )
        assert metrics["logic_ticks"] == metrics["rollout_logic_ticks"]
        assert metrics["sample_policy_version"] == 0
        assert trainer.policy_version == 1
    finally:
        trainer.close()


def test_rollout_collection_runs_explicit_bookkeeping_phases_once_per_step():
    torch.set_num_threads(1)
    config = PPOConfig(
        dim=16,
        chunk_length=2,
        rollout_steps=3,
        num_envs=2,
        epochs=1,
    )
    trainer = GamepadPPO(config)
    try:
        with (
            patch.object(
                trainer, "_sample_policy_step", wraps=trainer._sample_policy_step
            ) as sample,
            patch.object(
                trainer,
                "_audit_sampled_reports",
                wraps=trainer._audit_sampled_reports,
            ) as audit,
            patch.object(
                trainer, "_step_environments", wraps=trainer._step_environments
            ) as transition,
            patch.object(trainer, "_rollout_row", wraps=trainer._rollout_row) as row,
            patch.object(
                trainer, "_archive_transitions", wraps=trainer._archive_transitions
            ) as archive,
            patch.object(
                trainer, "_finish_rollout_step", wraps=trainer._finish_rollout_step
            ) as finish,
            patch.object(
                trainer,
                "_record_batch_boundaries",
                wraps=trainer._record_batch_boundaries,
            ) as boundaries,
        ):
            phases = Mock()
            phases.attach_mock(sample, "sample")
            phases.attach_mock(audit, "audit")
            phases.attach_mock(transition, "transition")
            phases.attach_mock(row, "row")
            phases.attach_mock(archive, "archive")
            phases.attach_mock(finish, "finish")
            phases.attach_mock(boundaries, "boundaries")
            rollout = trainer._collect_rollout()

        assert len(rollout.data) == config.rollout_steps
        assert sample.call_count == config.rollout_steps
        assert audit.call_count == config.rollout_steps
        assert transition.call_count == config.rollout_steps
        assert row.call_count == config.rollout_steps
        assert archive.call_count == config.rollout_steps
        assert finish.call_count == config.rollout_steps
        assert boundaries.call_count == 1
        assert [call[0] for call in phases.mock_calls] == [
            phase
            for _ in range(config.rollout_steps)
            for phase in (
                "sample",
                "audit",
                "transition",
                "row",
                "archive",
                "finish",
            )
        ] + ["boundaries"]
    finally:
        trainer.close()


def test_dpad5_ppo_uses_legal_reports_and_distinct_checkpoint_semantics():
    torch.set_num_threads(1)
    config = PPOConfig(
        dim=16,
        chunk_length=2,
        rollout_steps=4,
        num_envs=2,
        epochs=1,
        action_head="dpad5",
    )
    trainer = GamepadPPO(config)
    try:
        before = trainer.model.actor.weight.detach().clone()
        metrics = trainer.update()
        checkpoint = trainer.checkpoint()

        assert not torch.equal(before, trainer.model.actor.weight)
        assert metrics["action_head"] == "dpad5"
        assert sum(metrics["sampled_direction_counts"]) == 16
        assert metrics["sampled_conflict_neutral_fraction"] == 0
        assert checkpoint["format"] == "gamepad-ppo-v4"
        assert checkpoint["training_objective"] == "snake-dpad5-categorical-v1"
        assert checkpoint["action_encoding"] == "xinput-dpad5-categorical-v1"
        validate_policy_checkpoint(checkpoint)
    finally:
        trainer.close()


def test_restore_rejects_cross_action_head_checkpoint():
    raw = GamepadPPO(PPOConfig(dim=16, num_envs=1, rollout_steps=1, epochs=1))
    structured = GamepadPPO(
        PPOConfig(
            dim=16,
            num_envs=1,
            rollout_steps=1,
            epochs=1,
            action_head="dpad5",
        )
    )
    try:
        with pytest.raises(ValueError, match="action head mismatch"):
            structured.restore(raw.checkpoint())
    finally:
        structured.close()
        raw.close()


def test_training_resume_rejects_pre_causal_objective_checkpoint():
    trainer = GamepadPPO(PPOConfig(dim=16, num_envs=1, rollout_steps=1, epochs=1))
    restored = GamepadPPO(PPOConfig(dim=16, num_envs=1, rollout_steps=1, epochs=1))
    try:
        payload = trainer.checkpoint()
        del payload["training_objective"]
        with pytest.raises(ValueError, match="training_objective mismatch"):
            restored.restore(payload)
    finally:
        restored.close()
        trainer.close()


def test_sequence_replay_matches_collection_before_update():
    trainer = GamepadPPO(PPOConfig(dim=16, num_envs=1, max_logic_steps=3))
    try:
        model = trainer.model
        obs = stack_observations(trainer.observations, "cpu")
        initial = torch.zeros(1, 16)
        first, _, hidden = distribution_from_observation(model, obs, initial)
        button, axis = (d.sample() for d in first)
        next_obs, _, _, _, _ = trainer.envs[0].step(
            reports_from_samples(button, axis)[0].numpy()
        )
        following = stack_observations([next_obs], "cpu")
        second, _, _ = distribution_from_observation(model, following, hidden)
        replay_first, _, replay_hidden = distribution_from_observation(
            model, obs, initial
        )
        replay_second, _, _ = distribution_from_observation(
            model, following, replay_hidden
        )
        torch.testing.assert_close(first[0].logits, replay_first[0].logits)
        torch.testing.assert_close(second[1].loc, replay_second[1].loc)
    finally:
        trainer.close()


def test_history_order_and_padding_affect_only_valid_reports():
    torch.manual_seed(3)
    model = GamepadPolicy(dim=16, chunk_length=2)
    board = torch.zeros(1, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    history = torch.zeros(1, 3, 20)
    history[:, 0, 0] = 1
    history[:, 1, 1] = 1
    context = {
        "history": history,
        "history_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "previous_chunk": torch.zeros(1, 2, 20),
        "previous_age": torch.zeros(1, 1),
    }
    args = (board, torch.zeros(1, 20), torch.zeros(1, 3))
    first, _, _ = model.distribution(*args, **context)
    history[:, 2] = 99  # padding must not enter feedback state
    second, _, _ = model.distribution(*args, **context)
    torch.testing.assert_close(first[0].logits, second[0].logits)
    context["history"] = history[:, [1, 0, 2]]
    third, _, _ = model.distribution(*args, **context)
    assert not torch.equal(first[0].logits, third[0].logits)

"""Core recurrent PPO collection and update regressions."""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo import GamepadPPO, collection_stop  # noqa: E402
from resnake_gym.gamepad_ppo_contract import PPOConfig  # noqa: E402
from resnake_gym.training.ppo_update import (  # noqa: E402
    generalized_advantages,
    recurrent_windows,
)


def test_event_collection_stops_for_food_or_the_hard_limit_not_for_death():
    config = PPOConfig(
        rollout_steps=7,
        minimum_steps=3,
        event_target=1,
        collection_mode="events",
    )

    assert collection_stop(config, steps=2, foods=1, terminals=0) is None
    assert collection_stop(config, steps=3, foods=0, terminals=20) is None
    assert collection_stop(config, steps=3, foods=1, terminals=20) == "food_target"
    assert collection_stop(config, steps=7, foods=0, terminals=20) == "hard_limit"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sil_updates", -1, "update count"),
        ("sil_weight", -0.1, "weight"),
        ("sil_batch_size", 0, "batch size"),
        ("sil_priority_alpha", -0.1, "priority alpha"),
        ("sil_priority_alpha", 1.1, "priority alpha"),
    ],
)
def test_trainer_rejects_invalid_sil_settings_before_collection(field, value, message):
    values = {
        "width": 7,
        "height": 6,
        "dim": 8,
        "num_envs": 1,
        "rollout_steps": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        GamepadPPO(PPOConfig(**values))


def test_gae_uses_variable_tick_discounts_and_does_not_cross_resets():
    rewards = torch.tensor([[1.0], [10.0], [100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 8.0)
    discounts = torch.tensor([[0.5**2], [0.5**3], [0.5**5]])
    terminated = torch.tensor([[False], [True], [False]])
    truncated = torch.tensor([[True], [False], [False]])

    result = generalized_advantages(
        rewards,
        values,
        next_values,
        discounts,
        terminated,
        truncated,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(result, torch.tensor([[3.0], [10.0], [100.25]]))


def test_recurrent_windows_cover_every_sample_without_crossing_a_reset():
    resets = torch.zeros(300, 2, dtype=torch.bool)
    resets[0, 0] = True
    resets[140, 0] = True
    resets[270, 0] = True
    resets[73, 1] = True
    resets[202, 1] = True

    windows = recurrent_windows(resets, burn_in=64, unroll=128)
    coverage = torch.zeros_like(resets, dtype=torch.int64)
    for window in windows:
        env = window["env"]
        burn_start = window["burn_start"]
        learn_start = window["learn_start"]
        learn_end = window["learn_end"]
        coverage[learn_start:learn_end, env] += 1
        assert learn_end - learn_start <= 128
        assert learn_start - burn_start <= 64
        assert not resets[burn_start + 1 : learn_end, env].any()
    assert coverage.eq(1).all()


def test_one_update_runs_each_training_phase_once():
    torch.set_num_threads(1)
    config = PPOConfig(
        width=7,
        height=6,
        dim=8,
        chunk_length=1,
        num_envs=1,
        rollout_steps=2,
        recurrent_burn_in=1,
        recurrent_unroll=2,
        epochs=1,
        sil_updates=0,
        collection_mode="fixed",
        perturbation_min=100,
        perturbation_max=100,
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
        ):
            metrics = trainer.update()

        assert collect.call_count == 1
        assert optimize.call_count == 1
        assert optimize_sil.call_count == 1
        assert refresh.call_count == 1
        assert metrics["rollout_transitions"] == 2
        assert metrics["recurrent_samples_per_epoch"] == 2
        assert trainer.policy_version == 1
    finally:
        trainer.close()

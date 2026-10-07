import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo import GamepadPPO, PPOConfig  # noqa: E402
from resnake_gym.sil import SequenceReplay  # noqa: E402


def test_returns_keep_variable_tick_discounts_and_bootstrap():
    replay = SequenceReplay(1)
    replay.append(0, {"reward": 1.0, "discount": 0.5}, False)
    replay.append(0, {"reward": 2.0, "discount": 0.25}, True, bootstrap=4.0)
    assert [r["return"] for r in replay.episodes[0]] == [2.5, 3.0]


def test_archive_sil_update_and_restore(tmp_path):
    torch.set_num_threads(1)
    c = PPOConfig(
        dim=16, num_envs=1, rollout_steps=3, epochs=1, max_logic_steps=1, sil_updates=1
    )
    trainer = GamepadPPO(c, archive=tmp_path / "archive")
    try:
        training_metrics = trainer.update()
        assert training_metrics["sil_replay_failure_episodes"] == 3
        assert training_metrics["sil_replay_success_sequences"] == 0
        assert training_metrics["sil_sampled_failure_episodes"] == 1
        assert len(list((tmp_path / "archive").glob("*.pt"))) == 3
        # Controlled unit fixture verifies gradients, not training demonstrations.
        for episode in trainer.replay.episodes:
            for row in episode:
                row["return"] = 2.0
        before = trainer.model.actor.weight.detach().clone()
        result = trainer.replay.update(trainer.model, trainer.optimizer, "cpu")
        assert result["sil_valid"] > 0
        assert not torch.equal(before, trainer.model.actor.weight)
        assert trainer.replay.last_sampling_metrics == {
            "sil_replay_success_sequences": 0,
            "sil_replay_completed_success_episodes": 0,
            "sil_replay_retained_success_fragments": 0,
            "sil_replay_active_success_snapshots": 0,
            "sil_replay_failure_episodes": 3,
            "sil_replay_sampleable_sequences": 3,
            "sil_target_success_fraction": 0.0,
            "sil_sampled_success_sequences": 0,
            "sil_sampled_active_success_snapshots": 0,
            "sil_sampled_retained_success_fragments": 0,
            "sil_sampled_failure_episodes": 1,
            "sil_sampled_success_fraction": 0.0,
        }
        copy = GamepadPPO(c)
        try:
            copy.restore(trainer.checkpoint())
            assert copy.decisions == trainer.decisions
            assert len(copy.replay.episodes) == 3
            assert copy.replay.last_sampling_metrics["sil_replay_failure_episodes"] == 3
        finally:
            copy.close()
    finally:
        trainer.close()


def _episode(length, *, success=False, priority=1.0, label="episode"):
    return [
        {
            "sil_priority": priority,
            "food_events": [{"tick": 1}] if success and index == 0 else [],
            "label": label,
        }
        for index in range(length)
    ]


def test_stratified_target_and_within_stratum_importance_correction():
    replay = SequenceReplay(1)
    replay.episodes = [
        _episode(1, success=True, priority=0.01),
        _episode(2, success=True, priority=10.0),
        _episode(3, priority=0.01),
        _episode(1, priority=10.0),
    ]
    probabilities, correction = replay.sampling_distribution()
    target = torch.tensor([1 / 6, 1 / 3, 3 / 8, 1 / 8])
    assert torch.allclose(probabilities[:2].sum(), torch.tensor(0.5))
    assert torch.allclose(probabilities[2:].sum(), torch.tensor(0.5))
    assert probabilities[1] > target[1]
    assert torch.allclose(probabilities * correction, target)
    assert correction.max() <= 10
    uniform, weights = replay.sampling_distribution(alpha=0)
    assert torch.allclose(uniform, target)
    assert torch.allclose(weights, torch.ones(4))


def test_success_is_protected_from_failure_fifo():
    replay = SequenceReplay(1, capacity=3)

    def append(label, *, food=False, won=False):
        replay.append(
            0,
            {
                "reward": 1.0 if food else 0.0,
                "discount": 0.0,
                "food_events": [{"tick": 1}] if food else [],
                "won": won,
                "label": label,
            },
            True,
        )

    append("food", food=True)
    append("old-failure")
    append("newer-failure")
    for index in range(5):
        append(f"failure-{index}")
    assert len(replay.episodes) == 3
    assert "food" in [episode[0]["label"] for episode in replay.episodes]
    assert sum(replay.is_success_episode(e) for e in replay.episodes) == 1

    # A newly discovered strict win evicts a failure, not the older food success.
    append("win", won=True)
    assert {episode[0]["label"] for episode in replay.episodes} >= {"food", "win"}
    assert sum(replay.is_success_episode(e) for e in replay.episodes) == 2


def test_latest_food_snapshot_is_immediate_bounded_and_archived(tmp_path):
    replay = SequenceReplay(2, capacity=1, archive=tmp_path / "archive")
    replay.append(
        1,
        {"reward": 0.0, "discount": 0.0, "food_events": [], "label": "failure"},
        True,
    )
    replay.append(
        0,
        {"reward": 1.0, "discount": 0.5, "food_events": [], "label": "start"},
        False,
    )
    replay.append(
        0,
        {
            "reward": 2.0,
            "discount": 0.25,
            "food_events": [{"tick": 2}],
            "label": "first-food",
        },
        False,
        bootstrap=4.0,
    )
    first = replay.success_snapshots[0]
    assert [row["return"] for row in first] == [2.5, 3.0]
    assert first is not replay.pending[0]
    assert first[0] is not replay.pending[0][0]
    assert len(replay.episodes) == 1  # Snapshot does not consume completed capacity.
    assert replay.last_sampling_metrics["sil_replay_active_success_snapshots"] == 1
    path = tmp_path / "archive" / "pending-success-env-0000.pt"
    assert path.exists()

    replay.append(
        0,
        {
            "reward": 3.0,
            "discount": 0.5,
            "food_events": [{"tick": 3}],
            "label": "latest-food",
        },
        False,
        bootstrap=2.0,
    )
    assert len(replay.success_snapshots) == 1
    assert replay.success_snapshots[0] is not first
    assert len(replay.success_snapshots[0]) == 3
    assert len(list((tmp_path / "archive").glob("pending-success-env-*.pt"))) == 1

    replay.append(
        0,
        {
            "reward": -1.0,
            "discount": 0.0,
            "food_events": [],
            "label": "terminal",
        },
        True,
    )
    assert replay.success_snapshots == {}
    assert not path.exists()
    assert len(replay.episodes) == 1
    assert [row["label"] for row in replay.episodes[0]] == [
        "start",
        "first-food",
        "latest-food",
        "terminal",
    ]
    assert (tmp_path / "archive" / "episode-00000001.pt").exists()


def test_zero_success_falls_back_to_failure_distribution():
    replay = SequenceReplay(1)
    replay.episodes = [_episode(1, priority=0.01), _episode(2, priority=10.0)]
    probabilities, correction = replay.sampling_distribution(alpha=0)
    assert torch.allclose(probabilities, torch.tensor([1 / 3, 2 / 3]))
    assert torch.allclose(correction, torch.ones(2))

    empty = SequenceReplay(1)
    probabilities, correction = empty.sampling_distribution()
    assert probabilities.numel() == correction.numel() == 0


def test_batched_replay_padding_and_long_prefix():
    import copy

    torch.set_num_threads(1)
    trainer = GamepadPPO(
        PPOConfig(dim=16, num_envs=1, rollout_steps=2, epochs=1, max_logic_steps=1)
    )
    try:
        trainer.update()
        row = trainer.replay.episodes[0][0]
        row["return"] = 2.0
        successful = [copy.deepcopy(row)]
        successful[0]["sil_success"] = True
        successful[0]["food_events"] = [{"tick": 1}]
        failure = [copy.deepcopy(row) for _ in range(130)]
        for replay_row in failure:
            replay_row["sil_success"] = False
            replay_row["food_events"] = []
        trainer.replay.episodes = [successful, failure]
        # Force both lengths into a batch; exercise masked padding and TBPTT boundary.
        from unittest.mock import patch

        with patch("torch.multinomial", return_value=torch.tensor([0, 1])):
            result = trainer.replay.update(
                trainer.model,
                trainer.optimizer,
                "cpu",
                batch_size=2,
                priority_alpha=0.6,
            )
        assert result["sil_rows"] == 131
        assert result["sil_valid"] == 131
        assert all(e[0]["sil_priority"] > 0 for e in trainer.replay.episodes)
        assert trainer.replay.last_sampling_metrics == {
            "sil_replay_success_sequences": 1,
            "sil_replay_completed_success_episodes": 1,
            "sil_replay_retained_success_fragments": 0,
            "sil_replay_active_success_snapshots": 0,
            "sil_replay_failure_episodes": 1,
            "sil_replay_sampleable_sequences": 2,
            "sil_target_success_fraction": 0.5,
            "sil_sampled_success_sequences": 1,
            "sil_sampled_active_success_snapshots": 0,
            "sil_sampled_retained_success_fragments": 0,
            "sil_sampled_failure_episodes": 1,
            "sil_sampled_success_fraction": 0.5,
        }
    finally:
        trainer.close()

"""Core regressions for autonomous success replay."""

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_ppo import GamepadPPO  # noqa: E402
from resnake_gym.gamepad_ppo_contract import PPOConfig  # noqa: E402
from resnake_gym.sil import SequenceReplay  # noqa: E402


def _transition(label, *, reward=0.0, discount=0.0, food=False, won=False):
    return {
        "label": label,
        "reward": reward,
        "discount": discount,
        "food_events": [{"tick": 1}] if food else [],
        "score": int(food),
        "won": won,
    }


def test_returns_use_each_transition_discount_and_terminal_bootstrap():
    replay = SequenceReplay(1)
    replay.append(
        0,
        _transition("first", reward=1.0, discount=0.5),
        False,
    )
    replay.append(
        0,
        _transition("second", reward=2.0, discount=0.25),
        True,
        bootstrap=4.0,
    )

    assert [row["return"] for row in replay.episodes[0]] == [2.5, 3.0]


def test_failures_cannot_evict_a_retained_success():
    replay = SequenceReplay(1, capacity=3)
    replay.append(0, _transition("success", food=True), True)
    for index in range(6):
        replay.append(0, _transition(f"failure-{index}"), True)

    labels = [episode[0]["label"] for episode in replay.episodes]
    assert len(labels) == 3
    assert "success" in labels
    assert sum(replay.is_success_episode(item) for item in replay.episodes) == 1


def test_food_creates_an_immediate_bounded_success_snapshot():
    replay = SequenceReplay(1)
    replay.append(
        0,
        _transition("start", reward=1.0, discount=0.5),
        False,
    )
    replay.append(
        0,
        _transition("first-food", reward=2.0, discount=0.25, food=True),
        False,
        bootstrap=4.0,
    )

    first = replay.success_snapshots[0]
    assert [row["return"] for row in first] == [2.5, 3.0]
    assert first is not replay.pending[0]
    assert replay.last_sampling_metrics["sil_replay_active_success_snapshots"] == 1

    replay.append(
        0,
        _transition("newer-food", reward=3.0, discount=0.5, food=True),
        False,
        bootstrap=2.0,
    )
    assert len(replay.success_snapshots) == 1
    assert len(replay.success_snapshots[0]) == 3
    assert replay.success_snapshots[0] is not first


def test_checkpoint_promotes_a_live_success_snapshot_to_retained_replay():
    config = PPOConfig(
        width=7,
        height=6,
        dim=8,
        num_envs=1,
        rollout_steps=1,
        epochs=1,
    )
    trainer = GamepadPPO(config)
    restored = GamepadPPO(config)
    try:
        trainer.replay.append(
            0,
            _transition("food", reward=1.0, discount=0.5, food=True),
            False,
            bootstrap=2.0,
        )
        payload = trainer.checkpoint()

        restored.restore(payload)

        assert restored.replay.pending == [[]]
        assert restored.replay.success_snapshots == {}
        assert len(restored.replay.episodes) == 1
        retained = restored.replay.episodes[0][0]
        assert retained["sil_success"] is True
        assert retained["sil_retained_success_fragment"] is True
        assert retained["sil_snapshot_active"] is False
    finally:
        restored.close()
        trainer.close()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"weight": -0.1}, "weight"),
        ({"batch_size": 0}, "batch size"),
        ({"priority_alpha": -0.1}, "priority alpha"),
        ({"priority_alpha": 1.1}, "priority alpha"),
    ],
)
def test_sil_rejects_invalid_optimization_settings_before_sampling(arguments, message):
    replay = SequenceReplay(1)

    with pytest.raises(ValueError, match=message):
        replay.update(None, None, "cpu", **arguments)

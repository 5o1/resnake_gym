import math

import pytest

from resnake_gym.progress_replay import (
    PROGRESS_STRATA,
    AutonomousProgressReplay,
    progress_stratum,
)


def _transition(index, *, policy_version=3):
    return {
        "observation": index,
        "behavior_logp": -0.1 * (index + 1),
        "controller_tick": index,
        "execution_ticks": [index],
        "policy_version": policy_version,
    }


def _append(
    replay,
    score,
    *,
    won=False,
    contains_food_event=False,
    policy_version=3,
    item_kind="episode",
):
    return replay.append_autonomous(
        [_transition(score, policy_version=policy_version)],
        score_end=score,
        won=won,
        contains_food_event=contains_food_event,
        policy_version=policy_version,
        item_kind=item_kind,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "0"),
        (1, "1-4"),
        (4, "1-4"),
        (5, "5-9"),
        (9, "5-9"),
        (10, "10-24"),
        (24, "10-24"),
        (25, "25-49"),
        (49, "25-49"),
        (50, "50-99"),
        (99, "50-99"),
        (100, "100+"),
        (1000, "100+"),
    ],
)
def test_fixed_score_strata(score, expected):
    assert progress_stratum(score) == expected


def test_win_overrides_score_and_first_fruit_is_not_named_success():
    assert progress_stratum(0, won=True) == "win"
    replay = AutonomousProgressReplay(8)
    _append(replay, 1, item_kind="fragment")
    metadata = replay.items("1-4")[0].metadata

    assert metadata.data_origin == "autonomous_rl"
    assert metadata.item_kind == "fragment"
    assert metadata.positive_progress
    assert not metadata.event_bearing
    assert not metadata.high_progress
    assert not hasattr(metadata, "success")


@pytest.mark.parametrize("retention", ["fifo", "reservoir"])
def test_event_retention_is_not_faked_or_evicted_by_positive_score(retention):
    replay = AutonomousProgressReplay(16, retention=retention, seed=13)
    for _ in range(4):
        _append(replay, 1, contains_food_event=False)
    event = _append(replay, 1, contains_food_event=True)
    for _ in range(50):
        _append(replay, 1, contains_food_event=False)

    retained = replay.items("1-4")
    assert event.item_id in {item.metadata.item_id for item in retained}
    assert sum(item.metadata.contains_food_event for item in retained) == 1
    batch = replay.sample(20, event_fraction=1.0)
    assert {sample.item.metadata.item_id for sample in batch.samples} == {event.item_id}


def test_score_positive_without_food_is_not_an_event_sample():
    replay = AutonomousProgressReplay(16, retention="fifo", seed=17)
    no_event = _append(replay, 3, contains_food_event=False)
    event = _append(replay, 3, contains_food_event=True)

    batch = replay.sample(20, event_fraction=1.0)

    assert no_event.stratum == event.stratum == "1-4"
    assert {sample.item.metadata.item_id for sample in batch.samples} == {event.item_id}


@pytest.mark.parametrize("bad", [-1, -100])
def test_negative_score_is_rejected(bad):
    with pytest.raises(ValueError, match="non-negative"):
        progress_stratum(bad)


@pytest.mark.parametrize("bad", [True, 1.5, "1"])
def test_non_integer_score_is_rejected(bad):
    with pytest.raises(TypeError, match="score"):
        progress_stratum(bad)


def test_fixed_stratum_quotas_protect_rare_progress_and_bound_memory():
    replay = AutonomousProgressReplay(8, retention="fifo")
    win = _append(replay, 0, won=True)
    latest_zero = None
    for _ in range(100):
        latest_zero = _append(replay, 0)

    assert replay.stratum_capacities == {name: 1 for name in PROGRESS_STRATA}
    assert replay.stratum_sizes["0"] == 1
    assert replay.stratum_sizes["win"] == 1
    assert replay.seen_counts["0"] == 100
    assert replay.seen_counts["win"] == 1
    assert len(replay) == 2 <= replay.capacity
    assert replay.items("win")[0].metadata.item_id == win.item_id
    assert replay.items("0")[0].metadata.item_id == latest_zero.item_id


def test_reservoir_is_seeded_and_reports_historical_inclusion_probability():
    left = AutonomousProgressReplay(8, retention="reservoir", seed=19)
    right = AutonomousProgressReplay(8, retention="reservoir", seed=19)
    for score in [0] * 20 + [25] * 10:
        _append(left, score)
        _append(right, score)

    assert [item.metadata.item_id for item in left.items()] == [
        item.metadata.item_id for item in right.items()
    ]
    batch = left.sample(20)
    for sample in batch.samples:
        expected = 1 / left.seen_counts[sample.item.metadata.stratum]
        assert sample.reservoir_inclusion_probability == pytest.approx(expected)


def test_custom_capacities_must_cover_every_stratum_and_sum_to_total():
    with pytest.raises(ValueError, match=">= 8"):
        AutonomousProgressReplay(7)
    with pytest.raises(ValueError, match="every stratum"):
        AutonomousProgressReplay(8, stratum_capacities={"0": 8})
    capacities = {name: 1 for name in PROGRESS_STRATA}
    capacities["0"] = 2
    with pytest.raises(ValueError, match="sum"):
        AutonomousProgressReplay(8, stratum_capacities=capacities)


def test_metadata_presence_is_audited_and_default_sampling_is_safe():
    replay = AutonomousProgressReplay(16, retention="fifo")
    source = [_transition(0), _transition(1)]
    complete = replay.append_autonomous(
        source,
        score_end=5,
        won=False,
        contains_food_event=True,
        policy_version=3,
        item_kind="episode",
    )
    incomplete = replay.append_autonomous(
        [{"observation": "missing metadata"}],
        score_end=6,
        won=False,
        contains_food_event=False,
        policy_version=3,
        item_kind="fragment",
    )
    source[0]["observation"] = "mutated after insertion"

    stored = {item.metadata.item_id: item for item in replay.items("5-9")}
    complete_metadata = stored[complete.item_id].metadata
    incomplete_metadata = stored[incomplete.item_id].metadata
    assert complete_metadata.off_policy_ready
    assert complete_metadata.policy_version == 3
    assert complete_metadata.transition_count == 2
    assert stored[complete.item_id].transitions[0]["observation"] == 0
    assert not incomplete_metadata.behavior_logp_present
    assert not incomplete_metadata.timestamps_present
    assert not incomplete_metadata.decision_duration_present

    safe = replay.sample(10)
    assert safe.eligible_count == 1
    assert {sample.item.metadata.item_id for sample in safe.samples} == {
        complete.item_id
    }
    audit = replay.sample(
        10,
        require_behavior_logp=False,
        require_timestamps=False,
        require_decision_duration=False,
    )
    assert audit.eligible_count == 2


def test_transition_policy_version_mismatch_is_rejected():
    replay = AutonomousProgressReplay(8)
    with pytest.raises(ValueError, match="policy_version mismatch"):
        replay.append_autonomous(
            [_transition(0, policy_version=2)],
            score_end=0,
            won=False,
            contains_food_event=False,
            policy_version=3,
            item_kind="episode",
        )


def test_state_dict_roundtrip_preserves_items_reservoir_counts_and_rng():
    original = AutonomousProgressReplay(16, retention="reservoir", seed=41)
    for score in (0, 0, 0):
        _append(original, score)
    for score in (1, 5, 25, 100):
        _append(original, score, contains_food_event=True)
    restored = AutonomousProgressReplay(16, retention="reservoir", seed=999)

    restored.load_state_dict(original.state_dict())

    assert restored.stratum_sizes == original.stratum_sizes
    assert restored.seen_counts == original.seen_counts
    assert restored.event_seen_counts == original.event_seen_counts
    assert restored.non_event_seen_counts == original.non_event_seen_counts
    assert restored.event_size == original.event_size
    assert [item.metadata.item_id for item in restored.items()] == [
        item.metadata.item_id for item in original.items()
    ]
    left = original.sample(20, event_fraction=0.75)
    right = restored.sample(20, event_fraction=0.75)
    assert [sample.item.metadata.item_id for sample in left.samples] == [
        sample.item.metadata.item_id for sample in right.samples
    ]


def test_state_dict_rejects_capacity_or_retention_changes():
    original = AutonomousProgressReplay(16, retention="reservoir")
    _append(original, 1)
    with pytest.raises(ValueError, match="capacity mismatch"):
        AutonomousProgressReplay(24).load_state_dict(original.state_dict())
    with pytest.raises(ValueError, match="retention mismatch"):
        AutonomousProgressReplay(16, retention="fifo").load_state_dict(
            original.state_dict()
        )


def test_state_dict_rejects_old_event_ambiguous_semantics():
    replay = AutonomousProgressReplay(16)
    _append(replay, 1, contains_food_event=True)
    state = replay.state_dict()
    state["format"] = "autonomous-progress-replay-v1"

    with pytest.raises(ValueError, match="format mismatch"):
        AutonomousProgressReplay(16).load_state_dict(state)


def test_progress_mixture_returns_exact_proposals_and_uniform_is_weights():
    replay = AutonomousProgressReplay(24, retention="fifo", seed=7)
    for score in (0, 0):
        _append(replay, score)
    for score in (1, 2, 25, 26):
        _append(replay, score, contains_food_event=True)

    batch = replay.sample(
        200,
        event_fraction=0.75,
        high_progress_fraction=0.25,
    )
    assert batch.eligible_count == 6
    assert batch.support_size == 6
    assert dict(batch.effective_group_masses) == pytest.approx(
        {"non_event": 0.25, "event_low": 0.5, "high_event": 0.25}
    )
    proposals = {
        "non_event": 0.25 / 2,
        "event_low": 0.5 / 2,
        "high_event": 0.25 / 2,
    }
    for sample in batch.samples:
        expected_proposal = proposals[sample.proposal_group]
        assert sample.proposal_probability == pytest.approx(expected_proposal)
        assert sample.uniform_target_probability == pytest.approx(1 / 6)
        assert sample.importance_weight == pytest.approx((1 / 6) / expected_proposal)
        assert sample.proposal_probability * sample.importance_weight == (
            pytest.approx(sample.uniform_target_probability)
        )


def test_empty_requested_group_is_renormalized_or_strictly_rejected():
    replay = AutonomousProgressReplay(24, retention="fifo", seed=11)
    for score in (0, 0):
        _append(replay, score)
    for score in (1, 2):
        _append(replay, score, contains_food_event=True)

    batch = replay.sample(
        100,
        event_fraction=0.8,
        high_progress_fraction=0.2,
    )
    assert dict(batch.effective_group_masses) == pytest.approx(
        {"non_event": 0.25, "event_low": 0.75}
    )
    for sample in batch.samples:
        expected = 0.25 / 2 if sample.proposal_group == "non_event" else 0.75 / 2
        assert sample.proposal_probability == pytest.approx(expected)

    with pytest.raises(ValueError, match="high"):
        replay.sample(
            1,
            event_fraction=0.8,
            high_progress_fraction=0.2,
            strict_group_fractions=True,
        )


@pytest.mark.parametrize(
    ("positive", "high"),
    [(-0.1, None), (1.1, None), (None, math.nan), (0.2, 0.3)],
)
def test_invalid_sampling_fractions_are_rejected(positive, high):
    replay = AutonomousProgressReplay(8)
    _append(replay, 0)
    with pytest.raises((TypeError, ValueError)):
        replay.sample(
            1,
            event_fraction=positive,
            high_progress_fraction=high,
        )

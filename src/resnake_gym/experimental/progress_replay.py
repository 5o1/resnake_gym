"""Bounded progress-stratified replay for isolated off-policy experiments.

This module is deliberately standalone.  It does not modify the current SIL
buffer or any trainer, and it accepts only autonomous RL episodes/fragments.
In particular, score 1 is merely the ``1-4`` progress stratum; it is not named
or treated as task success.  ``score_end`` and ``contains_food_event`` are
stored independently: prior cumulative progress cannot masquerade as an event
which occurred inside the retained item.

Retention uses a fixed quota for each of eight strata and never borrows an
unused quota.  That choice can leave space unused when some strata have never
been observed, but it prevents a stream of score-zero items from evicting a
rare high-progress item.  Event-bearing items have priority within a score
stratum; event and non-event classes each use the remaining class capacity.
Within each class, callers choose either:

* ``reservoir`` (default): Algorithm R-style replacement, so the bucket is a
  uniform sample of all items offered to that stratum; or
* ``fifo``: retain the most recent items in that stratum.

The reservoir design follows Vitter, "Random Sampling with a Reservoir"
(ACM TOMS 1985): https://doi.org/10.1145/3147.3165.  This is a simple
Algorithm R implementation, not Vitter's faster Algorithm Z.

Future V-trace use motivates recording behaviour log-probability availability:
IMPALA forms policy importance ratios from target and behaviour probabilities
(Espeholt et al., 2018): https://arxiv.org/abs/1802.01561.  LASER studies
actor-critic learning with large shared replay and warns that strongly
off-policy V-trace also needs on-policy mixing/relevance control (Schmitt et
al., 2020): https://arxiv.org/abs/1909.11583.  This store implements neither
V-trace, trust regions, nor LASER's learner.  Its returned importance weight
corrects only the *replay item sampling distribution* to uniform over the
current proposal support.  A learner must separately form target/behaviour
policy ratios from the stored ``behavior_logp`` values.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

ProgressStratum = Literal[
    "0",
    "1-4",
    "5-9",
    "10-24",
    "25-49",
    "50-99",
    "100+",
    "win",
]
ReplayItemKind = Literal["episode", "fragment"]
RetentionMode = Literal["reservoir", "fifo"]

PROGRESS_STRATA: tuple[ProgressStratum, ...] = (
    "0",
    "1-4",
    "5-9",
    "10-24",
    "25-49",
    "50-99",
    "100+",
    "win",
)
HIGH_PROGRESS_STRATA = frozenset({"25-49", "50-99", "100+", "win"})

# Presence is deliberately audited rather than inferred from ``discount``.
# A discount can hide termination as well as duration and is not a timestamp.
TIMESTAMP_FIELDS = (
    "timestamp_ns",
    "controller_tick",
    "capture_tick",
    "submitted_ns",
    "applied_ns",
    "observation_captured_ns",
    "state_captured_ns",
)
DECISION_DURATION_FIELDS = (
    "decision_duration",
    "decision_duration_s",
    "duration_ticks",
    "delta_ticks",
    "delta_t",
    "execution_ticks",
    "ticks_advanced",
)


def progress_stratum(score: int, *, won: bool = False) -> ProgressStratum:
    """Return the fixed progress stratum; a strict win overrides score."""
    if type(won) is not bool:
        raise TypeError("won must be a bool")
    if type(score) is not int:
        raise TypeError("score must be an int")
    if score < 0:
        raise ValueError("score must be non-negative")
    if won:
        return "win"
    if score == 0:
        return "0"
    if score <= 4:
        return "1-4"
    if score <= 9:
        return "5-9"
    if score <= 24:
        return "10-24"
    if score <= 49:
        return "25-49"
    if score <= 99:
        return "50-99"
    return "100+"


@dataclass(frozen=True, slots=True)
class ProgressReplayMetadata:
    """Auditable item-level facts; no field labels an item as success."""

    item_id: int
    data_origin: Literal["autonomous_rl"]
    item_kind: ReplayItemKind
    score_end: int
    won: bool
    contains_food_event: bool
    stratum: ProgressStratum
    policy_version: int
    transition_count: int
    behavior_logp_present: bool
    timestamps_present: bool
    decision_duration_present: bool

    @property
    def off_policy_ready(self) -> bool:
        """Whether every transition has the three audited metadata classes."""
        return (
            self.behavior_logp_present
            and self.timestamps_present
            and self.decision_duration_present
        )

    @property
    def positive_progress(self) -> bool:
        return self.won or self.score_end > 0

    @property
    def event_bearing(self) -> bool:
        """Whether this item itself contains an observed food event."""
        return self.contains_food_event

    @property
    def high_progress(self) -> bool:
        return self.stratum in HIGH_PROGRESS_STRATA


@dataclass(frozen=True, slots=True)
class ProgressReplayItem:
    metadata: ProgressReplayMetadata
    transitions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ReplayInsertResult:
    item_id: int
    stratum: ProgressStratum
    stored: bool
    replaced_item_id: int | None


@dataclass(frozen=True, slots=True)
class ProgressReplaySample:
    """One with-replacement draw and its exact current-store proposal."""

    item: ProgressReplayItem
    proposal_group: str
    proposal_probability: float
    uniform_target_probability: float
    importance_weight: float
    reservoir_inclusion_probability: float | None


@dataclass(frozen=True, slots=True)
class ProgressReplayBatch:
    samples: tuple[ProgressReplaySample, ...]
    eligible_count: int
    support_size: int
    requested_event_fraction: float | None
    requested_high_progress_fraction: float | None
    effective_group_masses: tuple[tuple[str, float], ...]


class AutonomousProgressReplay:
    """A bounded, per-stratum autonomous replay store.

    ``event_fraction`` in :meth:`sample` means mass assigned to items which
    themselves contain a food event.  A positive ``score_end`` alone never
    puts an item in that group.  ``high_progress_fraction`` is the event-bearing
    subset whose cumulative score is at least 25 or which is a strict win.
    Fractions define the proposal mass of independent draws with replacement;
    a finite batch need not contain those exact empirical proportions.

    Proposal probabilities are conditional on the currently realised store.
    Event-bearing items have retention priority inside each score stratum.
    Under reservoir retention, event and non-event items each form a uniform
    reservoir over the capacity still available to that class; samples report
    that class-conditional marginal inclusion probability.  FIFO has no
    uniform historical inclusion probability, so that field is ``None``.
    """

    def __init__(
        self,
        capacity: int,
        *,
        stratum_capacities: Mapping[str, int] | None = None,
        retention: RetentionMode = "reservoir",
        seed: int = 0,
    ) -> None:
        if type(capacity) is not int or capacity < len(PROGRESS_STRATA):
            raise ValueError(f"capacity must be an int >= {len(PROGRESS_STRATA)}")
        if retention not in ("reservoir", "fifo"):
            raise ValueError("retention must be reservoir or fifo")
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        self.capacity = capacity
        self.retention = retention
        self._capacities = self._resolve_capacities(capacity, stratum_capacities)
        self._buckets: dict[ProgressStratum, list[ProgressReplayItem]] = {
            stratum: [] for stratum in PROGRESS_STRATA
        }
        self._seen: dict[ProgressStratum, int] = {
            stratum: 0 for stratum in PROGRESS_STRATA
        }
        self._event_seen: dict[ProgressStratum, int] = {
            stratum: 0 for stratum in PROGRESS_STRATA
        }
        self._non_event_seen: dict[ProgressStratum, int] = {
            stratum: 0 for stratum in PROGRESS_STRATA
        }
        self._next_item_id = 0
        self._rng = random.Random(seed)

    @staticmethod
    def _resolve_capacities(
        capacity: int, configured: Mapping[str, int] | None
    ) -> dict[ProgressStratum, int]:
        if configured is None:
            base, remainder = divmod(capacity, len(PROGRESS_STRATA))
            return {
                stratum: base + int(index < remainder)
                for index, stratum in enumerate(PROGRESS_STRATA)
            }
        if set(configured) != set(PROGRESS_STRATA):
            raise ValueError("stratum_capacities must specify every stratum")
        capacities: dict[ProgressStratum, int] = {}
        for stratum in PROGRESS_STRATA:
            value = configured[stratum]
            if type(value) is not int or value < 1:
                raise ValueError("every stratum capacity must be a positive int")
            capacities[stratum] = value
        if sum(capacities.values()) != capacity:
            raise ValueError("stratum capacities must sum to total capacity")
        return capacities

    @property
    def stratum_capacities(self) -> dict[ProgressStratum, int]:
        return dict(self._capacities)

    @property
    def stratum_sizes(self) -> dict[ProgressStratum, int]:
        return {key: len(value) for key, value in self._buckets.items()}

    @property
    def seen_counts(self) -> dict[ProgressStratum, int]:
        return dict(self._seen)

    @property
    def event_seen_counts(self) -> dict[ProgressStratum, int]:
        return dict(self._event_seen)

    @property
    def non_event_seen_counts(self) -> dict[ProgressStratum, int]:
        return dict(self._non_event_seen)

    @property
    def event_size(self) -> int:
        return sum(
            item.metadata.contains_food_event
            for bucket in self._buckets.values()
            for item in bucket
        )

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    @staticmethod
    def _all_rows_have(
        transitions: Sequence[Mapping[str, Any]], fields: tuple[str, ...]
    ) -> bool:
        return all(
            any(field in row and row[field] is not None for field in fields)
            for row in transitions
        )

    def append_autonomous(
        self,
        transitions: Sequence[Mapping[str, Any]],
        *,
        score_end: int,
        won: bool,
        contains_food_event: bool,
        policy_version: int,
        item_kind: ReplayItemKind,
    ) -> ReplayInsertResult:
        """Offer one autonomous episode/fragment to its protected stratum."""
        if item_kind not in ("episode", "fragment"):
            raise ValueError("item_kind must be episode or fragment")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a non-negative int")
        if type(contains_food_event) is not bool:
            raise TypeError("contains_food_event must be a bool")
        if not isinstance(transitions, Sequence) or not transitions:
            raise ValueError("transitions must be a non-empty sequence")
        if not all(isinstance(row, Mapping) for row in transitions):
            raise TypeError("every transition must be a mapping")
        for row in transitions:
            row_version = row.get("policy_version")
            if row_version is not None and row_version != policy_version:
                raise ValueError("transition policy_version mismatch")

        stratum = progress_stratum(score_end, won=won)
        item_id = self._next_item_id
        self._next_item_id += 1
        metadata = ProgressReplayMetadata(
            item_id=item_id,
            data_origin="autonomous_rl",
            item_kind=item_kind,
            score_end=score_end,
            won=won,
            contains_food_event=contains_food_event,
            stratum=stratum,
            policy_version=policy_version,
            transition_count=len(transitions),
            behavior_logp_present=self._all_rows_have(transitions, ("behavior_logp",)),
            timestamps_present=self._all_rows_have(transitions, TIMESTAMP_FIELDS),
            decision_duration_present=self._all_rows_have(
                transitions, DECISION_DURATION_FIELDS
            ),
        )
        copied = tuple(copy.deepcopy(dict(row)) for row in transitions)
        item = ProgressReplayItem(metadata=metadata, transitions=copied)

        bucket = self._buckets[stratum]
        quota = self._capacities[stratum]
        self._seen[stratum] += 1
        if contains_food_event:
            self._event_seen[stratum] += 1
        else:
            self._non_event_seen[stratum] += 1
        replaced_item_id = None
        stored = False
        event_indices = [
            index
            for index, existing in enumerate(bucket)
            if existing.metadata.contains_food_event
        ]
        non_event_indices = [
            index
            for index, existing in enumerate(bucket)
            if not existing.metadata.contains_food_event
        ]
        if contains_food_event and len(event_indices) < quota:
            if len(bucket) < quota:
                bucket.append(item)
            else:
                replacement = (
                    min(
                        non_event_indices,
                        key=lambda index: bucket[index].metadata.item_id,
                    )
                    if self.retention == "fifo"
                    else self._rng.choice(non_event_indices)
                )
                replaced_item_id = bucket[replacement].metadata.item_id
                bucket[replacement] = item
            stored = True
        elif contains_food_event and self.retention == "fifo":
            replacement = min(
                event_indices,
                key=lambda index: bucket[index].metadata.item_id,
            )
            replaced_item_id = bucket[replacement].metadata.item_id
            bucket[replacement] = item
            stored = True
        elif contains_food_event:
            slot = self._rng.randrange(self._event_seen[stratum])
            if slot < quota:
                replacement = event_indices[slot]
                replaced_item_id = bucket[replacement].metadata.item_id
                bucket[replacement] = item
                stored = True
        else:
            available = quota - len(event_indices)
            if len(non_event_indices) < available:
                bucket.append(item)
                stored = True
            elif available > 0 and self.retention == "fifo":
                replacement = min(
                    non_event_indices,
                    key=lambda index: bucket[index].metadata.item_id,
                )
                replaced_item_id = bucket[replacement].metadata.item_id
                bucket[replacement] = item
                stored = True
            elif available > 0:
                slot = self._rng.randrange(self._non_event_seen[stratum])
                if slot < available:
                    replacement = non_event_indices[slot]
                    replaced_item_id = bucket[replacement].metadata.item_id
                    bucket[replacement] = item
                    stored = True
        return ReplayInsertResult(
            item_id=item_id,
            stratum=stratum,
            stored=stored,
            replaced_item_id=replaced_item_id,
        )

    def items(
        self, stratum: ProgressStratum | None = None
    ) -> tuple[ProgressReplayItem, ...]:
        """Return defensive copies in deterministic stratum/bucket order."""
        if stratum is not None and stratum not in PROGRESS_STRATA:
            raise ValueError("unknown progress stratum")
        selected = (
            self._buckets[stratum]
            if stratum is not None
            else [item for name in PROGRESS_STRATA for item in self._buckets[name]]
        )
        return tuple(copy.deepcopy(item) for item in selected)

    def state_dict(self) -> dict[str, Any]:
        """Return plain checkpoint state, including reservoir RNG position."""
        return {
            "format": "autonomous-progress-replay-v2",
            "capacity": self.capacity,
            "retention": self.retention,
            "stratum_capacities": dict(self._capacities),
            "seen": dict(self._seen),
            "event_seen": dict(self._event_seen),
            "non_event_seen": dict(self._non_event_seen),
            "next_item_id": self._next_item_id,
            "rng_state": self._rng.getstate(),
            "buckets": {
                stratum: [
                    {
                        "metadata": asdict(item.metadata),
                        "transitions": copy.deepcopy(item.transitions),
                    }
                    for item in self._buckets[stratum]
                ]
                for stratum in PROGRESS_STRATA
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a store only when capacity and retention semantics match."""
        if state.get("format") != "autonomous-progress-replay-v2":
            raise ValueError("progress replay checkpoint format mismatch")
        if state.get("capacity") != self.capacity:
            raise ValueError("progress replay checkpoint capacity mismatch")
        if state.get("retention") != self.retention:
            raise ValueError("progress replay checkpoint retention mismatch")
        if state.get("stratum_capacities") != self._capacities:
            raise ValueError("progress replay stratum capacity mismatch")
        raw_buckets = state.get("buckets")
        raw_seen = state.get("seen")
        raw_event_seen = state.get("event_seen")
        raw_non_event_seen = state.get("non_event_seen")
        if not isinstance(raw_buckets, Mapping) or set(raw_buckets) != set(
            PROGRESS_STRATA
        ):
            raise ValueError("progress replay checkpoint strata mismatch")
        if not isinstance(raw_seen, Mapping) or set(raw_seen) != set(PROGRESS_STRATA):
            raise ValueError("progress replay checkpoint seen counts mismatch")
        if not isinstance(raw_event_seen, Mapping) or set(raw_event_seen) != set(
            PROGRESS_STRATA
        ):
            raise ValueError("progress replay event seen counts mismatch")
        valid_non_event_seen = isinstance(raw_non_event_seen, Mapping) and set(
            raw_non_event_seen
        ) == set(PROGRESS_STRATA)
        if not valid_non_event_seen:
            raise ValueError("progress replay non-event seen counts mismatch")
        buckets: dict[ProgressStratum, list[ProgressReplayItem]] = {}
        item_ids = set()
        for stratum in PROGRESS_STRATA:
            raw_items = raw_buckets[stratum]
            if (
                not isinstance(raw_items, Sequence)
                or len(raw_items) > self._capacities[stratum]
            ):
                raise ValueError("progress replay bucket exceeds its capacity")
            bucket = []
            for raw_item in raw_items:
                metadata = ProgressReplayMetadata(**raw_item["metadata"])
                if metadata.stratum != stratum or metadata.item_id in item_ids:
                    raise ValueError("progress replay item identity mismatch")
                if type(metadata.contains_food_event) is not bool:
                    raise ValueError("progress replay event marker is invalid")
                if metadata.stratum != progress_stratum(
                    metadata.score_end, won=metadata.won
                ):
                    raise ValueError("progress replay score stratum mismatch")
                transitions = tuple(
                    copy.deepcopy(dict(row)) for row in raw_item["transitions"]
                )
                if metadata.transition_count != len(transitions):
                    raise ValueError("progress replay transition count mismatch")
                item_ids.add(metadata.item_id)
                bucket.append(
                    ProgressReplayItem(metadata=metadata, transitions=transitions)
                )
            seen = raw_seen[stratum]
            if type(seen) is not int or seen < len(bucket):
                raise ValueError("progress replay seen count is invalid")
            event_seen = raw_event_seen[stratum]
            non_event_seen = raw_non_event_seen[stratum]
            retained_events = sum(item.metadata.contains_food_event for item in bucket)
            retained_non_events = len(bucket) - retained_events
            expected_events = min(event_seen, self._capacities[stratum])
            expected_non_events = min(
                non_event_seen,
                self._capacities[stratum] - expected_events,
            )
            if (
                type(event_seen) is not int
                or type(non_event_seen) is not int
                or event_seen < retained_events
                or non_event_seen < retained_non_events
                or event_seen + non_event_seen != seen
                or retained_events != expected_events
                or retained_non_events != expected_non_events
            ):
                raise ValueError("progress replay event seen count is invalid")
            buckets[stratum] = bucket
        next_item_id = state.get("next_item_id")
        if type(next_item_id) is not int or next_item_id < 0:
            raise ValueError("progress replay next item id is invalid")
        if item_ids and next_item_id <= max(item_ids):
            raise ValueError("progress replay next item id is stale")
        rng = random.Random()
        try:
            rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("progress replay RNG state is invalid") from exc
        self._buckets = buckets
        self._seen = {stratum: int(raw_seen[stratum]) for stratum in PROGRESS_STRATA}
        self._event_seen = {
            stratum: int(raw_event_seen[stratum]) for stratum in PROGRESS_STRATA
        }
        self._non_event_seen = {
            stratum: int(raw_non_event_seen[stratum]) for stratum in PROGRESS_STRATA
        }
        self._next_item_id = next_item_id
        self._rng = rng

    @staticmethod
    def _validate_fraction(name: str, value: float | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number or None")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be within [0, 1]")

    @staticmethod
    def _metadata_eligible(
        metadata: ProgressReplayMetadata,
        *,
        require_behavior_logp: bool,
        require_timestamps: bool,
        require_decision_duration: bool,
    ) -> bool:
        return not (
            (require_behavior_logp and not metadata.behavior_logp_present)
            or (require_timestamps and not metadata.timestamps_present)
            or (require_decision_duration and not metadata.decision_duration_present)
        )

    @staticmethod
    def _requested_groups(
        items: list[ProgressReplayItem],
        event_fraction: float | None,
        high_progress_fraction: float | None,
    ) -> list[tuple[str, float, list[ProgressReplayItem]]]:
        if event_fraction is None and high_progress_fraction is None:
            return [("all", 1.0, items)]
        if event_fraction is not None and high_progress_fraction is None:
            return [
                (
                    "non_event",
                    1 - event_fraction,
                    [item for item in items if not item.metadata.event_bearing],
                ),
                (
                    "event",
                    event_fraction,
                    [item for item in items if item.metadata.event_bearing],
                ),
            ]
        if event_fraction is None:
            assert high_progress_fraction is not None
            return [
                (
                    "not_high_event",
                    1 - high_progress_fraction,
                    [
                        item
                        for item in items
                        if not (
                            item.metadata.event_bearing and item.metadata.high_progress
                        )
                    ],
                ),
                (
                    "high_event",
                    high_progress_fraction,
                    [
                        item
                        for item in items
                        if item.metadata.event_bearing and item.metadata.high_progress
                    ],
                ),
            ]
        assert high_progress_fraction is not None
        return [
            (
                "non_event",
                1 - event_fraction,
                [item for item in items if not item.metadata.event_bearing],
            ),
            (
                "event_low",
                event_fraction - high_progress_fraction,
                [
                    item
                    for item in items
                    if item.metadata.event_bearing and not item.metadata.high_progress
                ],
            ),
            (
                "high_event",
                high_progress_fraction,
                [
                    item
                    for item in items
                    if item.metadata.event_bearing and item.metadata.high_progress
                ],
            ),
        ]

    def sample(
        self,
        batch_size: int,
        *,
        event_fraction: float | None = None,
        high_progress_fraction: float | None = None,
        strict_group_fractions: bool = False,
        require_behavior_logp: bool = True,
        require_timestamps: bool = True,
        require_decision_duration: bool = True,
    ) -> ProgressReplayBatch:
        """Draw with replacement and return exact proposal/IS information.

        Empty positive-mass groups raise when ``strict_group_fractions`` is
        true.  Otherwise, their mass is removed and the remaining non-empty
        positive-mass groups are renormalised.  Items assigned zero requested
        mass are outside the proposal support; the uniform target and IS weight
        are consequently defined over the actual support, not all stored data.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        self._validate_fraction("event_fraction", event_fraction)
        self._validate_fraction("high_progress_fraction", high_progress_fraction)
        if (
            event_fraction is not None
            and high_progress_fraction is not None
            and high_progress_fraction > event_fraction
        ):
            raise ValueError("high_progress_fraction cannot exceed event_fraction")

        stored = [
            item for stratum in PROGRESS_STRATA for item in self._buckets[stratum]
        ]
        eligible = [
            item
            for item in stored
            if self._metadata_eligible(
                item.metadata,
                require_behavior_logp=require_behavior_logp,
                require_timestamps=require_timestamps,
                require_decision_duration=require_decision_duration,
            )
        ]
        if not eligible:
            raise ValueError("no replay items satisfy the metadata requirements")

        requested = self._requested_groups(
            eligible, event_fraction, high_progress_fraction
        )
        missing = [name for name, mass, group in requested if mass > 0 and not group]
        if strict_group_fractions and missing:
            raise ValueError("requested replay groups are empty: " + ", ".join(missing))
        supported = [
            (name, mass, group) for name, mass, group in requested if mass > 0 and group
        ]
        total_mass = sum(mass for _, mass, _ in supported)
        if total_mass <= 0:
            raise ValueError("requested fractions have no non-empty proposal support")
        groups = [(name, mass / total_mass, group) for name, mass, group in supported]
        support_size = sum(len(group) for _, _, group in groups)
        uniform_probability = 1.0 / support_size

        samples = []
        for _ in range(batch_size):
            draw = self._rng.random()
            cumulative = 0.0
            selected_name, selected_mass, selected_items = groups[-1]
            for name, mass, group_items in groups:
                cumulative += mass
                if draw < cumulative:
                    selected_name, selected_mass, selected_items = (
                        name,
                        mass,
                        group_items,
                    )
                    break
            item = self._rng.choice(selected_items)
            proposal_probability = selected_mass / len(selected_items)
            inclusion_probability = None
            if self.retention == "reservoir":
                stratum = item.metadata.stratum
                if item.metadata.contains_food_event:
                    eligible_capacity = self._capacities[stratum]
                    seen = self._event_seen[stratum]
                else:
                    eligible_capacity = max(
                        0,
                        self._capacities[stratum] - self._event_seen[stratum],
                    )
                    seen = self._non_event_seen[stratum]
                inclusion_probability = min(1.0, eligible_capacity / seen)
            samples.append(
                ProgressReplaySample(
                    item=copy.deepcopy(item),
                    proposal_group=selected_name,
                    proposal_probability=proposal_probability,
                    uniform_target_probability=uniform_probability,
                    importance_weight=(uniform_probability / proposal_probability),
                    reservoir_inclusion_probability=inclusion_probability,
                )
            )
        return ProgressReplayBatch(
            samples=tuple(samples),
            eligible_count=len(eligible),
            support_size=support_size,
            requested_event_fraction=(
                float(event_fraction) if event_fraction is not None else None
            ),
            requested_high_progress_fraction=(
                float(high_progress_fraction)
                if high_progress_fraction is not None
                else None
            ),
            effective_group_masses=tuple((name, mass) for name, mass, _ in groups),
        )


__all__ = [
    "AutonomousProgressReplay",
    "DECISION_DURATION_FIELDS",
    "HIGH_PROGRESS_STRATA",
    "PROGRESS_STRATA",
    "ProgressReplayBatch",
    "ProgressReplayItem",
    "ProgressReplayMetadata",
    "ProgressReplaySample",
    "ReplayInsertResult",
    "TIMESTAMP_FIELDS",
    "progress_stratum",
]

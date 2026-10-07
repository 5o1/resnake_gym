"""Experimental isolated-fragment replay.

This replay is intentionally excluded from the supported event-trace training
API because sampling isolated fragments truncates event-scale credit assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from resnake_gym.experimental.progress_replay import (
    AutonomousProgressReplay,
    progress_stratum,
)
from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.gamepad_vtrace_fragments import validate_fragment


@dataclass(frozen=True)
class ExperimentalReplayConfig:
    """Settings for the unsupported isolated-fragment replay experiment."""

    capacity: int = 128
    retention: str = "reservoir"
    batch_fragments: int = 16
    event_fraction: float = 0.5
    high_progress_fraction: float = 0.1
    is_beta: float = 0.4
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.capacity < 1 or self.batch_fragments < 1:
            raise ValueError("experimental replay counts must be positive")
        if self.retention not in ("reservoir", "fifo"):
            raise ValueError("retention must be reservoir or fifo")
        if not (0 <= self.high_progress_fraction <= self.event_fraction <= 1):
            raise ValueError("invalid replay event/progress fractions")
        if not 0 <= self.is_beta <= 1:
            raise ValueError("is_beta must be in [0, 1]")


class ProgressFragmentReplay:
    """Keep full fragments beside a lightweight stratified sampling index.

    The generic progress store deep-copies only audit rows.  Large observation
    tensors stay in ``_payloads`` and are removed as soon as their index entry
    is rejected or replaced.  Thus RAM remains bounded by the store's fixed
    per-stratum quotas; empty strata intentionally do not lend their quota to
    score-zero traffic.
    """

    def __init__(
        self,
        config: VTraceConfig,
        replay_config: ExperimentalReplayConfig | None = None,
    ):
        self.config = config
        if replay_config is None:
            replay_config = ExperimentalReplayConfig(
                **config._experimental_replay_config
            )
        self.replay_config = replay_config
        self.index = AutonomousProgressReplay(
            replay_config.capacity,
            retention=replay_config.retention,
            seed=replay_config.seed + 70_001,
        )
        self._payloads: dict[int, dict[str, Any]] = {}
        self._payload_nbytes: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.index)

    @property
    def stratum_sizes(self) -> dict[str, int]:
        return self.index.stratum_sizes

    @property
    def event_size(self) -> int:
        return self.index.event_size

    @property
    def payload_bytes(self) -> int:
        return sum(self._payload_nbytes.values())

    @staticmethod
    def _fragment_nbytes(fragment: dict[str, Any]) -> int:
        arrays = [value for value in fragment.values() if isinstance(value, np.ndarray)]
        arrays.extend(fragment["observations"].values())
        arrays.extend(fragment["burn_observations"].values())
        arrays.extend(fragment["bootstrap_observation"].values())
        return sum(value.nbytes for value in arrays)

    def append(self, fragment: dict[str, Any]):
        validate_fragment(fragment, self.config)
        versions = np.unique(fragment["policy_versions"])
        if len(versions) != 1:
            raise ValueError("replay fragments must contain one behavior version")
        rows = [
            {
                "behavior_logp": float(fragment["behavior_log_probs"][index]),
                "controller_tick": int(fragment["controller_ticks"][index]),
                "capture_tick": int(fragment["capture_ticks"][index]),
                "execution_ticks": list(fragment["execution_ticks"][index]),
                "ticks_advanced": int(fragment["ticks_advanced"][index]),
                "policy_version": int(versions[0]),
            }
            for index in range(fragment["length"])
        ]
        result = self.index.append_autonomous(
            rows,
            score_end=int(fragment["score_end"]),
            won=bool(fragment["won"]),
            contains_food_event=bool(fragment["contains_food_event"]),
            policy_version=int(versions[0]),
            item_kind="fragment",
        )
        if result.replaced_item_id is not None:
            self._payloads.pop(result.replaced_item_id, None)
            self._payload_nbytes.pop(result.replaced_item_id, None)
        if result.stored:
            self._payloads[result.item_id] = fragment
            self._payload_nbytes[result.item_id] = self._fragment_nbytes(fragment)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "gamepad-vtrace-progress-fragments-v2",
            "index": self.index.state_dict(),
            "payloads": self._payloads,
            "payload_nbytes": dict(self._payload_nbytes),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != "gamepad-vtrace-progress-fragments-v2":
            raise ValueError("fragment replay checkpoint format mismatch")
        index = AutonomousProgressReplay(
            self.replay_config.capacity,
            retention=self.replay_config.retention,
            seed=self.replay_config.seed + 70_001,
        )
        index.load_state_dict(state["index"])
        payloads = state.get("payloads")
        if not isinstance(payloads, dict):
            raise ValueError("fragment replay payload table is invalid")
        metadata_by_id = {
            item.metadata.item_id: item.metadata for item in index.items()
        }
        if set(payloads) != set(metadata_by_id):
            raise ValueError("fragment replay payload/index identities differ")
        checked_payloads = {}
        checked_sizes = {}
        for item_id, fragment in payloads.items():
            validate_fragment(fragment, self.config)
            metadata = metadata_by_id[item_id]
            score = int(fragment["score_end"])
            won = bool(fragment["won"])
            policy_version = int(np.unique(fragment["policy_versions"])[0])
            if (
                metadata.item_kind != "fragment"
                or metadata.score_end != score
                or metadata.won != won
                or metadata.contains_food_event != bool(fragment["contains_food_event"])
                or metadata.stratum != progress_stratum(score, won=won)
                or metadata.policy_version != policy_version
                or metadata.transition_count != fragment["length"]
            ):
                raise ValueError("fragment replay payload/index metadata mismatch")
            checked_payloads[item_id] = fragment
            checked_sizes[item_id] = self._fragment_nbytes(fragment)
        stored_sizes = state.get("payload_nbytes", {})
        if stored_sizes and stored_sizes != checked_sizes:
            raise ValueError("fragment replay payload byte audit mismatch")
        self.index = index
        self._payloads = checked_payloads
        self._payload_nbytes = checked_sizes

    def sample(self) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
        batch = self.index.sample(
            self.replay_config.batch_fragments,
            event_fraction=self.replay_config.event_fraction,
            high_progress_fraction=self.replay_config.high_progress_fraction,
        )
        fragments = []
        weights = []
        groups: dict[str, int] = {}
        for sample in batch.samples:
            item_id = sample.item.metadata.item_id
            try:
                fragments.append(self._payloads[item_id])
            except KeyError as exc:
                raise RuntimeError("replay payload/index mismatch") from exc
            weights.append(sample.importance_weight**self.replay_config.is_beta)
            groups[sample.proposal_group] = groups.get(sample.proposal_group, 0) + 1
        return (
            fragments,
            np.asarray(weights, dtype=np.float32),
            {
                "replay_size": len(self),
                "replay_event_size": self.event_size,
                "replay_payload_bytes": self.payload_bytes,
                "replay_stratum_sizes": self.stratum_sizes,
                "replay_sample_groups": groups,
                "replay_support_size": batch.support_size,
                "replay_eligible_count": batch.eligible_count,
                "replay_effective_group_masses": dict(batch.effective_group_masses),
                "replay_is_beta": self.replay_config.is_beta,
                "replay_is_weight_mean": float(np.mean(weights)),
                "replay_is_weight_max": float(np.max(weights)),
            },
        )

"""Sequence SIL from autonomous experience, based on Oh et al., ICML 2018.

Project adaptation: protected autonomous-success episodes, a stratified replay
target, sequence priorities with within-stratum importance correction,
full-prefix GRU reconstruction and variable-duration returns. No demonstrations.
"""

import copy
from pathlib import Path

import torch

from resnake_gym.models import distribution_from_observation, policy_action_log_prob


def _row_policy_action(row):
    """Read v4 actions and the buttons/axes schema used by v3 checkpoints."""
    if "policy_action" in row:
        return row["policy_action"]
    return row["buttons"], row["axes"]


def _batch_policy_actions(rows, device):
    actions = [_row_policy_action(row) for row in rows]
    if isinstance(actions[0], tuple):
        if not all(isinstance(action, tuple) for action in actions):
            raise ValueError("replay batch mixes incompatible action encodings")
        return tuple(
            torch.cat([action[index] for action in actions]).to(device)
            for index in range(len(actions[0]))
        )
    if any(isinstance(action, tuple) for action in actions):
        raise ValueError("replay batch mixes incompatible action encodings")
    return torch.cat(actions).to(device)


class SequenceReplay:
    def __init__(self, num_envs, capacity=256, archive=None, success_fraction=0.5):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        if not 0.5 <= success_fraction < 1:
            raise ValueError("success fraction must be in [0.5, 1)")
        self.num_envs = num_envs
        self.pending = [[] for _ in range(num_envs)]
        self.success_snapshots = {}
        # Keep a plain list for checkpoint compatibility.  The eviction and
        # sampling rules below derive strata from episode contents, so older
        # checkpoints containing only ``episodes`` remain valid.
        self.episodes = []
        self.capacity = capacity
        self.success_fraction = success_fraction
        self.archive = Path(archive) if archive else None
        self.count = 0
        self.last_sampling_metrics = self._sampling_metrics()
        if self.archive:
            self.archive.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def is_success_episode(episode):
        """Food or a strict win marks autonomous success; no teacher is used."""
        if not episode:
            return False
        stored = episode[0].get("sil_success")
        if stored is not None:
            return bool(stored)
        return any(
            bool(row.get("food_events"))
            or bool(row.get("won", False))
            or float(row.get("score", 0)) > 0
            for row in episode
        )

    def _replay_items(self):
        return self.episodes + [
            self.success_snapshots[index] for index in sorted(self.success_snapshots)
        ]

    def _success_mask(self, episodes=None):
        episodes = self._replay_items() if episodes is None else episodes
        return torch.tensor(
            [self.is_success_episode(episode) for episode in episodes],
            dtype=torch.bool,
        )

    def _snapshot_mask(self):
        """Mask live per-environment snapshots, not retained snapshot fragments."""
        return torch.tensor(
            [False] * len(self.episodes) + [True] * len(self.success_snapshots),
            dtype=torch.bool,
        )

    def _retained_fragment_mask(self):
        return torch.tensor(
            [
                bool(episode[0].get("sil_retained_success_fragment", False))
                for episode in self.episodes
            ]
            + [False] * len(self.success_snapshots),
            dtype=torch.bool,
        )

    def _sampling_metrics(
        self, sampled_success=None, sampled_snapshot=None, sampled_retained=None
    ):
        completed = [
            episode
            for episode in self.episodes
            if not episode[0].get("sil_retained_success_fragment", False)
        ]
        completed_success_count = sum(
            self.is_success_episode(episode) for episode in completed
        )
        retained_count = len(self.episodes) - len(completed)
        snapshot_count = len(self.success_snapshots)
        success = self._success_mask()
        success_count = int(success.sum())
        failure_count = len(completed) - completed_success_count
        if success_count and failure_count:
            target_success = self.success_fraction
        else:
            target_success = float(bool(success_count))
        sampled_success_count = (
            int(sampled_success.sum()) if sampled_success is not None else 0
        )
        sampled_count = len(sampled_success) if sampled_success is not None else 0
        sampled_snapshot_count = (
            int(sampled_snapshot.sum()) if sampled_snapshot is not None else 0
        )
        sampled_retained_count = (
            int(sampled_retained.sum()) if sampled_retained is not None else 0
        )
        return {
            "sil_replay_success_sequences": success_count,
            "sil_replay_completed_success_episodes": completed_success_count,
            "sil_replay_retained_success_fragments": retained_count,
            "sil_replay_active_success_snapshots": snapshot_count,
            "sil_replay_failure_episodes": failure_count,
            "sil_replay_sampleable_sequences": len(self.episodes) + snapshot_count,
            "sil_target_success_fraction": target_success,
            "sil_sampled_success_sequences": sampled_success_count,
            "sil_sampled_active_success_snapshots": sampled_snapshot_count,
            "sil_sampled_retained_success_fragments": sampled_retained_count,
            "sil_sampled_failure_episodes": sampled_count - sampled_success_count,
            "sil_sampled_success_fraction": (
                sampled_success_count / sampled_count if sampled_count else 0.0
            ),
        }

    def _validate_env_index(self, index):
        if type(index) is not int or not 0 <= index < self.num_envs:
            raise IndexError("replay environment index out of range")

    def _snapshot_path(self, index, *, temporary=False):
        self._validate_env_index(index)
        if self.archive is None:
            return None
        suffix = "tmp" if temporary else "pt"
        return self.archive / f"pending-success-env-{index:04d}.{suffix}"

    @staticmethod
    def _set_returns(episode, bootstrap):
        value = bootstrap
        for row in reversed(episode):
            value = row["reward"] + row["discount"] * value
            row["return"] = value

    def _archive_snapshot(self, index):
        if self.archive is None:
            return
        target = self._snapshot_path(index)
        temporary = self._snapshot_path(index, temporary=True)
        torch.save(self.success_snapshots[index], temporary)
        temporary.replace(target)

    def _drop_snapshot(self, index):
        self.success_snapshots.pop(index, None)
        if self.archive is not None:
            self._snapshot_path(index).unlink(missing_ok=True)
            self._snapshot_path(index, temporary=True).unlink(missing_ok=True)

    def _replace_success_snapshot(self, index, bootstrap):
        snapshot = copy.deepcopy(self.pending[index])
        self._set_returns(snapshot, bootstrap)
        snapshot[0].update(
            sil_success=True,
            sil_success_snapshot=True,
            sil_snapshot_env=index,
            sil_snapshot_active=True,
            sil_retained_success_fragment=False,
            sil_snapshot_bootstrap=float(bootstrap),
        )
        # Exactly one snapshot per live environment: a later food atomically
        # replaces the earlier prefix instead of retaining overlapping copies.
        self.success_snapshots[index] = snapshot
        self._archive_snapshot(index)
        self.last_sampling_metrics = self._sampling_metrics()

    def _enforce_capacity(self):
        """Failures may replace failures, but can never evict a success."""
        while len(self.episodes) > self.capacity:
            failure = next(
                (
                    index
                    for index, episode in enumerate(self.episodes)
                    if not self.is_success_episode(episode)
                ),
                None,
            )
            # If every retained episode succeeded, only another success can
            # have overflowed the fixed total capacity; retire the oldest one.
            self.episodes.pop(0 if failure is None else failure)

    def refresh_sampling_metrics(self):
        """Rebuild derived replay state after restoring a checkpoint."""
        self._enforce_capacity()
        self.last_sampling_metrics = self._sampling_metrics()
        return dict(self.last_sampling_metrics)

    def restore_success_snapshots(self, snapshots):
        """Promote old live prefixes to retained fragments after environment reset."""
        self.pending = [[] for _ in range(self.num_envs)]
        for index in range(self.num_envs):
            self._drop_snapshot(index)
        for raw_index, episode in (snapshots or {}).items():
            index = int(raw_index)
            self._validate_env_index(index)
            if not episode:
                raise ValueError("success snapshot cannot be empty")
            snapshot = copy.deepcopy(episode)
            snapshot[0].update(
                sil_success=True,
                sil_success_snapshot=True,
                sil_snapshot_env=index,
                sil_snapshot_active=False,
                sil_retained_success_fragment=True,
            )
            self.episodes.append(snapshot)
            if self.archive is not None:
                target = self.archive / f"restored-success-env-{index:04d}.pt"
                temporary = self.archive / f"restored-success-env-{index:04d}.tmp"
                torch.save(snapshot, temporary)
                temporary.replace(target)
        return self.refresh_sampling_metrics()

    def append(self, index, record, done, bootstrap=0.0):
        self._validate_env_index(index)
        self.pending[index].append(record)
        if not done:
            if record.get("food_events"):
                self._replace_success_snapshot(index, bootstrap)
            return
        episode = self.pending[index]
        self.pending[index] = []
        self._set_returns(episode, bootstrap)
        episode[0]["sil_success"] = self.is_success_episode(episode)
        episode[0]["sil_success_snapshot"] = False
        episode[0]["sil_snapshot_active"] = False
        episode[0]["sil_retained_success_fragment"] = False
        if self.archive:
            torch.save(episode, self.archive / f"episode-{self.count:08d}.pt")
        self.count += 1
        self.episodes.append(episode)
        self._drop_snapshot(index)
        self.refresh_sampling_metrics()  # RAM eviction; archives remain on disk.

    def sampling_distribution(self, alpha=0.6):
        if not 0 <= alpha <= 1:
            raise ValueError("priority alpha must be in [0, 1]")
        episodes = self._replay_items()
        if not episodes:
            empty = torch.empty(0, dtype=torch.float32)
            return empty, empty
        lengths = torch.tensor([len(e) for e in episodes], dtype=torch.float32)
        priorities = torch.tensor(
            [e[0].get("sil_priority", 1.0) for e in episodes]
        ).clamp_min(1e-6)
        success = self._success_mask(episodes)
        failure = ~success
        target = torch.zeros_like(lengths)
        probabilities = torch.zeros_like(lengths)
        if success.any() and failure.any():
            stratum_masses = (
                (success, self.success_fraction),
                (failure, 1 - self.success_fraction),
            )
        elif success.any():
            stratum_masses = ((success, 1.0),)
        else:
            stratum_masses = ((failure, 1.0),)
        for mask, stratum_mass in stratum_masses:
            conditional_target = lengths[mask] / lengths[mask].sum()
            priority_mass = lengths[mask] * priorities[mask].pow(alpha)
            conditional_probability = (
                0.1 * conditional_target + 0.9 * priority_mass / priority_mass.sum()
            )
            # Importance correction is intentionally only within this target
            # stratum.  It must not undo the explicit success allocation by
            # correcting back to a global uniform-transition distribution.
            target[mask] = stratum_mass * conditional_target
            probabilities[mask] = stratum_mass * conditional_probability
        return probabilities, target / probabilities

    def update(
        self, model, optimizer, device, weight=0.1, batch_size=1, priority_alpha=0.0
    ):
        if batch_size < 1:
            raise ValueError("SIL batch size must be positive")
        episodes = self._replay_items()
        if not episodes:
            self.last_sampling_metrics = self._sampling_metrics()
            return {"sil_valid": 0, "sil_loss": 0.0, "sil_rows": 0}
        probabilities, corrections = self.sampling_distribution(priority_alpha)
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        sampled_success = self._success_mask(episodes)[indices]
        sampled_snapshot = self._snapshot_mask()[indices]
        sampled_retained = self._retained_fragment_mask()[indices]
        self.last_sampling_metrics = self._sampling_metrics(
            sampled_success, sampled_snapshot, sampled_retained
        )
        sampled_episodes = [episodes[int(i)] for i in indices]
        lengths = torch.tensor([len(e) for e in sampled_episodes], device=device)
        factors = corrections[indices].to(device) / lengths / batch_size
        hidden, losses, valid, total = None, [], 0, 0.0
        priorities = torch.zeros(batch_size, device=device)
        optimizer.zero_grad()
        for index in range(int(lengths.max())):
            # Repeating the last observation avoids invalid all-zero board padding.
            rows = [e[min(index, len(e) - 1)] for e in sampled_episodes]
            active = index < lengths
            observation = {
                k: torch.cat([row["obs"][k] for row in rows]).to(device)
                for k in rows[0]["obs"]
            }
            distribution, value, hidden = distribution_from_observation(
                model, observation, hidden
            )
            returns = torch.tensor([row["return"] for row in rows], device=device)
            delta = (returns - value[:, 0]).clamp_min(0) * active
            logp = policy_action_log_prob(
                distribution,
                _batch_policy_actions(rows, device),
            )
            losses.append(
                (factors * (-logp * delta.detach() + 0.5 * delta.square())).sum()
            )
            priorities += delta.detach()
            valid += int((delta.detach() > 0).sum())
            if len(losses) == 128 or index + 1 == int(lengths.max()):
                loss = weight * torch.stack(losses).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite SIL loss")
                loss.backward()
                total += float(loss.detach())
                losses = []
                hidden = hidden.detach()
        if valid:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True
            )
            optimizer.step()
        for episode, priority in zip(
            sampled_episodes, (priorities / lengths).tolist(), strict=True
        ):
            episode[0]["sil_priority"] = max(priority, 1e-6)
        return {"sil_valid": valid, "sil_loss": total, "sil_rows": int(lengths.sum())}

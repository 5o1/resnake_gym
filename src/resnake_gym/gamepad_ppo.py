"""Recurrent PPO baseline for mixed gamepad chunks, no teachers or action masks.

References: PPO arXiv:1707.06347; GAE arXiv:1506.02438; CleanRL
ppo_atari_lstm.py for sequence-preserving recurrent updates. Independently
implemented for mixed controls and variable-duration decision intervals.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from resnake_gym.gamepad import direction_request
from resnake_gym.gamepad_ppo_contract import (
    CHECKPOINT_FORMAT,
    COLLECTION_SEMANTICS_VERSION,
    RECURRENT_UPDATE_VERSION,
    REPLAY_OBJECTIVE_VERSION,
    STRUCTURED_CHECKPOINT_FORMAT,
    STRUCTURED_TRAINING_OBJECTIVE_VERSION,
    TRAINING_OBJECTIVE_VERSION,
    PPOConfig,
    checkpoint_semantics,
    validate_policy_checkpoint,
)
from resnake_gym.gamepad_runtime import build_model, make_env, stack_observations
from resnake_gym.models import (
    distribution_from_observation,
    policy_action_entropy,
    policy_action_log_prob,
    reports_from_policy_action,
    sample_policy_action,
)
from resnake_gym.sil import SequenceReplay
from resnake_gym.training.ppo_update import (
    PPOOptimization,
    RolloutBatch,
    prepare_rollout,
    recurrent_windows,
)
from resnake_gym.training.ppo_update import (
    generalized_advantages as generalized_advantages,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "COLLECTION_SEMANTICS_VERSION",
    "GamepadPPO",
    "PPOConfig",
    "RECURRENT_UPDATE_VERSION",
    "REPLAY_OBJECTIVE_VERSION",
    "STRUCTURED_CHECKPOINT_FORMAT",
    "STRUCTURED_TRAINING_OBJECTIVE_VERSION",
    "TRAINING_OBJECTIVE_VERSION",
    "build_model",
    "checkpoint_semantics",
    "make_env",
    "stack_observations",
    "validate_policy_checkpoint",
]


def collection_stop(config, steps, foods, terminals):
    """Stop after a complete vector step; only food satisfies an event quota.

    ``terminals`` remains an argument because it is useful collection audit data
    and older callers already provide it.  It is deliberately *not* part of the
    success count: dying repeatedly must not make an event rollout look useful.
    ``rollout_steps`` is always the hard per-environment collection limit.
    """
    if (
        config.collection_mode == "events"
        and steps >= config.minimum_steps
        and foods >= config.event_target
    ):
        return "food_target"
    if steps >= config.rollout_steps:
        return "hard_limit" if config.collection_mode == "events" else "fixed_length"
    return None


def _map_policy_action(action, function):
    if isinstance(action, tuple):
        return tuple(function(value) for value in action)
    return function(action)


def _slice_policy_action(action, index):
    return _map_policy_action(action, lambda value: value[index : index + 1])


_recurrent_windows = recurrent_windows


def _slice_observation(observation, env):
    return {key: value[env : env + 1] for key, value in observation.items()}


@dataclass(frozen=True, slots=True)
class _PolicySample:
    """Policy outputs retained across one vector-environment transition."""

    observation: dict[str, torch.Tensor]
    hidden_before: torch.Tensor
    action: Any
    log_probability: torch.Tensor
    value: torch.Tensor
    hidden_after: torch.Tensor
    reports: np.ndarray


@dataclass(frozen=True, slots=True)
class _EnvironmentTransition:
    """Batched environment outputs and their bootstrap values."""

    items: list[tuple]
    next_observations: list[dict[str, Any]]
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    discounts: torch.Tensor
    next_value: torch.Tensor


class GamepadPPO:
    def __init__(self, config, device="cpu", archive=None):
        if min(config.num_envs, config.rollout_steps, config.epochs, config.dim) < 1:
            raise ValueError("PPO counts must be positive")
        if config.recurrent_burn_in < 0 or config.recurrent_unroll < 1:
            raise ValueError("invalid recurrent window lengths")
        if not 0.5 <= config.replay_success_fraction < 1:
            raise ValueError("replay success fraction must be in [0.5, 1)")
        checkpoint_semantics(config.action_head)
        self.config = config
        if config.collection_mode not in ("fixed", "events"):
            raise ValueError("unknown collection mode")
        if config.collection_mode == "events" and not (
            1 <= config.minimum_steps <= config.rollout_steps
            and config.event_target > 0
        ):
            raise ValueError("invalid event collection bounds")
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.model = build_model(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.envs = [make_env(config) for _ in range(config.num_envs)]
        self.observations = [
            env.reset(seed=config.seed + i)[0] for i, env in enumerate(self.envs)
        ]
        self.hidden = torch.zeros(config.num_envs, config.dim, device=self.device)
        self.reset_mask = torch.ones(
            config.num_envs, dtype=torch.bool, device=self.device
        )
        self.logic_ticks = 0
        self.decisions = 0
        self.policy_version = 0
        self.episode_ids = [0] * config.num_envs
        self.completed_food_episodes = 0
        self.replay = SequenceReplay(
            config.num_envs,
            capacity=config.replay_capacity,
            archive=archive,
            success_fraction=config.replay_success_fraction,
        )

    def update(self):
        """Collect one rollout and apply the existing PPO and SIL objectives."""
        rollout = self._collect_rollout()
        prepared = prepare_rollout(rollout.data, self.config)
        optimization = self._optimize_ppo(rollout.data, prepared)
        sil_metrics = self._optimize_sil()
        self._refresh_hidden(rollout.initial_hidden, rollout.data)
        self.policy_version += 1
        return self._assemble_metrics(rollout, prepared, optimization, sil_metrics)

    def _collect_rollout(self) -> RolloutBatch:
        rollout = RolloutBatch(initial_hidden=self.hidden.detach().clone())
        starts = [env.tick for env in self.envs]
        for _ in range(self.config.rollout_steps):
            sample = self._sample_policy_step()
            self._audit_sampled_reports(rollout, sample.reports)
            transition = self._step_environments(sample)
            rollout.data.append(self._rollout_row(sample, transition))
            self.hidden = sample.hidden_after
            self.reset_mask = transition.terminated | transition.truncated
            self._archive_transitions(rollout, starts, sample, transition)
            if self._finish_rollout_step(rollout, transition):
                break
        self._record_batch_boundaries(rollout, starts)
        # Food/batch boundaries are audit metadata only. GAE receives the
        # original ordered transitions and only terminal/reset masks.
        return rollout

    def _sample_policy_step(self) -> _PolicySample:
        observation = stack_observations(self.observations, self.device)
        with torch.no_grad():
            self.hidden = self.hidden * (~self.reset_mask)[:, None]
            hidden_before = self.hidden.detach().clone()
            distribution, value, hidden_after = distribution_from_observation(
                self.model, observation, self.hidden
            )
            action = sample_policy_action(distribution, stochastic=True)
            log_probability = policy_action_log_prob(distribution, action)
            reports = (
                reports_from_policy_action(
                    distribution,
                    action,
                    continuous_buttons=self.config.chunk_rho is not None,
                )
                .cpu()
                .numpy()
            )
        return _PolicySample(
            observation=observation,
            hidden_before=hidden_before,
            action=action,
            log_probability=log_probability,
            value=value,
            hidden_after=hidden_after,
            reports=reports,
        )

    @staticmethod
    def _audit_sampled_reports(rollout: RolloutBatch, reports: np.ndarray) -> None:
        for report_chunk in reports:
            for report in report_chunk:
                direction = direction_request(report)
                category = {None: 0, 0: 1, 2: 2, 3: 3, 1: 4}[direction]
                rollout.sampled_direction_counts[category] += 1
                dpad_active = bool(np.any(report[:4] >= 0.5))
                rollout.sampled_dpad_active += int(dpad_active)
                rollout.sampled_conflict_neutral += int(
                    dpad_active and direction is None
                )

    def _step_environments(self, sample: _PolicySample) -> _EnvironmentTransition:
        items = [
            env.step(report)
            for env, report in zip(self.envs, sample.reports, strict=True)
        ]
        next_observations = [item[0] for item in items]
        rewards = torch.tensor([item[1] for item in items], device=self.device)
        terminated = torch.tensor([item[2] for item in items], device=self.device)
        truncated = torch.tensor([item[3] for item in items], device=self.device)
        discounts = torch.tensor(
            [item[4]["bootstrap_discount"] for item in items],
            device=self.device,
        )
        with torch.no_grad():
            _, next_value, _ = distribution_from_observation(
                self.model,
                stack_observations(next_observations, self.device),
                sample.hidden_after,
            )
        return _EnvironmentTransition(
            items=items,
            next_observations=next_observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            discounts=discounts,
            next_value=next_value,
        )

    def _rollout_row(
        self,
        sample: _PolicySample,
        transition: _EnvironmentTransition,
    ) -> dict[str, Any]:
        return {
            "obs": sample.observation,
            "reset": self.reset_mask.clone(),
            "h_before": sample.hidden_before,
            "policy_action": sample.action,
            "logp": sample.log_probability,
            "value": sample.value[:, 0],
            "next_value": transition.next_value[:, 0],
            "reward": transition.rewards,
            "discount": transition.discounts,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
        }

    def _archive_transitions(
        self,
        rollout: RolloutBatch,
        starts: list[int],
        sample: _PolicySample,
        transition: _EnvironmentTransition,
    ) -> None:
        for env_index, (_, _, term, trunc, info) in enumerate(transition.items):
            self._archive_replay_row(env_index, term, trunc, info, sample, transition)
            self._record_transition_events(rollout, starts, env_index, info)
            self._record_transition_audit(rollout, info)
            if term or trunc:
                self._finish_episode(
                    rollout,
                    starts,
                    env_index,
                    term,
                    trunc,
                    info,
                    transition.next_observations,
                )

    def _archive_replay_row(
        self,
        env_index: int,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        sample: _PolicySample,
        transition: _EnvironmentTransition,
    ) -> None:
        self.replay.append(
            env_index,
            dict(
                obs={
                    key: tensor[env_index : env_index + 1].detach().cpu().clone()
                    for key, tensor in sample.observation.items()
                },
                policy_action=_map_policy_action(
                    _slice_policy_action(sample.action, env_index),
                    lambda tensor: tensor.detach().cpu().clone(),
                ),
                reward=float(transition.rewards[env_index]),
                discount=(
                    0.0 if terminated else float(transition.discounts[env_index])
                ),
                terminated=terminated,
                truncated=truncated,
                behavior_logp=float(sample.log_probability[env_index]),
                policy_version=self.policy_version,
                tick_rewards=info["tick_rewards"],
                execution_ticks=info["execution_ticks"],
                executed_sources=info["executed_sources"].tolist(),
                actual_history=transition.next_observations[env_index][
                    "history"
                ].tolist(),
                food_events=info["food_events"],
                score=info["score"],
                env_index=env_index,
                episode_id=self.episode_ids[env_index],
                termination_reason=info["termination_reason"],
                controller_tick=info["controller_tick"],
                capture_tick=info["capture_tick"],
                command_origin_tick=info["command_origin_tick"],
                command_arrival_tick=info["command_arrival_tick"],
                timebase=info["timebase"],
                reward_version="food-efficiency-pbrs-v1",
            ),
            terminated or truncated,
            0.0 if terminated else float(transition.next_value[env_index, 0]),
        )

    def _record_transition_events(
        self,
        rollout: RolloutBatch,
        starts: list[int],
        env_index: int,
        info: dict[str, Any],
    ) -> None:
        rollout.food_events.extend(info["food_events"])
        for event in info["food_events"]:
            rollout.fragments.append(
                dict(
                    env=env_index,
                    start=starts[env_index],
                    end=event["tick"],
                    boundary="food",
                    episode=self.episode_ids[env_index],
                    policy_version=self.policy_version,
                )
            )
            starts[env_index] = event["tick"]

    def _record_transition_audit(
        self,
        rollout: RolloutBatch,
        info: dict[str, Any],
    ) -> None:
        rollout.no_food_ages.append(info["ticks_since_food"])
        self.logic_ticks += info["ticks_advanced"]
        rollout.rollout_logic_ticks += info["ticks_advanced"]
        rollout.expired_reports += info["expired_reports"]
        rollout.perturbation_events += len(info["perturbation_events"])
        rollout.executed_reports += info["ticks_advanced"]
        rollout.neutral_reports += int((info["executed_sources"] < 0).sum())

    def _finish_episode(
        self,
        rollout: RolloutBatch,
        starts: list[int],
        env_index: int,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        next_observations: list[dict[str, Any]],
    ) -> None:
        rollout.terminals += int(terminated)
        rollout.truncations += int(truncated)
        food_success = bool(info["score"] > 0)
        rollout.completed_food_episodes += int(food_success)
        self.completed_food_episodes += int(food_success)
        rollout.fragments.append(
            dict(
                env=env_index,
                start=starts[env_index],
                end=info["logic_steps"],
                boundary="terminal" if terminated else "time_limit",
                episode=self.episode_ids[env_index],
                policy_version=self.policy_version,
            )
        )
        starts[env_index] = 0
        rollout.episodes.append(
            {
                "score": info["score"],
                "won": info["won"],
                "ticks": info["logic_steps"],
                "termination": info["termination_reason"],
                "env": env_index,
                "episode": self.episode_ids[env_index],
                "food_success": food_success,
            }
        )
        next_observations[env_index] = self.envs[env_index].reset()[0]
        self.episode_ids[env_index] += 1

    def _finish_rollout_step(
        self,
        rollout: RolloutBatch,
        transition: _EnvironmentTransition,
    ) -> bool:
        self.observations = transition.next_observations
        self.decisions += self.config.num_envs
        rollout.stop_reason = collection_stop(
            self.config,
            len(rollout.data),
            len(rollout.food_events),
            rollout.terminals,
        )
        return bool(rollout.stop_reason)

    def _record_batch_boundaries(
        self,
        rollout: RolloutBatch,
        starts: list[int],
    ) -> None:
        for env_index, env in enumerate(self.envs):
            if env.tick > starts[env_index]:
                rollout.fragments.append(
                    dict(
                        env=env_index,
                        start=starts[env_index],
                        end=env.tick,
                        episode=self.episode_ids[env_index],
                        boundary="batch_cutoff",
                        policy_version=self.policy_version,
                    )
                )

    def _optimize_ppo(self, data, prepared) -> PPOOptimization:
        c = self.config
        epochs_done = 0
        burn_in_grad_enabled = False
        learn_grad_enabled = True
        for _ in range(c.epochs):
            self.optimizer.zero_grad()
            policy_total = torch.zeros((), device=self.device)
            value_total = torch.zeros((), device=self.device)
            entropy_total = torch.zeros((), device=self.device)
            kl_total = torch.zeros((), device=self.device)
            for window in prepared.windows:
                env = window["env"]
                burn_start = window["burn_start"]
                learn_start = window["learn_start"]
                learn_end = window["learn_end"]
                hidden = data[burn_start]["h_before"][env : env + 1].detach()
                with torch.no_grad():
                    for step in range(burn_start, learn_start):
                        burn_in_grad_enabled |= torch.is_grad_enabled()
                        row = data[step]
                        hidden = hidden * (~row["reset"][env : env + 1])[:, None]
                        _, _, hidden = distribution_from_observation(
                            self.model,
                            _slice_observation(row["obs"], env),
                            hidden,
                        )
                hidden = hidden.detach()
                logps, values, entropies = [], [], []
                for step in range(learn_start, learn_end):
                    learn_grad_enabled &= torch.is_grad_enabled()
                    row = data[step]
                    hidden = hidden * (~row["reset"][env : env + 1])[:, None]
                    distribution, value, hidden = distribution_from_observation(
                        self.model,
                        _slice_observation(row["obs"], env),
                        hidden,
                    )
                    logps.append(
                        policy_action_log_prob(
                            distribution,
                            _slice_policy_action(row["policy_action"], env),
                        )
                    )
                    values.append(value[:, 0])
                    entropies.append(policy_action_entropy(distribution))
                logps = torch.cat(logps)
                values = torch.cat(values)
                entropies = torch.cat(entropies)
                window_slice = slice(learn_start, learn_end)
                logratio = logps - prepared.old_logp[window_slice, env]
                ratio = logratio.exp()
                window_advantages = prepared.advantages[window_slice, env]
                surrogate = torch.minimum(
                    ratio * window_advantages,
                    ratio.clamp(1 - c.clip, 1 + c.clip) * window_advantages,
                )
                policy_sum = -surrogate.sum()
                value_sum = (
                    0.5 * (values - prepared.targets[window_slice, env]).square().sum()
                )
                entropy_sum = entropies.sum()
                window_loss = (
                    policy_sum + value_sum - c.entropy_coefficient * entropy_sum
                ) / prepared.total_samples
                if not torch.isfinite(window_loss):
                    raise FloatingPointError("nonfinite PPO objective")
                window_loss.backward()
                policy_total += policy_sum.detach()
                value_total += value_sum.detach()
                entropy_total += entropy_sum.detach()
                kl_total += (((ratio - 1) - logratio).sum()).detach()
            epoch_approx_kl = kl_total / prepared.total_samples
            if epochs_done and epoch_approx_kl.item() > c.target_kl:
                self.optimizer.zero_grad()
                approx_kl = epoch_approx_kl
                break
            policy_loss = policy_total / prepared.total_samples
            value_loss = value_total / prepared.total_samples
            latent_entropy = entropy_total / prepared.total_samples
            loss = policy_loss + value_loss - c.entropy_coefficient * latent_entropy
            approx_kl = epoch_approx_kl
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 0.5, error_if_nonfinite=True
            )
            self.optimizer.step()
            epochs_done += 1
        return PPOOptimization(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            latent_entropy=latent_entropy,
            approx_kl=approx_kl,
            grad_norm=grad_norm,
            epochs_done=epochs_done,
            burn_in_grad_enabled=burn_in_grad_enabled,
            learn_grad_enabled=learn_grad_enabled,
        )

    def _optimize_sil(self):
        c = self.config
        sil_metrics = {"sil_valid": 0, "sil_loss": 0.0, "sil_rows": 0}
        sampled_success_sequences = 0
        sampled_active_success_snapshots = 0
        sampled_retained_success = 0
        sampled_failure = 0
        for _ in range(c.sil_updates):
            result = self.replay.update(
                self.model,
                self.optimizer,
                self.device,
                c.sil_weight,
                c.sil_batch_size,
                c.sil_priority_alpha,
            )
            for key, value in result.items():
                sil_metrics[key] += value
            sampled_success_sequences += self.replay.last_sampling_metrics[
                "sil_sampled_success_sequences"
            ]
            sampled_active_success_snapshots += self.replay.last_sampling_metrics[
                "sil_sampled_active_success_snapshots"
            ]
            sampled_retained_success += self.replay.last_sampling_metrics[
                "sil_sampled_retained_success_fragments"
            ]
            sampled_failure += self.replay.last_sampling_metrics[
                "sil_sampled_failure_episodes"
            ]
        sampling_metrics = dict(self.replay.last_sampling_metrics)
        sampling_metrics["sil_sampled_success_sequences"] = sampled_success_sequences
        sampling_metrics["sil_sampled_active_success_snapshots"] = (
            sampled_active_success_snapshots
        )
        sampling_metrics["sil_sampled_retained_success_fragments"] = (
            sampled_retained_success
        )
        sampling_metrics["sil_sampled_failure_episodes"] = sampled_failure
        sampled_total = sampled_success_sequences + sampled_failure
        sampling_metrics["sil_sampled_success_fraction"] = (
            sampled_success_sequences / sampled_total if sampled_total else 0.0
        )
        sil_metrics.update(sampling_metrics)
        return sil_metrics

    def _refresh_hidden(self, initial_hidden, data):
        # Recompute end memory under updated weights. The initial rollout state
        # is fixed (standard truncated-BPTT approximation), not full-episode replay.
        with torch.no_grad():
            hidden = initial_hidden
            for row in data:
                _, _, hidden = distribution_from_observation(
                    self.model, row["obs"], hidden * (~row["reset"])[:, None]
                )
            self.hidden = hidden

    def _assemble_metrics(self, rollout, prepared, optimization, sil_metrics):
        c = self.config
        rollout_steps = len(rollout.data)
        sampled_direction_total = int(rollout.sampled_direction_counts.sum())
        nonzero_direction_probabilities = rollout.sampled_direction_counts[
            rollout.sampled_direction_counts > 0
        ] / max(sampled_direction_total, 1)
        empirical_direction_entropy = float(
            -(
                nonzero_direction_probabilities
                * np.log(nonzero_direction_probabilities)
            ).sum()
        )
        event_collection = c.collection_mode == "events"
        food_target_met = (
            event_collection and len(rollout.food_events) >= c.event_target
        )
        retained_success_fragments = sum(
            bool(episode[0].get("sil_retained_success_fragment", False))
            for episode in self.replay.episodes
        )
        return {
            **sil_metrics,
            "replay_episodes": len(self.replay.episodes) - retained_success_fragments,
            "replay_success_snapshots": len(self.replay.success_snapshots),
            "replay_retained_success_fragments": retained_success_fragments,
            "replay_sampleable_sequences": (
                len(self.replay.episodes) + len(self.replay.success_snapshots)
            ),
            "replay_completed_episodes": self.replay.count,
            "logic_ticks": self.logic_ticks,
            "rollout_logic_ticks": rollout.rollout_logic_ticks,
            "decisions": self.decisions,
            "loss": optimization.loss.item(),
            "policy_loss": optimization.policy_loss.item(),
            "value_loss": optimization.value_loss.item(),
            "latent_entropy": optimization.latent_entropy.item(),
            "approx_kl": optimization.approx_kl.item(),
            "grad_norm": optimization.grad_norm.item(),
            "epochs": optimization.epochs_done,
            "recurrent_window_count": len(prepared.windows),
            "recurrent_samples_per_epoch": prepared.recurrent_samples,
            "recurrent_max_learn_steps": prepared.max_learn_steps,
            "recurrent_max_burn_in_steps": prepared.max_burn_in_steps,
            "recurrent_burn_in_steps_per_epoch": prepared.burn_in_steps,
            "recurrent_windows_crossing_resets": prepared.reset_crossings,
            "recurrent_burn_in_grad_enabled": optimization.burn_in_grad_enabled,
            "recurrent_learn_grad_enabled": optimization.learn_grad_enabled,
            "episodes": rollout.episodes,
            "food_count": len(rollout.food_events),
            "food_events": rollout.food_events,
            "max_ticks_since_food": max(rollout.no_food_ages, default=0),
            "completed_episodes": len(rollout.episodes),
            "completed_food_episodes": rollout.completed_food_episodes,
            "completed_food_episodes_total": self.completed_food_episodes,
            "rollout_decisions_per_env": rollout_steps,
            "rollout_transitions": rollout_steps * c.num_envs,
            "collection_stop": rollout.stop_reason,
            "collection_food_target": c.event_target if event_collection else 0,
            "collection_food_target_met": food_target_met,
            "collection_food_shortfall": (
                max(c.event_target - len(rollout.food_events), 0)
                if event_collection
                else 0
            ),
            "collection_minimum_reached": rollout_steps >= c.minimum_steps,
            "collection_hard_limit_reached": rollout_steps >= c.rollout_steps,
            "terminal_events": rollout.terminals,
            "truncation_events": rollout.truncations,
            "fragments": rollout.fragments,
            "sample_policy_version": self.policy_version - 1,
            "expired_reports": rollout.expired_reports,
            "perturbation_events": rollout.perturbation_events,
            "missing_command_fraction": (
                rollout.neutral_reports / max(rollout.executed_reports, 1)
            ),
            "action_head": c.action_head,
            "sampled_direction_counts": rollout.sampled_direction_counts.tolist(),
            "sampled_direction_labels": [
                "neutral",
                "up",
                "down",
                "left",
                "right",
            ],
            "sampled_effective_direction_entropy": empirical_direction_entropy,
            "sampled_conflict_neutral_fraction": (
                rollout.sampled_conflict_neutral / max(sampled_direction_total, 1)
            ),
            "sampled_dpad_active_fraction": (
                rollout.sampled_dpad_active / max(sampled_direction_total, 1)
            ),
        }

    def checkpoint(self):
        semantics = checkpoint_semantics(self.config.action_head)
        return {
            "reward_version": "food-efficiency-pbrs-v1",
            "format": semantics["format"],
            "scene_encoding": "scene-grid-v2",
            "action_encoding": semantics["action_encoding"],
            "time_encoding": "policy-time-v2",
            "training_objective": semantics["training_objective"],
            "collection_semantics": COLLECTION_SEMANTICS_VERSION,
            "recurrent_update": RECURRENT_UPDATE_VERSION,
            "replay_objective": REPLAY_OBJECTIVE_VERSION,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "logic_ticks": self.logic_ticks,
            "decisions": self.decisions,
            "note": (
                "model, optimizer, completed replay episodes, retained success "
                "fragments and replay-only active snapshots; resets live "
                "games/RNG/pending episodes"
            ),
            "policy_version": self.policy_version,
            "episode_ids": list(self.episode_ids),
            "completed_food_episodes": self.completed_food_episodes,
            "replay_count": self.replay.count,
            "replay_episodes": self.replay.episodes,
            "replay_success_snapshots": self.replay.success_snapshots,
        }

    def restore(self, payload):
        validate_policy_checkpoint(payload)
        restored_head = payload.get("config", {}).get("action_head", "raw")
        if restored_head != self.config.action_head:
            raise ValueError("checkpoint action head mismatch")
        if payload.get("reward_version") != "food-efficiency-pbrs-v1":
            raise ValueError("reward version mismatch")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.logic_ticks = payload["logic_ticks"]
        self.decisions = payload["decisions"]
        self.policy_version = payload.get("policy_version", 0)
        self.replay.episodes = payload.get("replay_episodes", [])
        self.replay.count = payload.get("replay_count", len(self.replay.episodes))
        self.replay.restore_success_snapshots(
            payload.get("replay_success_snapshots", {})
        )
        episode_ids = payload.get("episode_ids")
        if episode_ids is not None:
            if len(episode_ids) != self.config.num_envs:
                raise ValueError("checkpoint environment count mismatch")
            self.episode_ids = list(episode_ids)
        self.completed_food_episodes = payload.get(
            "completed_food_episodes",
            sum(
                any(row.get("food_events") for row in episode)
                for episode in self.replay.episodes
            ),
        )

    def close(self):
        for env in self.envs:
            env.close()

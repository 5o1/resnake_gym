"""V-trace learner implementation and tensor batching helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from resnake_gym.gamepad_runtime import build_model, stack_observations
from resnake_gym.gamepad_vtrace_contract import (
    VTraceConfig,
    checkpoint_metadata,
    validate_checkpoint,
)
from resnake_gym.gamepad_vtrace_credit import validate_credit_trace
from resnake_gym.models import (
    distribution_from_observation,
    policy_action_entropy,
    policy_action_log_prob,
)
from resnake_gym.training.vtrace_learner_audit import (
    action_audit_metrics,
    credit_trace_metrics,
)
from resnake_gym.vtrace import VTraceReturns, vtrace_from_log_probs


@dataclass(frozen=True)
class _UpdateRequest:
    """Validated behavior-policy provenance for one learner update."""

    target_policy_version: int
    versions_by_trace: list[np.ndarray]
    versions: np.ndarray


@dataclass(frozen=True)
class _VTraceBatch:
    """Padded tensors and fixed targets shared by every BPTT window."""

    target_log_probs: Tensor
    target_slot_log_probs: Tensor
    behavior_slot_log_probs: Tensor
    discounts: Tensor
    returns: VTraceReturns
    mask: Tensor
    lengths: Tensor
    hidden_drift: Tensor
    window_hiddens: dict[int, Tensor]
    loss_weights: Tensor
    denominator: Tensor
    actions_by_trace: list[np.ndarray]

    @property
    def max_length(self) -> int:
        return self.mask.shape[0]


@dataclass(frozen=True)
class _OptimizationResult:
    """Detached objective values produced by one optimizer step."""

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    latent_entropy: Tensor
    entropy_per_action_variable: Tensor
    entropy_normalized: Tensor
    entropy_beta_h: Tensor
    grad_norm: Tensor
    backward_calls: int


def _trace_array(trace: dict[str, Any], key: str) -> np.ndarray:
    return np.concatenate([fragment[key] for fragment in trace["fragments"]], axis=0)


def _trace_observation_at(
    trace: dict[str, Any], time_index: int
) -> dict[str, np.ndarray]:
    if not 0 <= time_index < trace["length"]:
        raise IndexError("credit trace observation index is out of range")
    for fragment in trace["fragments"]:
        if time_index < fragment["length"]:
            return {
                key: values[time_index]
                for key, values in fragment["observations"].items()
            }
        time_index -= fragment["length"]
    raise AssertionError("credit trace length metadata is inconsistent")


def _batch_observation_at(
    traces: list[dict[str, Any]],
    time_index: int,
    device: torch.device,
) -> dict[str, Tensor]:
    observations = [
        _trace_observation_at(trace, min(time_index, trace["length"] - 1))
        for trace in traces
    ]
    return stack_observations(observations, device)


def _batch_burn_observation_at(
    traces: list[dict[str, Any]],
    time_index: int,
    device: torch.device,
) -> dict[str, Tensor]:
    observations = []
    for trace in traces:
        fragment = trace["fragments"][0]
        if fragment["burn_length"]:
            index = min(time_index, fragment["burn_length"] - 1)
            observations.append(
                {
                    key: values[index]
                    for key, values in fragment["burn_observations"].items()
                }
            )
        else:
            observations.append(
                {key: values[0] for key, values in fragment["observations"].items()}
            )
    return stack_observations(observations, device)


def _bootstrap_observations(
    traces: list[dict[str, Any]], device: torch.device
) -> dict[str, Tensor]:
    return stack_observations(
        [trace["fragments"][-1]["bootstrap_observation"] for trace in traces], device
    )


def _padded_trace_array(
    traces: list[dict[str, Any]],
    key: str,
    max_length: int,
    *,
    dtype,
    fill=0,
) -> np.ndarray:
    result = np.full((max_length, len(traces)), fill, dtype=dtype)
    for batch, trace in enumerate(traces):
        result[: trace["length"], batch] = _trace_array(trace, key)
    return result


class GamepadVTraceLearner:
    """One GPU learner with event-scale targets and bounded recurrent BPTT."""

    def __init__(self, config: VTraceConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.model = build_model(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.update_count = 0
        self.logic_ticks = 0
        self.transitions = 0

    def _target_credit_traces(
        self, traces: list[dict[str, Any]]
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[int, Tensor],
    ]:
        """No-gradient pass over whole traces, retaining only window states."""
        c = self.config
        if not traces:
            raise ValueError("V-trace learner requires at least one credit trace")
        for trace in traces:
            validate_credit_trace(trace, c)
        first_fragments = [trace["fragments"][0] for trace in traces]
        lengths = torch.tensor(
            [trace["length"] for trace in traces], device=self.device
        )
        max_length = int(lengths.max())
        behavior_initial_hidden = torch.as_tensor(
            np.stack([fragment["initial_hidden"] for fragment in first_fragments]),
            device=self.device,
            dtype=torch.float32,
        )
        hidden = torch.as_tensor(
            np.stack([fragment["burn_h0"] for fragment in first_fragments]),
            device=self.device,
            dtype=torch.float32,
        )
        burn_lengths = torch.tensor(
            [fragment["burn_length"] for fragment in first_fragments],
            device=self.device,
        )
        max_burn = int(burn_lengths.max())
        actions_by_trace = [_trace_array(trace, "policy_actions") for trace in traces]
        target_log_probs = []
        values = []
        target_slot_log_probs = []
        window_hiddens: dict[int, Tensor] = {}
        with torch.no_grad():
            for time_index in range(max_burn):
                active_indices = torch.nonzero(
                    time_index < burn_lengths, as_tuple=False
                )[:, 0]
                active_traces = [traces[int(index)] for index in active_indices.cpu()]
                observation = _batch_burn_observation_at(
                    active_traces, time_index, self.device
                )
                _, _, candidate_hidden = distribution_from_observation(
                    self.model,
                    observation,
                    hidden.index_select(0, active_indices),
                )
                hidden = hidden.index_copy(0, active_indices, candidate_hidden)
            hidden = hidden.detach()
            hidden_drift = (hidden - behavior_initial_hidden).norm(dim=-1)
            for time_index in range(max_length):
                if time_index % c.bptt_window == 0:
                    window_hiddens[time_index] = hidden.detach().clone()
                active_indices = torch.nonzero(time_index < lengths, as_tuple=False)[
                    :, 0
                ]
                active_traces = [traces[int(index)] for index in active_indices.cpu()]
                observation = _batch_observation_at(
                    active_traces, time_index, self.device
                )
                distribution, value, candidate_hidden = distribution_from_observation(
                    self.model,
                    observation,
                    hidden.index_select(0, active_indices),
                )
                action = torch.as_tensor(
                    np.stack(
                        [
                            actions_by_trace[int(index)][time_index]
                            for index in active_indices.cpu()
                        ]
                    ),
                    device=self.device,
                    dtype=torch.long,
                )
                log_prob = policy_action_log_prob(distribution, action)
                target_log_probs.append(
                    torch.zeros(len(traces), device=self.device).index_copy(
                        0, active_indices, log_prob
                    )
                )
                values.append(
                    torch.zeros(len(traces), device=self.device).index_copy(
                        0, active_indices, value[:, 0]
                    )
                )
                distribution_slot_logits = distribution.logits
                target_slot_log_probs.append(
                    torch.zeros(
                        len(traces),
                        c.chunk_length,
                        5,
                        device=self.device,
                    ).index_copy(0, active_indices, distribution_slot_logits)
                )
                hidden = hidden.index_copy(0, active_indices, candidate_hidden)
            _, bootstrap_value, _ = distribution_from_observation(
                self.model, _bootstrap_observations(traces, self.device), hidden
            )
        bootstrap_value = bootstrap_value[:, 0]
        mask = torch.arange(max_length, device=self.device)[:, None] < lengths[None]
        values = torch.stack(values)
        # For a shorter sequence, constant bootstrap-value padding makes every
        # padded delta exactly zero and carries its own V_T to the global padded
        # horizon.  Padded rows are still excluded from every learner loss.
        values = torch.where(mask, values, bootstrap_value[None])
        return (
            torch.stack(target_log_probs),
            values,
            bootstrap_value,
            mask,
            lengths,
            hidden_drift,
            torch.stack(target_slot_log_probs),
            window_hiddens,
        )

    def _validate_update_request(
        self,
        traces: list[dict[str, Any]],
    ) -> _UpdateRequest:
        if not traces:
            raise ValueError("V-trace learner requires at least one credit trace")
        for trace in traces:
            validate_credit_trace(trace, self.config)
        versions_by_trace = [_trace_array(trace, "policy_versions") for trace in traces]
        versions = np.concatenate(versions_by_trace)
        target_policy_version = self.update_count
        if np.any(versions > target_policy_version):
            raise RuntimeError("behavior policy version is newer than learner target")
        return _UpdateRequest(
            target_policy_version=target_policy_version,
            versions_by_trace=versions_by_trace,
            versions=versions,
        )

    def _prepare_vtrace_batch(self, traces: list[dict[str, Any]]) -> _VTraceBatch:
        c = self.config
        (
            target_log_probs,
            values,
            bootstrap_value,
            mask,
            lengths,
            hidden_drift,
            target_slot_log_probs,
            window_hiddens,
        ) = self._target_credit_traces(traces)
        max_length = mask.shape[0]
        behavior_log_probs = torch.as_tensor(
            _padded_trace_array(
                traces,
                "behavior_log_probs",
                max_length,
                dtype=np.float32,
            ),
            device=self.device,
        )
        behavior_slot_log_probs = torch.zeros_like(target_slot_log_probs)
        for batch_index, trace in enumerate(traces):
            length = trace["length"]
            behavior_slot_log_probs[:length, batch_index] = torch.as_tensor(
                _trace_array(trace, "behavior_slot_log_probs"),
                device=self.device,
                dtype=torch.float32,
            )
        behavior_slot_log_probs = torch.where(
            mask[..., None, None],
            behavior_slot_log_probs,
            target_slot_log_probs.detach(),
        )
        rewards = torch.as_tensor(
            _padded_trace_array(traces, "rewards", max_length, dtype=np.float32),
            device=self.device,
        )
        discounts = torch.as_tensor(
            _padded_trace_array(
                traces,
                "discounts",
                max_length,
                dtype=np.float32,
                fill=1,
            ),
            device=self.device,
        )
        # Equal target/behavior probabilities and constant values make padded
        # V-trace deltas exactly zero; every learner loss still masks them out.
        behavior_log_probs = torch.where(
            mask, behavior_log_probs, target_log_probs.detach()
        )
        rewards = torch.where(mask, rewards, 0.0)
        discounts = torch.where(mask, discounts, 1.0)
        returns = vtrace_from_log_probs(
            target_log_probs,
            behavior_log_probs,
            discounts,
            rewards,
            values,
            bootstrap_value,
            rho_bar=c.rho_bar,
            c_bar=c.c_bar,
            pg_rho_bar=c.pg_rho_bar,
        )
        loss_weights = mask.to(torch.float32)
        return _VTraceBatch(
            target_log_probs=target_log_probs,
            target_slot_log_probs=target_slot_log_probs,
            behavior_slot_log_probs=behavior_slot_log_probs,
            discounts=discounts,
            returns=returns,
            mask=mask,
            lengths=lengths,
            hidden_drift=hidden_drift,
            window_hiddens=window_hiddens,
            loss_weights=loss_weights,
            denominator=loss_weights.sum(),
            actions_by_trace=[
                _trace_array(trace, "policy_actions") for trace in traces
            ],
        )

    def _backward_window(
        self,
        traces: list[dict[str, Any]],
        batch: _VTraceBatch,
        window_start: int,
        window_end: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = batch.window_hiddens[window_start].detach()
        grad_log_probs = []
        grad_values = []
        grad_entropies = []
        for time_index in range(window_start, window_end):
            active_indices = torch.nonzero(time_index < batch.lengths, as_tuple=False)[
                :, 0
            ]
            active_traces = [traces[int(index)] for index in active_indices.cpu()]
            observation = _batch_observation_at(active_traces, time_index, self.device)
            distribution, value, candidate_hidden = distribution_from_observation(
                self.model,
                observation,
                hidden.index_select(0, active_indices),
            )
            action = torch.as_tensor(
                np.stack(
                    [
                        batch.actions_by_trace[int(index)][time_index]
                        for index in active_indices.cpu()
                    ]
                ),
                device=self.device,
                dtype=torch.long,
            )
            grad_log_probs.append(
                torch.zeros(len(traces), device=self.device).index_copy(
                    0,
                    active_indices,
                    policy_action_log_prob(distribution, action),
                )
            )
            grad_values.append(
                torch.zeros(len(traces), device=self.device).index_copy(
                    0, active_indices, value[:, 0]
                )
            )
            grad_entropies.append(
                torch.zeros(len(traces), device=self.device).index_copy(
                    0, active_indices, policy_action_entropy(distribution)
                )
            )
            hidden = hidden.index_copy(0, active_indices, candidate_hidden)
        block_weights = batch.loss_weights[window_start:window_end]
        policy_part = -(
            block_weights
            * torch.stack(grad_log_probs)
            * batch.returns.pg_advantages[window_start:window_end]
        ).sum()
        value_part = (
            block_weights
            * (
                torch.stack(grad_values) - batch.returns.vs[window_start:window_end]
            ).square()
        ).sum()
        entropy_part = (block_weights * torch.stack(grad_entropies)).sum()
        block_loss = (
            policy_part
            + self.config.value_coefficient * value_part
            - self.config.entropy_coefficient * entropy_part
        ) / batch.denominator
        if not torch.isfinite(block_loss):
            raise FloatingPointError("nonfinite V-trace objective")
        block_loss.backward()
        return policy_part.detach(), value_part.detach(), entropy_part.detach()

    def _optimize_batch(
        self, traces: list[dict[str, Any]], batch: _VTraceBatch
    ) -> _OptimizationResult:
        c = self.config
        self.optimizer.zero_grad()
        policy_numerator = torch.zeros((), device=self.device)
        value_numerator = torch.zeros((), device=self.device)
        entropy_numerator = torch.zeros((), device=self.device)
        backward_calls = 0
        for window_start in range(0, batch.max_length, c.bptt_window):
            window_end = min(window_start + c.bptt_window, batch.max_length)
            policy_part, value_part, entropy_part = self._backward_window(
                traces, batch, window_start, window_end
            )
            policy_numerator += policy_part
            value_numerator += value_part
            entropy_numerator += entropy_part
            backward_calls += 1
        policy_loss = policy_numerator / batch.denominator
        value_loss = value_numerator / batch.denominator
        latent_entropy = entropy_numerator / batch.denominator
        action_variable_count = c.chunk_length
        entropy_per_action_variable = latent_entropy / action_variable_count
        entropy_normalized = entropy_per_action_variable / float(np.log(5.0))
        entropy_beta_h = c.entropy_coefficient * latent_entropy
        loss = (
            policy_loss
            + c.value_coefficient * value_loss
            - c.entropy_coefficient * latent_entropy
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), c.max_grad_norm, error_if_nonfinite=True
        )
        # The optimizer creates the next policy version only after all target
        # probabilities have been evaluated with the current parameters.
        self.optimizer.step()
        self.update_count += 1
        return _OptimizationResult(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            latent_entropy=latent_entropy,
            entropy_per_action_variable=entropy_per_action_variable,
            entropy_normalized=entropy_normalized,
            entropy_beta_h=entropy_beta_h,
            grad_norm=grad_norm,
            backward_calls=backward_calls,
        )

    def _loss_metrics(
        self, optimization: _OptimizationResult
    ) -> dict[str, float | int]:
        action_variable_count = self.config.chunk_length
        return {
            "loss": float(optimization.loss.detach()),
            "policy_loss": float(optimization.policy_loss.detach()),
            "value_loss": float(optimization.value_loss.detach()),
            "latent_entropy": float(optimization.latent_entropy.detach()),
            "entropy_joint_nats": float(optimization.latent_entropy.detach()),
            "entropy_action_variables_per_decision": action_variable_count,
            "entropy_per_action_variable_nats": float(
                optimization.entropy_per_action_variable.detach()
            ),
            "entropy_normalized_fraction": float(
                optimization.entropy_normalized.detach()
            ),
            "entropy_beta_h": float(optimization.entropy_beta_h.detach()),
            "grad_norm": float(optimization.grad_norm.detach()),
        }

    def update(
        self,
        traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply exactly one optimizer step to complete event-scale traces."""
        request = self._validate_update_request(traces)
        self.model.train()
        batch = self._prepare_vtrace_batch(traces)
        optimization = self._optimize_batch(traces, batch)
        valid_count = int(batch.mask.sum())
        logic_ticks = sum(
            int(_trace_array(trace, "ticks_advanced").sum()) for trace in traces
        )
        self.logic_ticks += logic_ticks
        self.transitions += valid_count
        return {
            "learner_source": "fresh",
            "learner_update": self.update_count,
            "target_policy_version": request.target_policy_version,
            "learner_transitions": valid_count,
            "learner_logic_ticks": logic_ticks,
            **self._loss_metrics(optimization),
            **action_audit_metrics(
                target_policy_version=request.target_policy_version,
                versions=request.versions,
                returns=batch.returns,
                mask=batch.mask,
                target_slot_log_probs=batch.target_slot_log_probs,
                behavior_slot_log_probs=batch.behavior_slot_log_probs,
                rho_bar=self.config.rho_bar,
                device=self.device,
            ),
            "cumulative_trained_logic_ticks": self.logic_ticks,
            "cumulative_trained_transitions": self.transitions,
            **credit_trace_metrics(
                traces=traces,
                versions_by_trace=request.versions_by_trace,
                discounts=batch.discounts,
                returns=batch.returns,
                mask=batch.mask,
                hidden_drift=batch.hidden_drift,
                c_bar=self.config.c_bar,
                bptt_window=self.config.bptt_window,
                backward_calls=optimization.backward_calls,
            ),
        }

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            **checkpoint_metadata(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update": self.update_count,
            "logic_ticks": self.logic_ticks,
            "transitions": self.transitions,
            "note": (
                "learner state only; the credit assembler is a separate runtime "
                "sidecar; actor games and actor RNG are not checkpointed"
            ),
        }
        payload["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        return payload

    def restore(self, payload: dict[str, Any]) -> None:
        validate_checkpoint(payload)
        restored = VTraceConfig(**payload["config"])
        topology_keys = {"actor_processes", "envs_per_actor", "queue_capacity"}
        for key in asdict(restored).keys() - topology_keys:
            if getattr(restored, key) != getattr(self.config, key):
                raise ValueError(f"resume config mismatch: {key}")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = int(payload.get("update", 0))
        self.logic_ticks = int(payload.get("logic_ticks", 0))
        self.transitions = int(payload.get("transitions", 0))
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_states" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])

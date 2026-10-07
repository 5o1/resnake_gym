"""Pure metric assembly for one V-trace learner update."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from resnake_gym.vtrace import VTraceReturns


def _trace_array(trace: dict[str, Any], key: str) -> np.ndarray:
    return np.concatenate([fragment[key] for fragment in trace["fragments"]], axis=0)


def importance_effective_sample_size(log_weights: Tensor) -> Tensor:
    """Compute ratio ESS without exponentiating unbounded log weights."""
    if log_weights.ndim != 1 or not log_weights.numel():
        raise ValueError("importance log weights must be a non-empty vector")
    if not torch.isfinite(log_weights).all():
        raise ValueError("importance log weights must be finite")
    centered = log_weights - log_weights.max()
    log_ess = 2 * torch.logsumexp(centered, dim=0) - torch.logsumexp(
        2 * centered, dim=0
    )
    return log_ess.exp().clamp(max=log_weights.numel())


def action_audit_metrics(
    *,
    target_policy_version: int,
    versions: np.ndarray,
    returns: VTraceReturns,
    mask: Tensor,
    target_slot_log_probs: Tensor,
    behavior_slot_log_probs: Tensor,
    rho_bar: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Measure behavior/target drift over valid action transitions."""
    valid_rhos = returns.rhos[mask]
    valid_log_rhos = returns.log_rhos[mask]
    target_probabilities = target_slot_log_probs.detach().exp()
    policy_kl = (
        (
            target_probabilities
            * (target_slot_log_probs.detach() - behavior_slot_log_probs)
        )
        .sum((-1, -2))
        .clamp_min(0)
    )
    valid_policy_kl = policy_kl[mask]
    valid_count = int(mask.sum())
    lags = target_policy_version - torch.as_tensor(versions, device=device)
    effective_sample_size = importance_effective_sample_size(valid_log_rhos)
    # Scale by the largest ratio so a finite float32 mean cannot overflow.
    rho_max = valid_rhos.max()
    rho_mean = rho_max * (valid_rhos / rho_max).mean() if bool(rho_max > 0) else rho_max
    return {
        "rho_mean": float(rho_mean),
        "rho_max": float(rho_max),
        "rho_clipped_fraction": float((valid_rhos > rho_bar).float().mean()),
        "importance_ess": float(effective_sample_size),
        "importance_ess_fraction": float(effective_sample_size / valid_count),
        "sample_behavior_target_log_ratio": float((-valid_log_rhos).mean()),
        "behavior_target_joint_kl_mean": float(valid_policy_kl.mean()),
        "behavior_target_joint_kl_max": float(valid_policy_kl.max()),
        "actor_lag_mean": float(lags.float().mean()),
        "actor_lag_max": int(lags.max()),
    }


def credit_trace_metrics(
    *,
    traces: list[dict[str, Any]],
    versions_by_trace: list[np.ndarray],
    discounts: Tensor,
    returns: VTraceReturns,
    mask: Tensor,
    hidden_drift: Tensor,
    c_bar: float,
    bptt_window: int,
    backward_calls: int,
) -> dict[str, float | int]:
    """Summarize event-credit boundaries, segmentation and recurrent windows."""
    trace_lengths = np.asarray([trace["length"] for trace in traces])
    segment_counts = np.asarray([trace["segment_count"] for trace in traces])
    segment_lengths = np.asarray(
        [fragment["length"] for trace in traces for fragment in trace["fragments"]]
    )
    mixed_versions = np.asarray([len(np.unique(item)) for item in versions_by_trace])
    cap_hits = sum(trace["credit_boundary"] == "safety_cap" for trace in traces)
    over_128 = int((trace_lengths > 128).sum())
    bootstrap_boundaries = sum(
        float(_trace_array(trace, "discounts")[-1]) > 0 for trace in traces
    )
    artificial_boundaries = sum(
        trace["credit_boundary"] in ("safety_cap", "resume") for trace in traces
    )
    transport_logs = []
    log_c_bar = float(np.log(c_bar))
    for batch_index, length in enumerate(trace_lengths):
        if length <= 1:
            transport_logs.append(0.0)
            continue
        trace_discounts = discounts[: length - 1, batch_index]
        trace_log_cs = torch.minimum(
            returns.log_rhos[: length - 1, batch_index],
            torch.full_like(returns.log_rhos[: length - 1, batch_index], log_c_bar),
        )
        transport_logs.append(float((trace_discounts.log() + trace_log_cs).sum()))
    transport_logs_array = np.asarray(transport_logs, dtype=np.float64)
    return {
        "learner_fragments": int(segment_counts.sum()),
        "unroll_max": int(segment_lengths.max()),
        "unroll_mean": float(segment_lengths.mean()),
        "padding_fraction": float(1 - int(mask.sum()) / mask.numel()),
        "recurrent_burn_in_max": max(
            trace["fragments"][0]["burn_length"] for trace in traces
        ),
        "recurrent_hidden_drift_mean": float(hidden_drift.mean()),
        "recurrent_hidden_drift_max": float(hidden_drift.max()),
        "credit_trace_count": len(traces),
        "credit_trace_len_mean": float(trace_lengths.mean()),
        "credit_trace_len_p50": float(np.percentile(trace_lengths, 50)),
        "credit_trace_len_p90": float(np.percentile(trace_lengths, 90)),
        "credit_trace_len_p99": float(np.percentile(trace_lengths, 99)),
        "credit_trace_len_max": int(trace_lengths.max()),
        "credit_trace_cap_hit_count": cap_hits,
        "credit_trace_cap_hit_rate": cap_hits / len(traces),
        "credit_trace_over_128_count": over_128,
        "credit_trace_over_128_rate": over_128 / len(traces),
        "segments_per_trace_mean": float(segment_counts.mean()),
        "segments_per_trace_max": int(segment_counts.max()),
        "mixed_policy_versions_mean": float(mixed_versions.mean()),
        "mixed_policy_versions_max": int(mixed_versions.max()),
        "unique_behavior_versions_mean": float(mixed_versions.mean()),
        "unique_behavior_versions_max": int(mixed_versions.max()),
        "transport_log_weight_mean": float(transport_logs_array.mean()),
        "transport_log_weight_min": float(transport_logs_array.min()),
        "bptt_windows_per_update": backward_calls,
        "bptt_backward_calls": backward_calls,
        "trace_window_count": int(
            sum(
                (int(length) + bptt_window - 1) // bptt_window
                for length in trace_lengths
            )
        ),
        "credit_bootstrap_boundary_count": bootstrap_boundaries,
        "credit_bootstrap_boundary_rate": bootstrap_boundaries / len(traces),
        "credit_artificial_boundary_count": artificial_boundaries,
        "credit_artificial_boundary_rate": artificial_boundaries / len(traces),
    }

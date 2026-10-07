"""V-trace targets for recurrent, variable-duration actor-critic rollouts.

This is the IMPALA V-trace recursion from Espeholt et al. (2018).  The
caller supplies one already-summed *joint* action log-probability per
``[time, batch]`` transition.  For ReSnake, that is the log-probability of
the complete gamepad chunk, not a per-slot probability.

``discounts[t, b]`` is also supplied per transition.  A variable-duration
transition lasting ``K`` game ticks should therefore pass ``gamma ** K``;
a terminal transition passes zero.  The function deliberately does not
infer either duration or termination from another tensor.

The returned targets are detached.  Use ``pg_advantages`` as a constant in
the policy loss and ``vs`` as a constant target in the value loss.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor


class VTraceReturns(NamedTuple):
    """Detached V-trace targets and importance-weight diagnostics."""

    vs: Tensor
    pg_advantages: Tensor
    log_rhos: Tensor
    rhos: Tensor
    clipped_rhos: Tensor
    cs: Tensor
    clipped_pg_rhos: Tensor


def _validate_inputs(
    target_action_log_probs: Tensor,
    behavior_action_log_probs: Tensor,
    discounts: Tensor,
    rewards: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
) -> None:
    sequence_shape = rewards.shape
    named_sequences = {
        "target_action_log_probs": target_action_log_probs,
        "behavior_action_log_probs": behavior_action_log_probs,
        "discounts": discounts,
        "rewards": rewards,
        "values": values,
    }
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [T, B], got {sequence_shape}")
    for name, tensor in named_sequences.items():
        if tensor.shape != sequence_shape:
            raise ValueError(
                f"{name} must have shape {sequence_shape}, got {tensor.shape}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if tensor.device != rewards.device:
            raise ValueError(f"{name} must be on device {rewards.device}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
    expected_bootstrap_shape = (sequence_shape[1],)
    if bootstrap_value.shape != expected_bootstrap_shape:
        raise ValueError(
            "bootstrap_value must have shape [B], "
            f"expected {expected_bootstrap_shape}, got {bootstrap_value.shape}"
        )
    if not bootstrap_value.is_floating_point():
        raise TypeError("bootstrap_value must be a floating-point tensor")
    if bootstrap_value.device != rewards.device:
        raise ValueError(f"bootstrap_value must be on device {rewards.device}")
    if not torch.isfinite(bootstrap_value).all():
        raise ValueError("bootstrap_value must contain only finite values")
    if torch.any((discounts < 0) | (discounts > 1)):
        raise ValueError("discounts must be within [0, 1]")


def _validate_clip(name: str, threshold: float | None) -> None:
    if threshold is not None and threshold <= 0:
        raise ValueError(f"{name} must be positive or None")


def _max_finite_exp_input(values: Tensor) -> Tensor:
    """Largest log value whose exponential is finite in ``values.dtype``."""
    log_max = torch.log(
        torch.tensor(torch.finfo(values.dtype).max, device=values.device)
    )
    return torch.nextafter(log_max, torch.full_like(log_max, -torch.inf))


def _ratio_from_log(log_rhos: Tensor, threshold: float | None = None) -> Tensor:
    """Exponentiate a log ratio with an optional upper clip, without overflow."""
    upper_log = _max_finite_exp_input(log_rhos)
    if threshold is not None:
        log_threshold = torch.tensor(
            math.log(threshold), dtype=log_rhos.dtype, device=log_rhos.device
        )
        upper_log = torch.minimum(upper_log, log_threshold)
    return torch.exp(torch.clamp(log_rhos, max=upper_log))


def vtrace_from_log_probs(
    target_action_log_probs: Tensor,
    behavior_action_log_probs: Tensor,
    discounts: Tensor,
    rewards: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
    *,
    rho_bar: float | None = 1.0,
    c_bar: float | None = 1.0,
    pg_rho_bar: float | None = 1.0,
) -> VTraceReturns:
    r"""Compute finite-horizon IMPALA V-trace targets.

    Let ``rho_t = pi(a_t|x_t) / mu(a_t|x_t)``.  This implements

    .. math::

       \delta_t = \min(\bar\rho, \rho_t)
          (r_t + d_t V_{t+1} - V_t)

       v_s = V_s + \sum_{t=s}^{T-1}
          \left(\prod_{i=s}^{t-1} d_i\min(\bar c,\rho_i)\right)\delta_t

    and the policy-gradient advantage

    .. math::

       \min(\bar\rho_{pg},\rho_t)(r_t + d_t v_{t+1} - V_t).

    ``bootstrap_value`` is both ``V_T`` in the target recursion and
    ``v_T`` in the final policy-gradient advantage.  Set a terminal
    transition's discount to zero, so values after that boundary cannot
    affect preceding targets.

    Args:
        target_action_log_probs: Target-policy joint chunk log-probabilities,
            shape ``[T, B]``.
        behavior_action_log_probs: Behavior-policy joint chunk
            log-probabilities recorded by actors, shape ``[T, B]``.
        discounts: Explicit per-transition SMDP discounts, shape ``[T, B]``.
        rewards: Discounted rewards aggregated within each transition,
            shape ``[T, B]``.
        values: Target-policy baseline values ``V_0 .. V_{T-1}``, shape
            ``[T, B]``.
        bootstrap_value: Baseline value ``V_T``, shape ``[B]``.
        rho_bar: Value-target importance-ratio clip, or ``None`` for no clip.
        c_bar: Trace importance-ratio clip, or ``None`` for no clip.
        pg_rho_bar: Policy-gradient importance-ratio clip, or ``None``.

    Returns:
        Detached targets and ratio diagnostics.  No output retains an
        autograd edge to any input.
    """
    _validate_inputs(
        target_action_log_probs,
        behavior_action_log_probs,
        discounts,
        rewards,
        values,
        bootstrap_value,
    )
    _validate_clip("rho_bar", rho_bar)
    _validate_clip("c_bar", c_bar)
    _validate_clip("pg_rho_bar", pg_rho_bar)
    if rho_bar is not None and c_bar is not None and c_bar > rho_bar:
        raise ValueError("c_bar cannot exceed rho_bar")

    with torch.no_grad():
        log_rhos = target_action_log_probs - behavior_action_log_probs
        # Finite inputs can still overflow when subtracted at the dtype limit.
        # Saturating that diagnostic preserves its sign and keeps every returned
        # ratio finite.  Training clips are applied in log space before exp, so
        # an extreme off-policy sample cannot overflow on its way to a small
        # V-trace threshold such as one.
        limit = torch.finfo(log_rhos.dtype).max
        log_rhos = torch.nan_to_num(
            log_rhos,
            nan=0.0,
            posinf=limit,
            neginf=-limit,
        )
        rhos = _ratio_from_log(log_rhos)
        clipped_rhos = _ratio_from_log(log_rhos, rho_bar)
        cs = _ratio_from_log(log_rhos, c_bar)
        clipped_pg_rhos = _ratio_from_log(log_rhos, pg_rho_bar)

        if rewards.shape[0] == 0:
            empty = values.clone()
            return VTraceReturns(
                vs=empty,
                pg_advantages=empty.clone(),
                log_rhos=log_rhos,
                rhos=rhos,
                clipped_rhos=clipped_rhos,
                cs=cs,
                clipped_pg_rhos=clipped_pg_rhos,
            )

        values_t_plus_1 = torch.cat((values[1:], bootstrap_value.unsqueeze(0)))
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        correction = torch.zeros_like(bootstrap_value)
        vs_minus_values = []
        for index in range(rewards.shape[0] - 1, -1, -1):
            correction = deltas[index] + discounts[index] * cs[index] * correction
            vs_minus_values.append(correction)
        vs_minus_values.reverse()
        vs = values + torch.stack(vs_minus_values)

        vs_t_plus_1 = torch.cat((vs[1:], bootstrap_value.unsqueeze(0)))
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return VTraceReturns(
        vs=vs,
        pg_advantages=pg_advantages,
        log_rhos=log_rhos,
        rhos=rhos,
        clipped_rhos=clipped_rhos,
        cs=cs,
        clipped_pg_rhos=clipped_pg_rhos,
    )


__all__ = ["VTraceReturns", "vtrace_from_log_probs"]

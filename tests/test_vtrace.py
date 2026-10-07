import math

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.vtrace import vtrace_from_log_probs  # noqa: E402


def _tensor(values, *, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def test_rho_one_matches_hand_computed_impala_recursion():
    result = vtrace_from_log_probs(
        target_action_log_probs=_tensor([[0.0], [0.0]]),
        behavior_action_log_probs=_tensor([[0.0], [0.0]]),
        discounts=_tensor([[0.5], [0.25]]),
        rewards=_tensor([[1.0], [2.0]]),
        values=_tensor([[0.2], [0.4]]),
        bootstrap_value=_tensor([0.8]),
    )

    assert torch.allclose(result.rhos, _tensor([[1.0], [1.0]]))
    assert torch.allclose(result.vs, _tensor([[2.1], [2.2]]))
    assert torch.allclose(result.pg_advantages, _tensor([[1.9], [1.8]]))


def test_value_trace_and_policy_gradient_use_their_own_ratio_clips():
    log_rhos = _tensor([[math.log(4.0)], [math.log(0.5)]])
    result = vtrace_from_log_probs(
        target_action_log_probs=log_rhos,
        behavior_action_log_probs=_tensor([[0.0], [0.0]]),
        discounts=_tensor([[1.0], [1.0]]),
        rewards=_tensor([[0.0], [1.0]]),
        values=_tensor([[0.0], [0.0]]),
        bootstrap_value=_tensor([0.0]),
        rho_bar=2.0,
        c_bar=1.0,
        pg_rho_bar=3.0,
    )

    assert torch.allclose(result.rhos, _tensor([[4.0], [0.5]]))
    assert torch.allclose(result.clipped_rhos, _tensor([[2.0], [0.5]]))
    assert torch.allclose(result.cs, _tensor([[1.0], [0.5]]))
    assert torch.allclose(result.clipped_pg_rhos, _tensor([[3.0], [0.5]]))
    assert torch.allclose(result.vs, _tensor([[0.5], [0.5]]))
    assert torch.allclose(result.pg_advantages, _tensor([[1.5], [0.5]]))


def test_variable_duration_smdp_discounts_are_used_per_transition():
    # gamma=0.5 with durations K=(1, 2), hence discounts=(0.5, 0.25).
    result = vtrace_from_log_probs(
        target_action_log_probs=_tensor([[0.0], [0.0]]),
        behavior_action_log_probs=_tensor([[0.0], [0.0]]),
        discounts=_tensor([[0.5], [0.25]]),
        rewards=_tensor([[1.0], [2.0]]),
        values=_tensor([[0.0], [0.0]]),
        bootstrap_value=_tensor([4.0]),
    )

    assert torch.allclose(result.vs, _tensor([[2.5], [3.0]]))
    assert torch.allclose(result.pg_advantages, _tensor([[2.5], [3.0]]))


def test_zero_discount_cuts_trace_at_terminal_boundary():
    result = vtrace_from_log_probs(
        target_action_log_probs=_tensor([[0.0], [0.0], [0.0]]),
        behavior_action_log_probs=_tensor([[0.0], [0.0], [0.0]]),
        discounts=_tensor([[1.0], [0.0], [1.0]]),
        rewards=_tensor([[0.0], [2.0], [100.0]]),
        values=_tensor([[0.0], [0.0], [0.0]]),
        bootstrap_value=_tensor([50.0]),
    )

    # The post-terminal 150 target cannot leak across transition 1.
    assert torch.allclose(result.vs, _tensor([[2.0], [2.0], [150.0]]))
    assert torch.allclose(result.pg_advantages[:2], _tensor([[2.0], [2.0]]))


def test_targets_detach_inputs_but_support_external_actor_critic_losses():
    target_log_probs = _tensor([[0.0], [0.0]], requires_grad=True)
    behavior_log_probs = _tensor([[0.0], [0.0]], requires_grad=True)
    values = _tensor([[0.0], [0.0]], requires_grad=True)
    rewards = _tensor([[1.0], [2.0]], requires_grad=True)
    result = vtrace_from_log_probs(
        target_action_log_probs=target_log_probs,
        behavior_action_log_probs=behavior_log_probs,
        discounts=_tensor([[0.5], [0.0]]),
        rewards=rewards,
        values=values,
        bootstrap_value=_tensor([99.0], requires_grad=True),
    )

    for output in result:
        assert not output.requires_grad
        assert output.grad_fn is None

    policy_loss = -(target_log_probs * result.pg_advantages).sum()
    value_loss = 0.5 * (values - result.vs).square().sum()
    (policy_loss + value_loss).backward()

    assert torch.allclose(target_log_probs.grad, -result.pg_advantages)
    assert torch.allclose(values.grad, values.detach() - result.vs)
    assert behavior_log_probs.grad is None
    assert rewards.grad is None


def test_joint_log_probs_must_be_one_scalar_per_time_and_batch():
    with pytest.raises(ValueError, match=r"shape \[T, B\]"):
        vtrace_from_log_probs(
            target_action_log_probs=_tensor([[[0.0, 0.0]]]),
            behavior_action_log_probs=_tensor([[[0.0, 0.0]]]),
            discounts=_tensor([[[1.0, 1.0]]]),
            rewards=_tensor([[[0.0, 0.0]]]),
            values=_tensor([[[0.0, 0.0]]]),
            bootstrap_value=_tensor([0.0]),
        )


@pytest.mark.parametrize("name", ["rho_bar", "c_bar", "pg_rho_bar"])
def test_importance_ratio_clip_must_be_positive(name):
    kwargs = {name: 0.0}
    with pytest.raises(ValueError, match=name):
        vtrace_from_log_probs(
            target_action_log_probs=_tensor([[0.0]]),
            behavior_action_log_probs=_tensor([[0.0]]),
            discounts=_tensor([[0.0]]),
            rewards=_tensor([[0.0]]),
            values=_tensor([[0.0]]),
            bootstrap_value=_tensor([0.0]),
            **kwargs,
        )


def test_trace_clip_cannot_exceed_value_ratio_clip():
    with pytest.raises(ValueError, match="c_bar cannot exceed rho_bar"):
        vtrace_from_log_probs(
            target_action_log_probs=_tensor([[0.0]]),
            behavior_action_log_probs=_tensor([[0.0]]),
            discounts=_tensor([[0.0]]),
            rewards=_tensor([[0.0]]),
            values=_tensor([[0.0]]),
            bootstrap_value=_tensor([0.0]),
            rho_bar=1.0,
            c_bar=2.0,
        )


def test_nonfinite_inputs_and_invalid_discounts_are_rejected():
    common = dict(
        target_action_log_probs=_tensor([[0.0]]),
        behavior_action_log_probs=_tensor([[0.0]]),
        rewards=_tensor([[0.0]]),
        values=_tensor([[0.0]]),
        bootstrap_value=_tensor([0.0]),
    )
    with pytest.raises(ValueError, match="discounts must be within"):
        vtrace_from_log_probs(discounts=_tensor([[1.1]]), **common)
    common["target_action_log_probs"] = _tensor([[math.nan]])
    with pytest.raises(ValueError, match="finite"):
        vtrace_from_log_probs(discounts=_tensor([[0.0]]), **common)


def test_extreme_log_ratios_keep_raw_diagnostics_and_clipped_targets_finite():
    result = vtrace_from_log_probs(
        target_action_log_probs=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        behavior_action_log_probs=torch.tensor(
            [[-1_000.0], [-100.0]], dtype=torch.float32
        ),
        discounts=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        rewards=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        values=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        bootstrap_value=torch.tensor([0.0], dtype=torch.float32),
    )

    for value in result:
        assert torch.isfinite(value).all()
    assert torch.equal(result.clipped_rhos, torch.ones_like(result.clipped_rhos))
    assert torch.equal(result.cs, torch.ones_like(result.cs))
    assert torch.equal(result.clipped_pg_rhos, torch.ones_like(result.clipped_pg_rhos))

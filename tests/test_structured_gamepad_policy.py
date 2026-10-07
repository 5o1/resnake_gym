import pytest

torch = pytest.importorskip("torch")

from resnake_gym import gamepad  # noqa: E402
from resnake_gym.models import (  # noqa: E402
    DPAD5_LABELS,
    GamepadPolicy,
    policy_action_entropy,
    policy_action_log_prob,
    reports_from_policy_action,
    sample_policy_action,
)


def policy_inputs(batch=2):
    board = torch.zeros(batch, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    return board, torch.zeros(batch, 20), torch.zeros(batch, 3)


def test_dpad5_distribution_is_one_independent_category_per_chunk_position():
    model = GamepadPolicy(dim=16, chunk_length=3)
    distribution, value, hidden = model.distribution(*policy_inputs())

    assert isinstance(distribution, torch.distributions.Categorical)
    assert distribution.logits.shape == (2, 3, 5)
    assert distribution.batch_shape == (2, 3)
    assert distribution.event_shape == ()
    assert value.shape == (2, 1)
    assert hidden.shape == (2, 16)


def test_dpad5_categories_map_to_exact_legal_full_gamepad_reports():
    logits = torch.zeros(1, 5, 5)
    distribution = torch.distributions.Categorical(logits=logits)
    categories = torch.arange(5).unsqueeze(0)
    reports = reports_from_policy_action(distribution, categories)

    expected = torch.zeros(1, 5, 20)
    expected[0, 1, 0] = 1
    expected[0, 2, 1] = 1
    expected[0, 3, 2] = 1
    expected[0, 4, 3] = 1
    torch.testing.assert_close(reports, expected)
    torch.testing.assert_close(reports[0], torch.tensor(gamepad.dpad5_reports()))
    assert DPAD5_LABELS == ("neutral", "up", "down", "left", "right")
    assert DPAD5_LABELS is gamepad.DPAD5_LABELS
    assert (reports[..., :4].sum(-1) <= 1).all()
    assert (reports[..., 4:] == 0).all()


def test_dpad5_sampling_scoring_and_entropy_are_exact_categorical_quantities():
    torch.manual_seed(5)
    logits = torch.randn(4, 3, 5)
    distribution = torch.distributions.Categorical(logits=logits)
    action = sample_policy_action(distribution)

    assert action.shape == (4, 3)
    torch.testing.assert_close(
        policy_action_log_prob(distribution, action),
        distribution.log_prob(action).sum(-1),
    )
    torch.testing.assert_close(
        policy_action_entropy(distribution), distribution.entropy().sum(-1)
    )
    torch.testing.assert_close(
        sample_policy_action(distribution, stochastic=False), logits.argmax(-1)
    )


def test_dpad5_forward_is_legal_and_has_straight_through_gradient():
    torch.manual_seed(7)
    model = GamepadPolicy(dim=16, chunk_length=3)
    reports, _, _ = model(*policy_inputs())

    assert reports.shape == (2, 3, 20)
    assert ((reports == 0) | (reports == 1)).all()
    assert (reports[..., :4].sum(-1) <= 1).all()
    assert (reports[..., 4:] == 0).all()

    weights = torch.arange(1, 21, dtype=reports.dtype)
    (reports * weights).sum().backward()
    assert model.actor.weight.grad is not None
    assert torch.isfinite(model.actor.weight.grad).all()
    assert model.actor.weight.grad.abs().sum() > 0


def test_dpad5_rejects_invalid_categories():
    distribution = torch.distributions.Categorical(logits=torch.zeros(1, 1, 5))
    with pytest.raises(TypeError, match="integer category"):
        reports_from_policy_action(distribution, torch.tensor([[1.0]]))
    with pytest.raises(ValueError, match="outside"):
        reports_from_policy_action(distribution, torch.tensor([[-1]]))

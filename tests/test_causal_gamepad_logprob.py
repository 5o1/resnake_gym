"""The raw head scores only controls that can affect Snake."""

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.models import GamepadPolicy, sample_log_prob  # noqa: E402


def _distribution(model):
    board = torch.zeros(2, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    return model.distribution(board, torch.zeros(2, 20), torch.zeros(2, 3))[0]


def test_raw_log_probability_ignores_noncausal_controller_outputs():
    torch.manual_seed(7)
    model = GamepadPolicy(dim=16, chunk_length=2, action_head="raw")
    buttons = torch.zeros(2, 2, 14)
    axes = torch.full((2, 2, 6), 0.25)
    before = sample_log_prob(_distribution(model), buttons, axes)

    with torch.no_grad():
        bias = model.actor.bias.reshape(2, 20)
        bias[:, 4:14].add_(10.0)
        bias[:, 16:20].sub_(10.0)

    after = sample_log_prob(_distribution(model), buttons, axes)
    torch.testing.assert_close(after, before)


def test_raw_log_probability_gradient_reaches_only_causal_controls():
    torch.manual_seed(11)
    model = GamepadPolicy(dim=16, chunk_length=2, action_head="raw")
    distribution = _distribution(model)
    buttons = torch.zeros(2, 2, 14)
    axes = distribution[1].loc.detach() + 0.5

    (-sample_log_prob(distribution, buttons, axes).sum()).backward()
    gradient = model.actor.bias.grad.reshape(2, 20)

    assert (gradient[:, :4].abs() > 0).all()
    assert (gradient[:, 14:16].abs() > 0).all()
    torch.testing.assert_close(gradient[:, 4:14], torch.zeros_like(gradient[:, 4:14]))
    torch.testing.assert_close(gradient[:, 16:], torch.zeros_like(gradient[:, 16:]))

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.models import GamepadPolicy, hard_controls_with_gradient  # noqa: E402


@pytest.mark.parametrize("size", [(20, 31), (31, 20), (17, 43)])
def test_same_weights_variable_boards_and_downstream_gradient(size):
    torch.manual_seed(42)
    model = GamepadPolicy(dim=16, chunk_length=3)
    downstream = torch.nn.Linear(20, 5)
    height, width = size
    board = torch.zeros(2, 9, height, width)
    board[:, 2, 0, 0] = 1  # Exercise border patch padding.
    report, value, hidden = model(board, torch.zeros(2, 20), torch.zeros(2, 3))
    assert report.shape == (2, 3, 20)
    assert ((report[..., :14] == 0) | (report[..., :14] == 1)).all()
    # This is a downstream toy module, NOT a tested VLA or robot simulator.
    downstream(report).square().mean().backward()
    assert model.actor.weight.grad.abs().sum() > 0
    assert model.encoder.spatial[0].weight.grad.abs().sum() > 0
    assert model.encoder.local[0].weight.grad.abs().sum() > 0
    assert downstream.weight.grad.abs().sum() > 0
    assert torch.isfinite(hidden).all() and torch.isfinite(value).all()


def test_single_checkpoint_accepts_multiple_sizes():
    model = GamepadPolicy(dim=16)
    for h, w in [(20, 31), (31, 20), (25, 40)]:
        board = torch.zeros(1, 9, h, w)
        board[:, 2, h // 2, w // 2] = 1
        report, _, _ = model(board, torch.zeros(1, 20), torch.zeros(1, 3))
        assert report.shape == (1, 8, 20)


def test_stochastic_hard_forward_soft_backward():
    logits = torch.zeros(4, 20, requires_grad=True)
    report = hard_controls_with_gradient(logits)
    assert ((report[:, :14] == 0) | (report[:, :14] == 1)).all()
    report.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(0) > 0).all()


def test_score_function_distributions_are_mixed():
    model = GamepadPolicy(dim=16, chunk_length=2)
    board = torch.zeros(1, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    (buttons, axes), _, _ = model.distribution(
        board, torch.zeros(1, 20), torch.zeros(1, 3)
    )
    assert buttons.sample().shape == (1, 2, 14)
    assert axes.sample().shape == (1, 2, 6)
    loss = -(
        buttons.log_prob(buttons.sample()).sum() + axes.log_prob(axes.sample()).sum()
    )
    loss.backward()
    assert model.actor.weight.grad.abs().sum() > 0

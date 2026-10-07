import pytest

torch = pytest.importorskip("torch")

from resnake_gym.models import GamepadPolicy  # noqa: E402


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


def test_score_function_distribution_is_categorical_dpad5():
    model = GamepadPolicy(dim=16, chunk_length=2)
    board = torch.zeros(1, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    distribution, _, _ = model.distribution(
        board, torch.zeros(1, 20), torch.zeros(1, 3)
    )
    assert distribution.sample().shape == (1, 2)
    loss = -distribution.log_prob(distribution.sample()).sum()
    loss.backward()
    assert model.actor.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("action_head", ["raw", "dpad5"])
def test_both_action_heads_expose_differentiable_20_control_chunks(action_head):
    model = GamepadPolicy(dim=16, chunk_length=2, action_head=action_head)
    board = torch.zeros(1, 9, 20, 31)
    board[:, 2, 10, 15] = 1

    report, _, _ = model(board, torch.zeros(1, 20), torch.zeros(1, 3))
    assert report.shape == (1, 2, 20)

    weights = torch.arange(1, 21, dtype=report.dtype)
    (report * weights).sum().backward()
    assert model.actor.weight.grad is not None
    assert torch.isfinite(model.actor.weight.grad).all()
    assert model.actor.weight.grad.abs().sum() > 0

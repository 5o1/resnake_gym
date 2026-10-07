import pytest

torch = pytest.importorskip("torch")

from resnake_gym.models import GamepadPolicy, sample_log_prob  # noqa: E402


def policy_distribution(model):
    board = torch.zeros(2, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    return model.distribution(board, torch.zeros(2, 20), torch.zeros(2, 3))[0]


@pytest.mark.parametrize("chunk_rho", [None, 0.7])
def test_snake_log_prob_is_invariant_to_noncausal_controller_outputs(chunk_rho):
    torch.manual_seed(7)
    model = GamepadPolicy(dim=16, chunk_length=3, chunk_rho=chunk_rho)
    distributions = policy_distribution(model)
    buttons = torch.zeros(2, 3, 14)
    axes = torch.full((2, 3, 6), 0.25)
    before = sample_log_prob(distributions, buttons, axes)
    entropy_before = sum(d.entropy().sum((-1, -2)) for d in distributions)

    with torch.no_grad():
        bias = model.actor.bias.reshape(3, 20)
        bias[:, 4:14].add_(10.0)
        bias[:, 16:20].sub_(10.0)

    distributions = policy_distribution(model)
    after = sample_log_prob(distributions, buttons, axes)
    entropy_after = sum(d.entropy().sum((-1, -2)) for d in distributions)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(entropy_after, entropy_before)

    # Sampling and model attributes still expose every controller field.
    assert distributions[0].sample().shape == (2, 3, 14)
    assert distributions[1].sample().shape == (2, 3, 6)
    button_location = (
        distributions[0].logits if chunk_rho is None else distributions[0].loc
    )
    assert button_location.shape == (2, 3, 14)
    assert distributions[1].loc.shape == (2, 3, 6)


def test_snake_log_prob_gradients_only_reach_dpad_and_left_stick():
    torch.manual_seed(11)
    model = GamepadPolicy(dim=16, chunk_length=2)
    distributions = policy_distribution(model)
    buttons = torch.zeros(2, 2, 14)
    # Offset from the means so both scored Gaussian axes have nonzero gradients.
    axes = distributions[1].loc.detach() + 0.5

    loss = -sample_log_prob(distributions, buttons, axes).sum()
    loss.backward()
    gradient = model.actor.bias.grad.reshape(2, 20)

    assert (gradient[:, :4].abs() > 0).all()
    assert (gradient[:, 14:16].abs() > 0).all()
    torch.testing.assert_close(gradient[:, 4:14], torch.zeros_like(gradient[:, 4:14]))
    torch.testing.assert_close(gradient[:, 16:], torch.zeros_like(gradient[:, 16:]))


def test_causal_controller_outputs_still_change_snake_log_prob():
    torch.manual_seed(13)
    model = GamepadPolicy(dim=16, chunk_length=2)
    distributions = policy_distribution(model)
    buttons = torch.zeros(2, 2, 14)
    axes = torch.zeros(2, 2, 6)
    before = sample_log_prob(distributions, buttons, axes)

    with torch.no_grad():
        bias = model.actor.bias.reshape(2, 20)
        bias[:, 0].add_(1.0)
        bias[:, 14].sub_(1.0)

    after = sample_log_prob(policy_distribution(model), buttons, axes)
    assert not torch.equal(after, before)


def test_noncausal_controller_feedback_cannot_affect_future_policy_state():
    """Marginalized controls must not return through recurrent observations."""
    torch.manual_seed(17)
    model = GamepadPolicy(dim=16, chunk_length=2)
    board = torch.zeros(1, 9, 20, 31)
    board[:, 2, 10, 15] = 1
    report = torch.zeros(1, 20)
    timing = torch.zeros(1, 3)
    context = {
        "history": torch.zeros(1, 3, 20),
        "history_mask": torch.ones(1, 3),
        "previous_chunk": torch.zeros(1, 2, 20),
        "previous_age": torch.zeros(1, 1),
    }
    before, before_value, before_hidden = model.distribution(
        board, report, timing, **context
    )

    nuisance = list(range(4, 14)) + list(range(16, 20))
    report[:, nuisance] = 0.75
    context["history"][:, :, nuisance] = 0.5
    context["previous_chunk"][:, :, nuisance] = -0.25
    after, after_value, after_hidden = model.distribution(
        board, report, timing, **context
    )
    torch.testing.assert_close(after[0].logits, before[0].logits)
    torch.testing.assert_close(after[1].loc, before[1].loc)
    torch.testing.assert_close(after_value, before_value)
    torch.testing.assert_close(after_hidden, before_hidden)

    # The same encoder must still react to feedback that can steer Snake.
    report[:, 0] = 1
    causal, _, causal_hidden = model.distribution(board, report, timing, **context)
    assert not torch.equal(causal[0].logits, before[0].logits)
    assert not torch.equal(causal_hidden, before_hidden)

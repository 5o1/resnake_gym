"""Probability contract for optional correlated raw controller chunks."""

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.chunk_normal import ChunkNormal  # noqa: E402


def test_ar1_density_entropy_and_gradient_match_multivariate_normal():
    torch.manual_seed(9)
    loc = torch.randn(2, 4, 3, requires_grad=True)
    scale = torch.full((4, 3), 0.7)
    rho = 0.8
    distribution = ChunkNormal(loc, scale, rho)
    indices = torch.arange(4)
    covariance = 0.7**2 * rho ** (indices[:, None] - indices[None, :]).abs()
    reference = torch.distributions.MultivariateNormal(
        loc.transpose(1, 2), covariance_matrix=covariance
    )

    sample = distribution.sample().detach()
    torch.testing.assert_close(
        distribution.log_prob(sample).sum((1, 2)),
        reference.log_prob(sample.transpose(1, 2)).sum(1),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        distribution.entropy().sum((1, 2)), reference.entropy().sum(1)
    )
    distribution.log_prob(sample).sum().backward()
    assert torch.isfinite(loc.grad).all()

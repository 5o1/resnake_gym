import pytest

torch = pytest.importorskip("torch")

from resnake_gym.chunk_normal import ChunkNormal  # noqa: E402


def test_joint_density_and_entropy_match_multivariate_normal():
    loc = torch.randn(2, 8, 3, requires_grad=True)
    scale = torch.ones(8, 3) * 0.7
    rho = 0.8
    distribution = ChunkNormal(loc, scale, rho)
    indices = torch.arange(8)
    covariance = 0.7**2 * rho ** (indices[:, None] - indices[None, :]).abs()
    reference = torch.distributions.MultivariateNormal(
        loc.transpose(1, 2), covariance_matrix=covariance
    )
    x = distribution.sample().detach()
    torch.testing.assert_close(
        distribution.log_prob(x).sum((1, 2)),
        reference.log_prob(x.transpose(1, 2)).sum(1),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        distribution.entropy().sum((1, 2)), reference.entropy().sum(1)
    )
    distribution.log_prob(x).sum().backward()
    assert torch.isfinite(loc.grad).all()


def test_correlated_ppo_update():
    from resnake_gym.gamepad_ppo import GamepadPPO, PPOConfig

    torch.set_num_threads(1)
    trainer = GamepadPPO(
        PPOConfig(dim=16, num_envs=1, rollout_steps=3, epochs=1, chunk_rho=0.8)
    )
    try:
        metrics = trainer.update()
        assert torch.isfinite(torch.tensor(metrics["loss"]))
    finally:
        trainer.close()

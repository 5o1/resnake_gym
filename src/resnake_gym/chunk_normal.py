"""Exact AR(1) Gaussian chunk distribution, independent across controls.

Temporal exploration motivation: Hollenstein et al., AAAI 2024. This uses
within-chunk AR(1), not their cross-step colored-noise implementation.
"""

import math

import torch


class ChunkNormal:
    def __init__(self, loc, scale, rho):
        if not 0 <= rho < 1:
            raise ValueError("correlation must lie in [0,1)")
        self.loc, self.scale, self.rho = loc, scale, rho
        self.innovation = math.sqrt(1 - rho**2)

    def sample(self):
        noise = torch.randn_like(self.loc)
        rows = [noise[:, 0]]
        for j in range(1, self.loc.shape[1]):
            rows.append(self.rho * rows[-1] + self.innovation * noise[:, j])
        return self.loc + self.scale * torch.stack(rows, 1)

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        innovations = torch.cat(
            (z[:, :1], (z[:, 1:] - self.rho * z[:, :-1]) / self.innovation), 1
        )
        logp = (
            -0.5 * innovations.square() - 0.5 * math.log(2 * math.pi) - self.scale.log()
        )
        correction = torch.zeros_like(logp)
        correction[:, 1:] = math.log(self.innovation)
        return logp - correction

    def entropy(self):
        result = torch.ones_like(self.loc) * (
            0.5 * math.log(2 * math.pi * math.e) + self.scale.log()
        )
        correction = torch.zeros_like(result)
        correction[:, 1:] = math.log(self.innovation)
        return result + correction

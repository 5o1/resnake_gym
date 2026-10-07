"""Tests for shared gamepad runtime code."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_runtime import (  # noqa: E402
    build_model,
    stack_observations,
)
from resnake_gym.gamepad_vtrace_contract import VTraceConfig  # noqa: E402


def test_runtime_builds_the_vtrace_policy_configuration():
    config = VTraceConfig(dim=16, chunk_length=3)
    model = build_model(config)
    assert model.chunk_length == 3


def test_stack_observations_preserves_fields_and_batches_values():
    observations = [
        {
            "scene": np.zeros((14, 2, 3), dtype=np.float32),
            "timing": np.array([0.25, 0.5], dtype=np.float32),
        },
        {
            "scene": np.ones((14, 2, 3), dtype=np.float32),
            "timing": np.array([0.75, 1.0], dtype=np.float32),
        },
    ]
    result = stack_observations(observations, "cpu")
    assert set(result) == {"scene", "timing"}
    assert result["scene"].shape == (2, 14, 2, 3)
    torch.testing.assert_close(
        result["timing"], torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    )

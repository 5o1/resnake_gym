"""Architecture-level compatibility tests for shared gamepad runtime code."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from resnake_gym import gamepad_ppo  # noqa: E402
from resnake_gym.gamepad_ppo_contract import (  # noqa: E402
    PPOConfig,
    validate_policy_checkpoint,
)
from resnake_gym.gamepad_runtime import (  # noqa: E402
    build_model,
    make_env,
    stack_observations,
)


def test_legacy_ppo_exports_are_thin_compatibility_aliases():
    assert gamepad_ppo.PPOConfig is PPOConfig
    assert gamepad_ppo.validate_policy_checkpoint is validate_policy_checkpoint
    assert gamepad_ppo.build_model is build_model
    assert gamepad_ppo.make_env is make_env
    assert gamepad_ppo.stack_observations is stack_observations


def test_runtime_supports_the_serialized_ppo_configuration():
    config = PPOConfig(dim=16, chunk_length=3)
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

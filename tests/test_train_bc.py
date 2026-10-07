"""Regression tests for vectorized oracle behavior-cloning samples."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = runpy.run_path(Path(__file__).parents[1] / "scripts/train_bc.py")
OracleBatchGenerator = _SCRIPT["OracleBatchGenerator"]


def _reference_batch(
    generator: object,
    batch_size: int,
    *,
    balanced_actions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the original scalar sampling and observation construction."""

    if balanced_actions:
        sampled_actions = generator.rng.integers(0, 3, size=batch_size)
        head_indices = np.fromiter(
            (
                generator.rng.choice(generator.indices_by_action[action])
                for action in sampled_actions
            ),
            dtype=np.int64,
            count=batch_size,
        )
    else:
        head_indices = generator.rng.integers(
            0,
            generator.capacity,
            size=batch_size,
        )

    lengths = generator.rng.integers(3, generator.capacity, size=batch_size)
    observations = np.empty(
        (batch_size, 9, generator.height, generator.width),
        dtype=np.float32,
    )
    for sample, (head_index, length) in enumerate(
        zip(head_indices, lengths, strict=True)
    ):
        food_offset = int(
            generator.rng.integers(1, generator.capacity - int(length) + 1)
        )
        observations[sample] = generator.observation(
            int(head_index),
            int(length),
            food_offset,
        )
    return observations, generator.labels[head_indices].copy()


@pytest.mark.parametrize("balanced_actions", [False, True])
def test_vectorized_batch_matches_scalar_reference(
    balanced_actions: bool,
) -> None:
    reference = OracleBatchGenerator(6, 6, seed=2026)
    vectorized = OracleBatchGenerator(6, 6, seed=2026)

    for batch_size in (1, 37):
        expected = _reference_batch(
            reference,
            batch_size,
            balanced_actions=balanced_actions,
        )
        actual = vectorized.batch(
            batch_size,
            balanced_actions=balanced_actions,
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert vectorized.rng.bit_generator.state == reference.rng.bit_generator.state

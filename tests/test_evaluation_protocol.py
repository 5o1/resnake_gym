"""Regression tests for the hashed gamepad evaluation contract."""

from resnake_gym.evaluation_protocol import gamepad_environment_protocol_fields
from resnake_gym.gamepad_runtime import make_env
from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.preview.evaluation import make_preview_env


def test_environment_protocol_records_the_effective_initial_snake_length():
    config = VTraceConfig(width=12, height=8, initial_length=5)
    protocol = gamepad_environment_protocol_fields(
        config,
        random_rotation=False,
    )
    env = make_env(config)
    try:
        effective_initial_length = env.env.kernel.initial_length
    finally:
        env.close()

    assert protocol["initial_snake_length"] == effective_initial_length == 5


def test_preview_environment_uses_the_protocol_initial_length():
    config = VTraceConfig(width=12, height=8, initial_length=5)
    env = make_preview_env(config, lambda *_args: None)
    try:
        assert env.env.kernel.initial_length == 5
    finally:
        env.close()

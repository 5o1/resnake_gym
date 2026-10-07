"""Process-output matching must be independent of caller path spelling."""

from resnake_gym.process_status import _output_directories


def test_relative_and_equals_output_arguments_resolve_against_process_cwd(tmp_path):
    cwd = tmp_path / "worker"
    expected = cwd / "runs" / "trial"

    assert _output_directories(
        ["python", "train_gamepad_ppo.py", "--output", "runs/trial"], cwd
    ) == [expected]
    assert _output_directories(
        ["python", "train_gamepad_vtrace.py", "--output=runs/trial"], cwd
    ) == [expected]

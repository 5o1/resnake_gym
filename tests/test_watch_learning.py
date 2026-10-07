"""Checkpoint-evaluation cache validation without starting subprocesses."""

import importlib.util
import json
import time
from pathlib import Path

from resnake_gym.evaluation_protocol import (
    frozen_gamepad_evaluation_request,
    protocol_sha256,
)


def module():
    path = Path(__file__).parents[1] / "scripts/watch_learning.py"
    spec = importlib.util.spec_from_file_location("watch_learning", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _evaluation(watch, policy, *, episodes=2, seed=610000, stochastic=False):
    request = frozen_gamepad_evaluation_request(
        sizes=["31x20"],
        episodes_per_size=episodes,
        base_seed=seed,
        stochastic=stochastic,
    )
    protocol = {**request, "version": "frozen-gamepad-eval-v1", "fixture": True}
    return {
        "checkpoint": str(policy),
        "checkpoint_sha256": watch.sha256_file(policy),
        "evaluation_request": request,
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": protocol_sha256(protocol),
        "deterministic_policy": not stochastic,
        "episodes": [{"seed": seed + index} for index in range(episodes)],
    }


def test_validation_cache_requires_exact_request_protocol_and_checkpoint(tmp_path):
    watch = module()
    policy = tmp_path / "policy-000010.pt"
    policy.write_bytes(b"first immutable policy")
    output = tmp_path / "validation-000010.json"
    output.write_text(json.dumps(_evaluation(watch, policy)))

    assert watch.valid_evaluation(output, policy, 2, 610000, False)
    assert not watch.valid_evaluation(output, policy, 2, 610001, False)
    assert not watch.valid_evaluation(output, policy, 2, 610000, True)

    policy.write_bytes(b"replacement policy")
    assert not watch.valid_evaluation(output, policy, 2, 610000, False)


def test_validation_cache_treats_arbitrary_json_or_missing_policy_as_invalid(tmp_path):
    watch = module()
    policy = tmp_path / "policy-000010.pt"
    policy.write_bytes(b"policy")
    output = tmp_path / "validation-000010.json"
    output.write_text("[]")

    assert not watch.valid_evaluation(output, policy, 1, 610000, False)
    policy.unlink()
    assert not watch.valid_evaluation(output, policy, 1, 610000, False)


def test_watcher_preserves_interval_final_policy_and_subprocess_contract(tmp_path):
    from resnake_gym.analysis.learning_watch import (
        LearningWatchOptions,
        watch_learning,
    )

    watch = module()
    interval_policy = tmp_path / "policy-000050.pt"
    final_policy = tmp_path / "policy-000051.pt"
    interval_policy.write_bytes(b"interval")
    final_policy.write_bytes(b"final")
    alive = iter((True, False))
    sleeps = []
    commands = []
    emitted = []

    def run_command(command, *, check, env):
        assert check is True
        assert env["OMP_NUM_THREADS"] == "2"
        assert env["MKL_NUM_THREADS"] == "2"
        policy = Path(command[2])
        output = Path(command[command.index("--output") + 1])
        episodes = int(command[command.index("--episodes") + 1])
        seed = int(command[command.index("--seed") + 1])
        stochastic = "--stochastic" in command
        assert command[command.index("--sizes") + 1] == "31x20"
        commands.append((policy.name, output.name, stochastic))
        payload = _evaluation(
            watch,
            policy,
            episodes=episodes,
            seed=seed,
            stochastic=stochastic,
        )
        for episode in payload["episodes"]:
            episode["won"] = False
        output.write_text(json.dumps(payload))

    watch_learning(
        LearningWatchOptions(
            run=tmp_path,
            evaluator=tmp_path / "evaluate_gamepad_ppo.py",
            pid=123,
            every=50,
            episodes=2,
            device="cpu",
            seed=620000,
        ),
        training_alive_fn=lambda _pid, _run: next(alive),
        now_fn=lambda: time.time() + 20,
        sleep_fn=sleeps.append,
        run_command_fn=run_command,
        emit_fn=emitted.append,
    )

    assert commands == [
        ("policy-000050.pt", "validation-000050.json", False),
        ("policy-000050.pt", "validation-000050-stochastic.json", True),
        ("policy-000051.pt", "validation-000051.json", False),
        ("policy-000051.pt", "validation-000051-stochastic.json", True),
    ]
    assert sleeps == [30]
    assert emitted[-1] == {
        "status": "training_process_exited",
        "run": str(tmp_path.resolve()),
    }

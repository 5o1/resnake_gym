import time

import numpy as np

from resnake_gym import gamepad
from resnake_gym.async_solvability import (
    CertificateWorker,
    _solve,
    compatible,
    occupancy_envelope,
)
from resnake_gym.certificates import rectangular_cycles, verify_cycle
from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv


class ControlledWorker:
    """Explicit delivery control for boundary tests; not production transport."""

    def __init__(self):
        self.jobs, self.results = [], []

    def submit(self, job):
        self.jobs.append(job)
        return True

    def complete(self):
        result = _solve(self.jobs[-1])
        result["result_received_ns"] = time.monotonic_ns()
        self.results.append(result)
        return result

    def poll(self):
        results, self.results = self.results, []
        return results

    def close(self):
        pass


def environment():
    env = PerturbedGamepadEnv(
        solver_mode="async", perturbation_interval=None, notice_ticks=2, logic_fps=1
    )
    env._worker = ControlledWorker()
    env.reset(seed=1, options={"food": (1, 0)})
    return env


def step(env):
    env.set_committed_reports(np.zeros((env.protected_ticks, 20), np.float32))
    return env.step(gamepad.neutral())[-1]


def test_ready_witness_is_rechecked_without_any_synchronous_search(monkeypatch):
    env = environment()
    env.reset(seed=1, options={"food": (10, 0)})
    # Use the constant-time rectangular template here.  An obstacle proposal
    # may consume its whole search budget under instrumentation, which tests
    # solver throughput rather than the activation-time recheck below.
    env.request_rotation(1)
    env._worker.complete()

    def forbidden(*args, **kwargs):
        raise AssertionError("activation must never invoke search")

    monkeypatch.setattr(
        "resnake_gym.envs.perturbed_gamepad.verify_solvability", forbidden
    )
    step(env)
    step(env)
    event = step(env)["perturbation_events"][0]
    assert event["status"] == "applied"
    validation = event["validation"]
    assert (
        event["submitted_ns"]
        <= validation["solver_started_ns"]
        <= validation["solver_completed_ns"]
        <= validation["result_received_ns"]
        <= validation["revalidated_ns"]
    )
    assert "witnesses" not in str(event) and 'cycle": [' not in str(event)
    np.testing.assert_array_equal(env.last_report, gamepad.neutral())


def test_body_compatibility_failure_skips_without_research(monkeypatch):
    env = environment()
    env.request_rotation(1)
    env._worker.complete()
    monkeypatch.setattr(
        "resnake_gym.envs.perturbed_gamepad.compatible", lambda *args: False
    )
    step(env)
    step(env)
    event = step(env)["perturbation_events"][0]
    assert event["reason"] == "async_certificate_invalidated"
    assert event["validation"]["status"] == "unknown"


def test_not_ready_skips_at_exact_tick_without_waiting():
    env = environment()
    env.request_obstacles(((0, 2), (1, 2)))
    step(env)
    step(env)
    event = step(env)["perturbation_events"][0]
    assert event["tick"] == event["activate_tick"] == 2
    assert event["reason"] == "async_not_ready"
    assert not env.kernel.obstacle_cells


def test_late_result_after_reset_is_discarded():
    env = environment()
    env.request_rotation(1)
    old = env._worker.complete()
    env.reset(seed=1)
    env.request_rotation(1)
    step(env)
    assert old["request_id"] != env.pending["request_id"]
    assert env._async_result is None and env._stale_results == 1


def test_world_change_invalidates_even_completed_certificate():
    env = environment()
    env.request_rotation(1)
    env._worker.complete()
    env._map_revision += 1
    step(env)
    step(env)
    event = step(env)["perturbation_events"][0]
    assert event["reason"] == "async_world_changed"
    assert event["status"] == "rejected"


def test_expired_completed_result_is_rejected():
    env = environment()
    env.request_rotation(1)
    result = env._worker.complete()
    result["solver_completed_ns"] = result["solve_deadline_ns"] + 1
    step(env)
    step(env)
    assert step(env)["perturbation_events"][0]["reason"] == "async_solver_late"


def test_new_food_occupying_candidate_is_rechecked():
    env = environment()
    env.request_obstacles(((0, 2), (0, 3)))
    env._worker.complete()
    step(env)
    step(env)
    env.kernel.relocate_food((0, 2))
    event = step(env)["perturbation_events"][0]
    assert event["reason"] == "occupied_at_activation"


def test_envelope_keeps_old_tail_and_bounds_all_reachable_heads():
    body = ((3, 3), (2, 3), (1, 3))
    envelope = occupancy_envelope(8, 8, body, 2)
    assert set(body) <= envelope
    assert (5, 3) in envelope and (5, 4) not in envelope


def test_fast_compatibility_matches_full_certificate_check():
    # Tiny exhaustive certificate check, never a policy rollout.
    for cycle in rectangular_cycles(4, 4):
        index = {v: i for i, v in enumerate(cycle)}
        for head in cycle:
            for neck in cycle:
                if abs(head[0] - neck[0]) + abs(head[1] - neck[1]) != 1:
                    continue
                for tail in cycle:
                    if (
                        tail == head
                        or abs(tail[0] - neck[0]) + abs(tail[1] - neck[1]) != 1
                    ):
                        continue
                    for heading in range(4):
                        body = (head, neck, tail)
                        assert compatible(index, cycle, body, heading) == verify_cycle(
                            cycle, 4, 4, set(), body, heading
                        )


def test_busy_queue_skips_submission_without_waiting():
    env = environment()
    env._worker.submit = lambda job: False
    env.request_rotation(1)
    step(env)
    step(env)
    event = step(env)["perturbation_events"][0]
    assert event["reason"] == "async_submission_unavailable"
    assert not event["submitted"]


def test_real_process_returns_search_witness_and_shuts_down():
    worker = CertificateWorker()
    started = time.monotonic_ns()
    job = {
        "request_id": ("test", 1),
        "world": ("test", 0),
        "submitted_ns": started,
        "solve_deadline_ns": started + 4_000_000_000,
        "budget": {"time_limit_ms": 2000},
        "state": (31, 20, ((1, 1), (4, 3)), ((15, 10), (14, 10), (13, 10)), 1, (10, 0)),
    }
    try:
        assert worker.submit(job)
        result = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            results = worker.poll()
            if results:
                result = results[0]
                break
            time.sleep(0.01)  # Test harness only, never environment.step.
        assert result and result["witnesses"]
        assert result["snapshot_verdict"]["reason"] == "searched_cycle"
        assert worker.process.pid is not None
    finally:
        worker.close()
    assert not worker.process.is_alive()

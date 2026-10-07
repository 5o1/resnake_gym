import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from resnake_gym import gamepad
from resnake_gym.certificates import certificate, obstacle_candidates, verify_cycle
from resnake_gym.envs.perturbed_gamepad import (
    OBSTACLE_SIZE_FILTER,
    PerturbedGamepadEnv,
    _ObstacleKernel,
)
from resnake_gym.wrappers.gamepad_async import AsyncGamepad


def advance(env, report=None):
    report = gamepad.neutral() if report is None else report
    env.set_committed_reports(np.tile(report, (env.protected_ticks, 1)))
    return env.step(report)


def environment(**kwargs):
    env = PerturbedGamepadEnv(perturbation_interval=None, notice_ticks=1, **kwargs)
    env.reset(seed=1, options={"food": (1, 1)})
    return env


def test_perturbed_environment_constructs_only_its_selected_kernel(monkeypatch):
    constructions = []

    class CountingKernel(_ObstacleKernel):
        def __init__(self, **kwargs):
            constructions.append(kwargs.copy())
            super().__init__(**kwargs)

    monkeypatch.setattr(PerturbedGamepadEnv, "kernel_type", CountingKernel)
    env = PerturbedGamepadEnv(perturbation_interval=None)

    assert isinstance(env.kernel, CountingKernel)
    assert len(constructions) == 1


def test_gym_checker():
    check_env(PerturbedGamepadEnv(), skip_render_check=True)


def test_random_rotation_is_disabled_by_default():
    env = PerturbedGamepadEnv(
        perturbation_interval=1, notice_ticks=1, obstacle_probability=0
    )
    env.reset(seed=1, options={"food": (1, 1)})
    env._automatic_proposal()
    event = env.pending if env.pending is not None else env.audit_events[-1]
    assert event["kind"] == "obstacle"
    assert env._info({})["random_rotation"] is False


def test_automatic_proposals_choose_a_certificate_eligible_size_up_front():
    env = environment()
    env.kernel.set_obstacles({(0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (10, 0)})
    proposals = []
    env._request = proposals.append
    for _ in range(100):
        env._automatic_proposal()
    assert proposals
    assert all(proposal["proposal_size_domain"] == (2, 4, 6) for proposal in proposals)
    assert all(
        proposal["proposal_size"] == len(proposal["cells"]) for proposal in proposals
    )
    assert all(
        (31 * 20 - 6 - proposal["proposal_size"]) % 2 == 0 for proposal in proposals
    )
    assert env._info({})["obstacle_size_filter"] == OBSTACLE_SIZE_FILTER


def test_automatic_proposal_records_an_empty_size_domain():
    env = environment(obstacle_batch_size=(1, 1))
    env.kernel.set_obstacles({(0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (10, 0)})
    env._automatic_proposal()
    event = env.audit_events[-1]
    assert event["reason"] == "proposal_size_domain_empty"
    assert event["proposal_size_domain"] == ()
    assert env.pending is None


def test_rotation_swaps_rectangle_without_rotating_controls():
    env = environment()
    env.request_rotation(1)
    assert env.scene()[12].max() == pytest.approx(1 / 3)
    advance(env)
    scene, _, _, _, info = advance(env)
    assert any(e["status"] == "applied" for e in info["perturbation_events"])
    assert (info["width"], info["height"]) == (20, 31)
    assert scene.shape == (14, 31, 31)
    assert scene[10].sum() == 620
    x, y = env.state.snake[0]
    right = gamepad.neutral()
    right[3] = 1
    advance(env, right)
    assert env.state.snake[0] == (x + 1, y)
    env.reset(seed=1)
    assert (env.state.width, env.state.height) == (31, 20)


def test_obstacle_activation_has_dynamic_certificate_and_no_route_in_info():
    env = environment()
    choices = obstacle_candidates(31, 20, set(), env.state.snake, env.state.food)
    assert choices
    accepted = False
    for cells in choices:
        env = environment()
        env.request_obstacles(cells)
        assert env.scene()[11].sum() == 2
        advance(env)
        scene, _, _, _, info = advance(env)
        if env.kernel.obstacle_cells:
            accepted = True
            assert scene[9].sum() == 2
            assert info["perturbation_events"][0]["reason"].startswith("certified_")
            assert "cycle" not in info and "route" not in info
            assert not env.kernel.obstacle_cells & env.state.occupied
            break
    assert accepted, "generator must not claim success by rejecting every candidate"


def test_obstacles_are_fatal_and_food_never_spawns_on_them():
    kernel = _ObstacleKernel(width=31, height=20)
    kernel.reset(seed=1)
    x, y = kernel.head
    kernel.add_obstacles(((x + 1, y),))
    for seed in range(100):
        kernel.reset(seed=seed)
        assert kernel.food not in kernel.obstacle_cells
    _, _, done, _, info = kernel.step(1)
    assert done and info["termination_reason"] == "obstacle_collision"


def test_missing_prefix_fails_closed():
    env = environment()
    env.request_rotation(1)
    env.step(gamepad.neutral())
    _, _, _, _, info = env.step(gamepad.neutral())
    assert info["perturbation_events"][0]["reason"] == "missing_committed_prefix"
    assert env.state.width == 31


def test_new_obstacle_on_body_rejected_at_activation():
    env = environment()
    env.request_obstacles([(16, 10), (17, 10)])
    advance(env)
    _, _, _, _, info = advance(env)
    assert info["perturbation_events"][0]["reason"] == "occupied_at_activation"
    assert not env.kernel.obstacle_cells


def test_known_unavoidable_prefix_is_rejected_not_corrected():
    env = environment(protected_ticks=30)
    env.request_rotation(1)
    advance(env)
    _, _, _, _, info = advance(env)
    assert info["perturbation_events"][0]["reason"] == "committed_prefix_collision"
    assert env.state.direction == 1


def test_certificate_fills_with_adversarial_food_in_validator_test_only():
    # This solver is only a certificate regression test, never rollout data.
    kernel = _ObstacleKernel(width=6, height=4)
    kernel.reset(seed=1)
    cycle = certificate(6, 4, set(), kernel.snake, kernel.direction)
    assert cycle and verify_cycle(cycle, 6, 4, set(), kernel.snake, kernel.direction)
    place_food = True
    for _ in range(24 * 24):
        index = cycle.index(kernel.head)
        free = [cell for cell in cycle if cell not in kernel.state.occupied]
        if not free:
            break
        if place_food:
            kernel.relocate_food(
                max(free, key=lambda cell: (cycle.index(cell) - index) % len(cycle))
            )
        target = cycle[(index + 1) % len(cycle)]
        place_food = target == kernel.food
        dx, dy = target[0] - kernel.head[0], target[1] - kernel.head[1]
        direction = ((0, -1), (1, 0), (0, 1), (-1, 0)).index((dx, dy))
        _, _, done, _, info = kernel.step(direction)
        if done:
            assert info["won"]
            return
    pytest.fail("verified certificate did not win")


def test_async_wrapper_supplies_committed_prefix():
    env = PerturbedGamepadEnv(
        perturbation_interval=1,
        notice_ticks=1,
        obstacle_probability=0,
        random_rotation=True,
    )
    wrapped = AsyncGamepad(env, decision_ticks=3, command_delay=0)
    wrapped.reset(seed=1)
    obs, *_ = wrapped.step(np.zeros((8, 20), np.float32))
    assert obs["board"].shape == (14, 31, 31)
    assert any(event["status"] == "applied" for event in env.audit_events)


def test_registered_perturbation_environment():
    import gymnasium as gym

    from resnake_gym import PERTURBED_ENV_ID

    env = gym.make(PERTURBED_ENV_ID, stick_deadzone=0.3)
    obs, info = env.reset(seed=3)
    assert obs.shape == (14, 31, 31)
    assert info["scene_version"] == "scene-grid-v2"
    env.close()


def test_general_separated_obstacles_are_accepted_without_changing_reports():
    env = environment(solver_budget={"time_limit_ms": None})
    cells = ((1, 2), (4, 4))
    env.request_obstacles(cells)
    advance(env)
    _, _, _, _, info = advance(env)
    event = info["perturbation_events"][0]
    assert event["status"] == "applied"
    assert event["validation"]["reason"] == "searched_cycle"
    assert env.kernel.obstacle_cells == set(cells)
    np.testing.assert_array_equal(env.last_report, gamepad.neutral())


def test_budget_unknown_is_logged_and_does_not_insert_obstacles():
    env = environment(solver_budget={"time_limit_ms": 0})
    env.request_obstacles(((1, 2), (4, 4)))
    advance(env)
    _, _, _, _, info = advance(env)
    event = info["perturbation_events"][0]
    assert event["status"] == "rejected"
    assert event["validation"]["status"] == "unknown"
    assert not env.kernel.obstacle_cells

import itertools

import numpy as np
import pytest

from resnake_gym.certificates import certificate, verify_cycle
from resnake_gym.solvability import (
    SolverBudget,
    _cycle_search,
    _exact_game,
    _successors,
    automatic_obstacle_size_choices,
    sample_obstacles,
    verify_solvability,
)

UNLIMITED_TIME = SolverBudget(cycle_nodes=5000, state_limit=50000, time_limit_ms=None)


def test_no_cycle_does_not_mean_unsolvable_and_food_matters():
    args = (4, 1, set(), ((1, 0), (0, 0)), 1)
    winning = verify_solvability(*args, (2, 0), budget=UNLIMITED_TIME)
    losing = verify_solvability(*args, (3, 0), budget=UNLIMITED_TIME)
    assert winning.status == "solvable"
    assert losing.status == "unsolvable"
    assert winning.reason == losing.reason == "exact_reachability_game"


def test_odd_single_obstacle_can_be_proved_solvable():
    result = verify_solvability(
        3, 2, {(2, 1)}, ((1, 1), (0, 1), (0, 0)), 1, (1, 0), budget=UNLIMITED_TIME
    )
    assert result.status == "solvable" and result.reason == "exact_reachability_game"


def test_budgets_are_unknown_not_unsolvable():
    args = (4, 1, set(), ((1, 0), (0, 0)), 1, (2, 0))
    for budget in (
        SolverBudget(state_limit=0),
        SolverBudget(exact_cell_limit=0),
        SolverBudget(time_limit_ms=0),
    ):
        assert verify_solvability(*args, budget=budget).status == "unknown"


def test_graph_necessary_conditions():
    separated = verify_solvability(4, 2, {(2, 0), (2, 1)}, ((0, 0),), 1, (1, 0))
    assert separated.reason == "disconnected_free_graph"
    imbalance = verify_solvability(4, 4, {(1, 1), (2, 2)}, ((0, 0),), 1, (1, 0))
    assert imbalance.reason == "hamilton_path_color_imbalance"
    assert imbalance.status == "unsolvable"


def test_full_size_separated_cells_outside_old_template_are_certified():
    # Full 31x20 board. Two NON-adjacent holes, not a removed U bend.
    snake = ((15, 10), (14, 10), (13, 10))
    obstacles = {(1, 1), (4, 3)}
    assert certificate(31, 20, obstacles, snake, 1) is None
    verdict = verify_solvability(
        31, 20, obstacles, snake, 1, (1, 0), budget=UNLIMITED_TIME
    )
    assert verdict.status == "solvable" and verdict.reason == "searched_cycle"
    assert verdict.proof_sha256 and "cycle" not in verdict.to_info()


def test_cycle_csp_matches_exhaustive_small_graph_cycles():
    # Tiny boards ONLY for exhaustive mathematical verification, not training.
    grid = {(x, y) for y in range(2) for x in range(3)}
    for removed_count in range(3):
        for removed in itertools.combinations(sorted(grid - {(0, 0)}), removed_count):
            free = grid - set(removed)
            neighbors = {
                v: {u for u in free if abs(u[0] - v[0]) + abs(u[1] - v[1]) == 1}
                for v in free
            }
            expected = any(
                verify_cycle(((0, 0),) + order, 3, 2, removed, ((0, 0),), 1)
                for order in itertools.permutations(sorted(free - {(0, 0)}))
            )
            cycle, _, _ = _cycle_search(
                free, neighbors, ((0, 0),), 1, 10000, lambda: False
            )
            assert bool(cycle) == expected
            if cycle:
                assert verify_cycle(cycle, 3, 2, removed, ((0, 0),), 1)


def test_exact_attractor_matches_independent_round_based_reference():
    free = {(x, y) for y in range(2) for x in range(3)}
    initial = (((0, 0),), 1, (2, 1))
    # Independent transition construction, including all four raw requests.
    graph, pending = {}, [initial]
    while pending:
        state = pending.pop()
        if state in graph:
            continue
        body, heading, food = state
        moves = []
        for requested in range(4):
            direction = heading if requested == (heading + 2) % 4 else requested
            dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
            target = (body[0][0] + dx, body[0][1] + dy)
            if target not in free or target in (body if target == food else body[:-1]):
                continue
            new = (target,) + (body if target == food else body[:-1])
            children = (
                set()
                if len(new) == len(free)
                else {(new, direction, apple) for apple in free - set(new)}
                if target == food
                else {(new, direction, food)}
            )
            moves.append(children)
            pending.extend(children - graph.keys())
        graph[state] = moves
    won = set()
    while True:
        updated = won | {
            state
            for state, moves in graph.items()
            if any(children <= won for children in moves)
        }
        if updated == won:
            break
        won = updated
    for state in sorted(graph)[:: max(1, len(graph) // 20)]:
        status, _, _ = _exact_game(state, free, 100000, lambda: False)
        assert (status == "solvable") == (state in won)


def test_successors_match_actual_game_growth_and_tail_motion():
    from resnake_gym.envs.perturbed_gamepad import _ObstacleKernel

    body, heading, food = ((1, 1), (0, 1), (0, 0)), 1, (2, 1)
    free = {(x, y) for y in range(4) for x in range(4)}
    predicted = set().union(*_successors((body, heading, food), free))
    for request in range(4):
        env = _ObstacleKernel(width=4, height=4, initial_length=2)
        env.reset(
            seed=1,
            options={"snake": body, "direction": heading, "food": food},
        )
        _, _, done, _, _ = env.step(request)
        if not done:
            assert (env.snake, env.direction, env.food) in predicted


def test_proposals_include_single_long_and_separated_obstacles_reproducibly():
    def draw():
        rng = np.random.default_rng(12)
        return [sample_obstacles(31, 20, {(15, 10)}, rng) for _ in range(100)]

    samples = draw()
    assert samples == draw()
    assert {len(cells) for cells, _ in samples} >= {1, 3, 4, 5, 6}
    assert {family for _, family in samples} == {"scatter", "line", "patch"}
    assert all((15, 10) not in cells for cells, _ in samples)


def test_automatic_size_domain_excludes_cycle_count_dead_ends():
    assert automatic_obstacle_size_choices(
        31, 20, 6, (1, 6), 32, exact_cell_limit=16
    ) == (2, 4, 6)
    assert (
        automatic_obstacle_size_choices(31, 20, 6, (1, 1), 32, exact_cell_limit=16)
        == ()
    )
    assert automatic_obstacle_size_choices(
        31, 20, 5, (1, 6), 32, exact_cell_limit=16
    ) == (1, 3, 5)
    assert automatic_obstacle_size_choices(3, 3, 0, (1, 4), 32, exact_cell_limit=0) == (
        1,
        3,
    )
    assert automatic_obstacle_size_choices(
        31, 20, 31, (1, 6), 32, exact_cell_limit=16
    ) == (1,)


def test_automatic_size_domain_preserves_sync_exact_search_cases():
    # Five free cells cannot have a cycle, but the synchronous exact game may
    # still prove a particular Snake state winnable (see the regression above).
    assert automatic_obstacle_size_choices(3, 2, 0, (1, 1), 6, exact_cell_limit=16) == (
        1,
    )
    assert (
        automatic_obstacle_size_choices(3, 2, 0, (1, 1), 6, exact_cell_limit=None) == ()
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"cycle_nodes": -1}, {"state_limit": 0.5}, {"time_limit_ms": float("nan")}],
)
def test_invalid_budgets_rejected(kwargs):
    with pytest.raises(ValueError):
        SolverBudget(**kwargs)

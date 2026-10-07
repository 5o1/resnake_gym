"""Private, budgeted all-future-foods reachability verifier.

Graph necessary conditions, Hamiltonian certificates, then finite AND/OR
reachability. No policy, demonstration, or action-repair interface is exposed.
See docs/solvability.html for quantifiers, sources and limitations.
"""

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

from resnake_gym.certificates import VECTORS, adjacent, certificate, verify_cycle


@dataclass(frozen=True)
class SolverBudget:
    cycle_nodes: int = 2000
    state_limit: int = 5000
    exact_cell_limit: int = 16
    time_limit_ms: float | None = 50.0

    def __post_init__(self):
        for value in (self.cycle_nodes, self.state_limit, self.exact_cell_limit):
            if type(value) is not int or value < 0:
                raise ValueError("solver counts must be nonnegative integers")
        if self.time_limit_ms is not None and not (
            0 <= self.time_limit_ms < float("inf")
        ):
            raise ValueError("time_limit_ms must be finite and nonnegative or None")


@dataclass(frozen=True)
class Verdict:
    status: str  # solvable / unsolvable / unknown, under the stated objective
    reason: str
    cycle_nodes: int
    states: int
    elapsed_ns: int
    proof_sha256: str | None = None
    objective: str = "eventual_fill_for_all_future_foods"
    verifier_version: str = "solvability-v1"

    def to_info(self):
        return asdict(self)


def _connected(vertices, neighbors):
    if not vertices:
        return True
    start = min(vertices)
    seen, stack = {start}, [start]
    while stack:
        for other in neighbors[stack.pop()] & vertices - seen:
            seen.add(other)
            stack.append(other)
    return len(seen) == len(vertices)


def _cycle_search(free, neighbors, snake, heading, limit, expired):
    """Degree-two / no-proper-subtour CSP, with body edges fixed.

    Standard Hamilton-cycle constraints (SCIP TSP example), implemented as a
    small bounded backtracking solver, NOT SCIP or the cited ASP implementation.
    Requiring contiguous body edges is sufficient, not necessary for game wins.
    """
    if len(free) % 2 or any(len(neighbors[v]) < 2 for v in free):
        return None, 0, "cycle_necessary_condition_failed"
    cells = sorted(free)
    ids = {v: i for i, v in enumerate(cells)}
    size = len(cells)
    allowed = [sum(1 << ids[u] for u in neighbors[v]) for v in cells]
    chosen = [0] * size

    def bits(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    for a, b in zip(snake, snake[1:], strict=False):
        a, b = ids[a], ids[b]
        chosen[a] |= 1 << b
        chosen[b] |= 1 << a
    stack = [(allowed, chosen)]
    nodes = 0
    while stack:
        if nodes >= limit or expired():
            return None, nodes, "cycle_budget_exhausted"
        possible, selected = stack.pop()
        nodes += 1
        valid, changed = True, True
        while valid and changed:
            if expired():
                return None, nodes, "cycle_budget_exhausted"
            changed = False
            for v in range(size):
                degree, available = selected[v].bit_count(), possible[v].bit_count()
                if degree > 2 or available < 2 or selected[v] & ~possible[v]:
                    valid = False
                    break
                if degree == 2:
                    for u in bits(possible[v] & ~selected[v]):
                        possible[u] &= ~(1 << v)
                        changed = True
                    possible[v] = selected[v]
                elif available == 2:
                    for u in bits(possible[v] & ~selected[v]):
                        selected[u] |= 1 << v
                        changed = True
                    selected[v] = possible[v]
            if not valid:
                break
            # Selected components have max degree two: paths or cycles.
            seen = set()
            for root in range(size):
                if root in seen:
                    continue
                component, todo = [], [root]
                seen.add(root)
                while todo:
                    v = todo.pop()
                    component.append(v)
                    for u in bits(selected[v]):
                        if u not in seen:
                            seen.add(u)
                            todo.append(u)
                ends = [v for v in component if selected[v].bit_count() < 2]
                if not ends and len(component) < size:
                    valid = False
                    break
                if len(ends) == 2 and len(component) < size:
                    a, b = ends
                    if possible[a] & (1 << b) and not selected[a] & (1 << b):
                        possible[a] &= ~(1 << b)
                        possible[b] &= ~(1 << a)
                        changed = True
        if not valid:
            continue
        branch = [v for v in range(size) if selected[v].bit_count() < 2]
        if not branch:
            # Check both orientations; current heading must remain legal.
            start = ids[snake[0]]
            for first in bits(selected[start]):
                route, previous, current = [start], start, first
                while current != start:
                    route.append(current)
                    nxt = next(bits(selected[current] & ~(1 << previous)))
                    previous, current = current, nxt
                cycle = tuple(cells[v] for v in route)
                indexes = {v: i for i, v in enumerate(cycle)}
                span = sum(
                    (indexes[a] - indexes[b]) % size
                    for a, b in zip(snake, snake[1:], strict=False)
                )
                delta = (cycle[1][0] - snake[0][0], cycle[1][1] - snake[0][1])
                if span < size and VECTORS.index(delta) != (heading + 2) % 4:
                    return cycle, nodes, "searched_cycle"
            continue
        v = min(
            branch, key=lambda i: (possible[i].bit_count(), -selected[i].bit_count(), i)
        )
        u = min(
            bits(possible[v] & ~selected[v]), key=lambda i: (possible[i].bit_count(), i)
        )
        excluded = possible.copy()
        excluded[v] &= ~(1 << u)
        excluded[u] &= ~(1 << v)
        stack.append((excluded, selected.copy()))
        included = selected.copy()
        included[v] |= 1 << u
        included[u] |= 1 << v
        stack.append((possible.copy(), included))
    return None, nodes, "body_contiguous_cycle_absent"


def _successors(state, free):
    """Each yielded set is one action's AND-set of possible food outcomes."""
    snake, heading, food = state
    for direction, (dx, dy) in enumerate(VECTORS):
        if direction == (heading + 2) % 4:
            continue  # reverse input equals straight in the kernel; duplicate
        target = (snake[0][0] + dx, snake[0][1] + dy)
        eating = target == food
        occupied = snake if eating else snake[:-1]
        if target not in free or target in occupied:
            continue
        body = (target,) + (snake if eating else snake[:-1])
        if len(body) == len(free):
            yield frozenset()  # terminal winning action
        elif eating:
            yield frozenset((body, direction, apple) for apple in free - set(body))
        else:
            yield frozenset({(body, direction, food)})


def _exact_game(initial, free, limit, expired):
    """Enumerate reachable arena; least attractor, not survival in a cycle.

    A move wins iff ALL its possible next food states win. A state wins iff
    SOME move wins. Exhaustion before closing the arena yields unknown.
    """
    queue = deque([initial])
    discovered = {initial}
    actions = []
    parents = defaultdict(list)
    while queue:
        if len(discovered) > limit or expired():
            return "unknown", "state_budget_exhausted", len(discovered)
        state = queue.popleft()
        for children in _successors(state, free):
            index = len(actions)
            actions.append([state, len(children)])
            for child in sorted(children):
                parents[child].append(index)
                if child not in discovered:
                    discovered.add(child)
                    if len(discovered) > limit:
                        return "unknown", "state_budget_exhausted", len(discovered)
                    queue.append(child)
    won = {state for state, count in actions if count == 0}
    queue = deque(won)
    while queue:
        if expired():
            return "unknown", "attractor_budget_exhausted", len(discovered)
        state = queue.popleft()
        for index in parents[state]:
            action = actions[index]
            action[1] -= 1
            if action[1] == 0 and action[0] not in won:
                won.add(action[0])
                queue.append(action[0])
    return (
        "solvable" if initial in won else "unsolvable",
        "exact_reachability_game",
        len(discovered),
    )


def verify_solvability(
    width, height, obstacles, snake, heading, food, *, budget=None, _witness=None
):
    """Return no routes: verdict is private generation metadata only.

    Unsolvable means no strategy guarantees eventual fill against ALL future
    food placements. It does not imply every particular food sequence loses.
    Cutoffs and future environmental interventions are outside this objective.
    """
    budget = budget or SolverBudget()
    began = time.monotonic_ns()
    deadline = (
        None
        if budget.time_limit_ms is None
        else began + int(budget.time_limit_ms * 1_000_000)
    )

    def expired():
        return deadline is not None and time.monotonic_ns() >= deadline

    def result(status, reason, nodes=0, states=0, proof=None):
        # Private worker transport only; never included in public verdict/info.
        if proof is not None and _witness is not None:
            _witness["cycle"] = tuple(proof)
        digest = hashlib.sha256(repr(proof).encode()).hexdigest() if proof else None
        return Verdict(
            status, reason, nodes, states, time.monotonic_ns() - began, digest
        )

    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("invalid dimensions")
    cells = {(x, y) for y in range(height) for x in range(width)}
    obstacles, snake = set(obstacles), tuple(snake)
    free = cells - obstacles
    if (
        not obstacles <= cells
        or not snake
        or len(set(snake)) != len(snake)
        or not set(snake) <= free
        or heading not in range(4)
        or not all(adjacent(a, b) for a, b in zip(snake, snake[1:], strict=False))
    ):
        raise ValueError("invalid snake or obstacle state")
    if len(snake) == len(free):
        return result("solvable", "already_filled")
    if food not in free - set(snake):
        raise ValueError("food must occupy a free non-body cell")
    neighbors = {v: {(v[0] + dx, v[1] + dy) for dx, dy in VECTORS} & free for v in free}
    if not _connected(free, neighbors):
        return result("unsolvable", "disconnected_free_graph")
    black = sum((x + y) % 2 == 0 for x, y in free)
    if abs(2 * black - len(free)) > 1:
        return result("unsolvable", "hamilton_path_color_imbalance")
    if sum(len(neighbors[v]) == 1 for v in free) > 2:
        return result("unsolvable", "hamilton_path_too_many_leaves")
    if next(_successors((snake, heading, food), free), None) is None:
        return result("unsolvable", "no_legal_move")
    if expired():
        return result("unknown", "time_budget_exhausted")
    cycle = certificate(width, height, obstacles, snake, heading)
    if cycle is not None:
        return result("solvable", "template_cycle", proof=cycle)
    cycle, nodes, reason = _cycle_search(
        free, neighbors, snake, heading, budget.cycle_nodes, expired
    )
    if cycle is not None:
        if not verify_cycle(cycle, width, height, obstacles, snake, heading):
            raise AssertionError("search produced an invalid cycle certificate")
        return result("solvable", "searched_cycle", nodes, proof=cycle)
    if len(free) <= budget.exact_cell_limit and not expired():
        status, reason, states = _exact_game(
            (snake, heading, food), free, budget.state_limit, expired
        )
        return result(status, reason, nodes, states)
    return result("unknown", reason, nodes)


def automatic_obstacle_size_choices(
    width,
    height,
    obstacle_count,
    sizes,
    max_obstacle_cells,
    *,
    exact_cell_limit,
):
    """Sizes that are not doomed by the active verifier's count constraint.

    ``exact_cell_limit=None`` denotes a cycle-certificate-only verifier.  Above
    the synchronous exact-search limit, a grid Hamilton cycle can only contain
    an even number of free cells.  This is a proposal-domain filter, not a
    solvability test: geometry, colour balance and the live snake state are
    still checked once, later, without resampling.
    """
    low, high = sizes
    choices = []
    for size in range(low, high + 1):
        if obstacle_count + size > max_obstacle_cells:
            continue
        free_after = width * height - obstacle_count - size
        if free_after < 0:
            continue
        cycle_only = exact_cell_limit is None or free_after > exact_cell_limit
        if not cycle_only or free_after % 2 == 0:
            choices.append(size)
    return tuple(choices)


def sample_obstacles(width, height, forbidden, rng, sizes=(1, 6)):
    """Unconditioned proposal: scattered cells, straight line, or grown patch.

    Returns (cells, family), without checking a Hamiltonian template or parity.
    Boundary/occupancy failures remain failed attempts; no hidden resampling.
    """
    free = sorted(
        {(x, y) for y in range(height) for x in range(width)} - set(forbidden)
    )
    family = ("scatter", "line", "patch")[int(rng.integers(3))]
    size = (
        sizes[0] if sizes[0] == sizes[1] else int(rng.integers(sizes[0], sizes[1] + 1))
    )
    if len(free) < size:
        return (), family
    if family == "scatter":
        return tuple(
            free[int(i)] for i in rng.choice(len(free), size, replace=False)
        ), family
    start = free[int(rng.integers(len(free)))]
    allowed = set(free)
    if family == "line":
        dx, dy = VECTORS[int(rng.integers(4))]
        cells = tuple((start[0] + i * dx, start[1] + i * dy) for i in range(size))
        return (cells if set(cells) <= allowed else ()), family
    cells = {start}
    while len(cells) < size:
        frontier = sorted(
            {(x + dx, y + dy) for x, y in cells for dx, dy in VECTORS} & allowed - cells
        )
        if not frontier:
            return (), family
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return tuple(sorted(cells)), family

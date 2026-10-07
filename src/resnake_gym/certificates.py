"""Private level-generation certificates, never a policy or training target.

Hamiltonian sufficiency: Playing Snake on a Graph, arXiv:2506.21281.
We additionally verify the CURRENT body's cyclic ordering and initial heading.
This is a conservative sufficient condition, not a complete solvability solver.
"""

VECTORS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def rectangular_cycles(width, height):
    cycles = []
    for transpose in (False, True):
        w, h = (height, width) if transpose else (width, height)
        if h % 2:
            continue
        path = [(x, 0) for x in range(w)]
        for y in range(1, h):
            xs = range(w - 1, 0, -1) if y % 2 else range(1, w)
            path.extend((x, y) for x in xs)
        path.extend((0, y) for y in range(h - 1, 0, -1))
        if transpose:
            path = [(y, x) for x, y in path]
        for flip_x, flip_y in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            variant = tuple(
                (width - 1 - x if flip_x else x, height - 1 - y if flip_y else y)
                for x, y in path
            )
            cycles.extend((variant, tuple(reversed(variant))))
    return cycles


def geometric_cycles(width, height, obstacles):
    free = {(x, y) for y in range(height) for x in range(width)} - set(obstacles)
    for cycle in rectangular_cycles(width, height):
        cycle = tuple(cell for cell in cycle if cell in free)
        if len(cycle) >= 4 and all(
            adjacent(a, b) for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True)
        ):
            yield cycle


def verify_cycle(cycle, width, height, obstacles, snake, heading):
    """Check a sufficient all-future-foods win certificate (without time cutoff).

    Body indices must be strictly ordered backwards in less than one full lap.
    Following the cycle then never overtakes the tail; each free food is reached
    within a lap. At most N*(N-length) moves suffice to fill all non-wall cells.
    """
    cycle, snake = tuple(cycle), tuple(snake)
    free = {(x, y) for y in range(height) for x in range(width)} - set(obstacles)
    if len(cycle) < 4 or len(cycle) != len(free) or set(cycle) != free:
        return False
    if not snake or len(set(snake)) != len(snake) or not set(snake) <= free:
        return False
    if not all(
        adjacent(a, b) for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True)
    ):
        return False
    if not all(adjacent(a, b) for a, b in zip(snake, snake[1:], strict=False)):
        return False
    indices = {cell: i for i, cell in enumerate(cycle)}
    span = sum(
        (indices[a] - indices[b]) % len(cycle)
        for a, b in zip(snake, snake[1:], strict=False)
    )
    if span >= len(cycle):
        return False
    target = cycle[(indices[snake[0]] + 1) % len(cycle)]
    delta = (target[0] - snake[0][0], target[1] - snake[0][1])
    return VECTORS.index(delta) != (heading + 2) % 4


def certificate(width, height, obstacles, snake, heading):
    return next(
        (
            cycle
            for cycle in geometric_cycles(width, height, obstacles)
            if verify_cycle(cycle, width, height, obstacles, snake, heading)
        ),
        None,
    )


def obstacle_candidates(width, height, obstacles, snake, food):
    """Legacy test fixture generator; NOT used by the current environment.

    Remove a two-cell U bend only when its endpoints remain adjacent. Retained
    for old certificate regression tests, not the obstacle proposal distribution.
    """
    forbidden = set(snake) | set(obstacles) | {food}
    found = set()
    for cycle in geometric_cycles(width, height, obstacles):
        n = len(cycle)
        for i in range(n):
            pair = tuple(sorted((cycle[i], cycle[(i + 1) % n])))
            if not set(pair) & forbidden and adjacent(
                cycle[(i - 1) % n], cycle[(i + 2) % n]
            ):
                found.add(pair)
    return sorted(found)

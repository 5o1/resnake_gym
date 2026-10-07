"""Gamepad-only disturbance task; scene-grid-v2 uses 14 padded spatial planes.

Generation certificates are private: no paths, demonstrations or action filters
are supplied to the policy. See docs/gamepad-api.html and task123-design.html.
"""

import copy
import time
import uuid
from collections import deque
from dataclasses import asdict

import numpy as np
from gymnasium import spaces

import resnake_gym.gamepad as gamepad
from resnake_gym.async_solvability import (
    CertificateWorker,
    compatible,
    occupancy_envelope,
)
from resnake_gym.certificates import VECTORS
from resnake_gym.envs.gamepad_env import GamepadSnakeEnv
from resnake_gym.envs.snake_env import SnakeEnv
from resnake_gym.solvability import (
    SolverBudget,
    automatic_obstacle_size_choices,
    sample_obstacles,
    verify_solvability,
)

SCENE_VERSION = "scene-grid-v2"
OBSTACLE_SIZE_FILTER = "cycle-certificate-count-v1"
SYNC_OBSTACLE_SAMPLER = "mixed-shapes-certificate-domain-v2"
ASYNC_OBSTACLE_SAMPLER = "mixed-shapes-lookahead-certificate-domain-v2"
SCENE_CHANNELS = SnakeEnv.observation_channels + (
    "obstacles",
    "on_board",
    "pending_obstacles",
    "pending_rotation_quarters",
    "pending_countdown",
)


class _ObstacleKernel(SnakeEnv):
    def __init__(self, **kwargs):
        self._obstacles = set()
        super().__init__(**kwargs)

    @property
    def obstacle_cells(self):
        """Immutable view of currently active obstacle cells."""

        return frozenset(self._obstacles)

    @property
    def obstacles(self):
        """Compatibility alias; mutation must use the explicit obstacle API."""

        return self.obstacle_cells

    def set_obstacles(self, cells):
        """Replace obstacles after validating board and occupancy constraints."""

        cells = {tuple(cell) for cell in cells}
        state = self.state
        if any(not self.contains_cell(cell) for cell in cells):
            raise ValueError("obstacles must be inside the board")
        if cells & state.occupied or state.food in cells:
            raise ValueError("obstacles cannot overlap the snake or food")
        self._obstacles = cells

    def add_obstacles(self, cells):
        """Add validated obstacles to the current board."""

        self.set_obstacles(self._obstacles | {tuple(cell) for cell in cells})

    def add_candidate_obstacles(self, cells):
        """Stage in-bounds obstacles on a counterfactual validation clone.

        Candidate cells may overlap the snapshot state; activation performs the
        occupancy check against the newer real state and fails closed.
        """

        cells = {tuple(cell) for cell in cells}
        if any(not self.contains_cell(cell) for cell in cells):
            raise ValueError("obstacles must be inside the board")
        self._obstacles.update(cells)

    def clear_obstacles(self):
        """Remove all obstacles."""

        self._obstacles.clear()

    def restore_board(self, width, height):
        """Restore board geometry before the next episode reset."""

        self.width, self.height = width, height
        self.observation_space = spaces.Box(
            0, 1, (9, self.height, self.width), np.float32
        )
        self._body_rank = np.zeros((self.height, self.width), np.int32)
        self.clear_obstacles()

    def relocate_food(self, cell):
        if tuple(cell) in self._obstacles:
            raise ValueError("food cannot overlap an obstacle")
        super().relocate_food(cell)

    def _sample_food(self):
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self._occupied and (x, y) not in self._obstacles
        ]
        return free[int(self.np_random.integers(len(free)))] if free else None

    def _advance_one_tick(self):
        x, y = self.head
        dx, dy = VECTORS[self.direction]
        if (x + dx, y + dy) in self._obstacles:
            self._termination_reason = "obstacle_collision"
            return self.living_reward + self.death_reward, True
        reward, done = super()._advance_one_tick()
        if not done and len(self._snake) == self.width * self.height - len(
            self._obstacles
        ):
            self._food = None
            self._won = True
            self._termination_reason = "board_filled"
            return reward + self.win_reward, True
        return reward, done

    def rotate(self, quarters):
        for _ in range(quarters):
            height = self.height

            def transform(cell, height=height):
                return height - 1 - cell[1], cell[0]

            self._snake = deque(transform(cell) for cell in self._snake)
            self._occupied = set(self._snake)
            self._obstacles = {transform(cell) for cell in self._obstacles}
            self._food = transform(self._food) if self._food is not None else None
            self._direction = (self._direction + 1) % 4
            self.width, self.height = self.height, self.width
        self.observation_space = spaces.Box(
            0, 1, (9, self.height, self.width), np.float32
        )
        self._body_rank = np.zeros((self.height, self.width), dtype=np.int32)
        for index, (x, y) in enumerate(self._snake):
            self._body_rank[y, x] = len(self._snake) - index


class PerturbedGamepadEnv(GamepadSnakeEnv):
    """Budget-verified arbitrary obstacle proposals and whole-board rotations.

    Certifies eventual fill for every future food placement after the supplied
    committed prefix, NOT success within an arbitrary evaluation time limit.
    Rejected attempts are recorded. No obstacle is accepted on mere connectivity.
    """

    kernel_type = _ObstacleKernel

    def __init__(
        self,
        *,
        perturbation_interval=(20, 40),
        obstacle_probability=0.5,
        random_rotation=False,
        notice_ticks=6,
        protected_ticks=3,
        max_obstacle_cells=32,
        obstacle_batch_size=(1, 6),
        solver_budget=None,
        solver_mode="sync",
        **kwargs,
    ):
        if kwargs.get("frame_skip", 1) != 1:
            raise ValueError("disturbance task requires frame_skip=1")
        if kwargs.get("render_mode") == "human":
            raise ValueError("use rgb_array and a separate UI; no render-driven clock")
        super().__init__(**kwargs)
        self._initial_size = (self.kernel.width, self.kernel.height)
        self.canvas_size = max(self._initial_size)
        self.observation_space = spaces.Box(
            0, 1, (14, self.canvas_size, self.canvas_size), np.float32
        )
        if perturbation_interval is not None:
            pair = (
                (perturbation_interval,) * 2
                if type(perturbation_interval) is int
                else tuple(perturbation_interval)
            )
            if (
                len(pair) != 2
                or any(type(v) is not int for v in pair)
                or not 1 <= pair[0] <= pair[1]
            ):
                raise ValueError("invalid perturbation interval")
            perturbation_interval = pair
        if type(notice_ticks) is not int or notice_ticks < 1:
            raise ValueError("notice_ticks must be a positive integer")
        if type(protected_ticks) is not int or protected_ticks < 1:
            raise ValueError("protected_ticks must be a positive integer")
        if not 0 <= obstacle_probability <= 1 or max_obstacle_cells < 0:
            raise ValueError("invalid disturbance parameters")
        if type(random_rotation) is not bool:
            raise ValueError("random_rotation must be a boolean")
        self.interval = perturbation_interval
        self.obstacle_probability = obstacle_probability
        self.random_rotation = random_rotation
        self.notice_ticks, self.protected_ticks = notice_ticks, protected_ticks
        self.max_obstacle_cells = max_obstacle_cells
        sizes = tuple(obstacle_batch_size)
        if (
            len(sizes) != 2
            or any(type(v) is not int for v in sizes)
            or not 1 <= sizes[0] <= sizes[1]
        ):
            raise ValueError("obstacle_batch_size must be a positive integer interval")
        self.obstacle_batch_size = sizes
        if solver_mode not in ("sync", "async"):
            raise ValueError("solver_mode must be sync or async")
        self.solver_mode = solver_mode
        default_budget = (
            {"time_limit_ms": notice_ticks / self.logic_fps * 1000}
            if solver_mode == "async"
            else {}
        )
        self.solver_budget = SolverBudget(
            **(default_budget if solver_budget is None else solver_budget)
        )
        self._worker = None
        self._clock_id = (
            "monotonic-local:" + uuid.uuid4().hex if solver_mode == "async" else None
        )
        self._async_result = None
        self._episode_id = None
        self._map_revision = 0
        self._request_counter = 0
        self._stale_results = 0
        self._last_validation = None
        self.pending = None
        self._committed = None

    def reset(self, *, seed=None, options=None):
        self.kernel.restore_board(*self._initial_size)
        self.pending, self._committed = None, None
        self._last_validation = None
        self.audit_events = []
        self._episode_id = uuid.uuid4().hex if self.solver_mode == "async" else None
        self._map_revision = 0
        self._async_result = None
        if self.solver_mode == "async" and self._worker is None:
            self._worker = CertificateWorker()
        if seed is not None or not hasattr(self, "_perturb_rng"):
            self._perturb_rng = np.random.default_rng(
                np.random.SeedSequence(seed).spawn(1)[0]
            )
        _, info = super().reset(seed=seed, options=options)
        self.next_attempt = self._interval()
        return self.scene(), self._info(info)

    def _interval(self):
        return (
            int(self._perturb_rng.integers(self.interval[0], self.interval[1] + 1))
            if self.interval
            else float("inf")
        )

    def set_committed_reports(self, reports):
        values = np.asarray(reports, dtype=np.float32)
        if values.shape != (self.protected_ticks, 20):
            raise ValueError("committed prefix must cover protected_ticks")
        self._committed = np.stack([gamepad.digital_report(row) for row in values])

    def bind_solver_clock(self, clock_id):
        """Bind async certificate timestamps to the runner's monotonic clock."""

        if self.solver_mode != "async":
            raise ValueError("only an async solver has an external clock domain")
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError("clock_id must be a non-empty string")
        self._clock_id = clock_id

    def request_rotation(self, quarters):
        if type(quarters) is not int or quarters not in (1, 2, 3):
            raise ValueError("rotation must be 1, 2 or 3 clockwise quarter turns")
        self._request({"kind": "rotation", "quarters": quarters})

    def request_obstacles(self, cells):
        cells = tuple(tuple(cell) for cell in cells)
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("obstacle proposal must contain distinct cells")
        for cell in cells:
            if (
                len(cell) != 2
                or any(type(v) is not int for v in cell)
                or not self.kernel.contains_cell(cell)
            ):
                raise ValueError("invalid obstacle cell")
        self._request({"kind": "obstacle", "cells": cells})

    def _request(self, proposal):
        state = self.state
        if not state.episode_started or state.episode_done:
            raise ValueError("proposal requires an active episode")
        if self.pending is not None:
            raise ValueError("a proposal is already pending")
        self.pending = dict(
            proposal,
            announced_tick=state.logic_steps,
            activate_tick=state.logic_steps + self.notice_ticks,
        )
        if self.solver_mode == "async":
            self._submit_certificate()
        self.audit_events.append(dict(self.pending, status="announced"))

    def _world(self):
        state = self.state
        return (
            self._episode_id,
            self._map_revision,
            state.width,
            state.height,
            tuple(sorted(self.kernel.obstacle_cells)),
        )

    def _submit_certificate(self):
        trial = self.kernel.clone_for_simulation()
        if self.pending["kind"] == "obstacle":
            trial.add_candidate_obstacles(self.pending["cells"])
        else:
            trial.rotate(self.pending["quarters"])
        trial_state = trial.state
        self._request_counter += 1
        submitted = time.monotonic_ns()
        self._async_result = None
        self.pending.update(
            request_id=(self._episode_id, self._request_counter),
            clock_id=self._clock_id,
            snapshot_tick=self.state.logic_steps,
            submitted_ns=submitted,
            solve_deadline_ns=submitted
            + round(self.notice_ticks / self.logic_fps * 1e9),
            map_revision=self._map_revision,
        )
        job = {
            key: self.pending[key]
            for key in ("request_id", "submitted_ns", "solve_deadline_ns")
        }
        job.update(
            world=self._world(),
            budget=asdict(self.solver_budget),
            state=(
                trial_state.width,
                trial_state.height,
                tuple(trial.obstacle_cells),
                trial_state.snake,
                trial_state.direction,
                trial_state.food,
            ),
        )
        self.pending["submitted"] = self._worker.submit(job)

    def _poll_certificate(self):
        if self._worker is None:
            return
        for result in self._worker.poll():
            if self.pending is not None and result["request_id"] == self.pending.get(
                "request_id"
            ):
                self._async_result = result
            else:
                self._stale_results += 1

    def _automatic_proposal(self):
        state = self.state
        obstacles = self.kernel.obstacle_cells
        if (
            not self.random_rotation
            or self._perturb_rng.random() < self.obstacle_probability
        ):
            size_domain = automatic_obstacle_size_choices(
                state.width,
                state.height,
                len(obstacles),
                self.obstacle_batch_size,
                self.max_obstacle_cells,
                exact_cell_limit=(
                    self.solver_budget.exact_cell_limit
                    if self.solver_mode == "sync"
                    else None
                ),
            )
            if not size_domain:
                self.audit_events.append(
                    {
                        "kind": "obstacle",
                        "status": "rejected",
                        "tick": state.logic_steps,
                        "reason": "proposal_size_domain_empty",
                        "proposal_size_domain": size_domain,
                        "proposal_size_filter": OBSTACLE_SIZE_FILTER,
                    }
                )
                return
            proposal_size = size_domain[
                int(self._perturb_rng.integers(len(size_domain)))
            ]
            forbidden = obstacles | state.occupied | {state.food}
            if self.solver_mode == "async":
                forbidden |= occupancy_envelope(
                    state.width,
                    state.height,
                    state.snake,
                    self.notice_ticks + self.protected_ticks,
                )
            cells, family = sample_obstacles(
                state.width,
                state.height,
                forbidden,
                self._perturb_rng,
                (proposal_size, proposal_size),
            )
            metadata = {
                "proposal_family": family,
                "proposal_size": proposal_size,
                "proposal_size_domain": size_domain,
                "proposal_size_filter": OBSTACLE_SIZE_FILTER,
            }
            if cells:
                self._request({"kind": "obstacle", "cells": cells, **metadata})
            else:
                self.audit_events.append(
                    {
                        "kind": "obstacle",
                        "status": "rejected",
                        "tick": state.logic_steps,
                        "reason": "proposal_geometry_or_limit",
                        **metadata,
                    }
                )
        else:
            self.request_rotation(int(self._perturb_rng.integers(1, 4)))

    def _activate(self, report):
        self._last_validation = None
        proposal = self.pending
        self.pending = None
        if self.solver_mode == "async":
            result = self._async_result
            self._last_validation = {
                "status": "unknown",
                "reason": "not_revalidated",
                "mode": "async",
                "checked_ns": time.monotonic_ns(),
            }
            if not proposal["submitted"]:
                return "async_submission_unavailable", None
            if result is None:
                return "async_not_ready", None
            if result["world"] != self._world():
                return "async_world_changed", None
            if "error" in result:
                self._last_validation["worker_error"] = result["error"]
                return "async_worker_error", None
            self._last_validation.update(
                {
                    key: result[key]
                    for key in (
                        "solver_started_ns",
                        "solver_completed_ns",
                        "result_received_ns",
                        "snapshot_verdict",
                    )
                }
            )
            if result["solver_completed_ns"] > proposal["solve_deadline_ns"]:
                return "async_solver_late", None
            if not result["witnesses"]:
                return "async_no_reusable_witness", None
        if self._committed is None:
            return "missing_committed_prefix", None
        prefix = self._committed.copy()
        prefix[0] = report
        trial = self.kernel.clone_for_simulation()
        # Evaluation cutoffs are not geometric solvability constraints.
        trial.set_time_limits(max_logic_steps=None, max_episode_seconds=None)
        if proposal["kind"] == "obstacle":
            cells = set(proposal["cells"])
            trial_state = trial.state
            if cells & (
                trial_state.occupied | trial.obstacle_cells | {trial_state.food}
            ):
                return "occupied_at_activation", None
            if len(cells | trial.obstacle_cells) > self.max_obstacle_cells:
                return "obstacle_limit", None
            trial.add_obstacles(cells)
        else:
            trial.rotate(proposal["quarters"])
        perturbed = trial.clone_for_simulation()
        for control in prefix:
            direction = gamepad.direction_request(control, self.stick_deadzone)
            _, _, done, _, _ = trial.step(
                trial.direction if direction is None else direction
            )
            if done:
                if trial.state.won:
                    self._last_validation = {
                        **(self._last_validation or {}),
                        "status": "solvable",
                        "reason": "committed_prefix_win",
                        "objective": "eventual_fill_given_committed_prefix",
                    }
                    return "certified_prefix_win", perturbed
                self._last_validation = {
                    **(self._last_validation or {}),
                    "status": "unsolvable",
                    "reason": "committed_prefix_collision",
                    "objective": "eventual_fill_given_committed_prefix",
                }
                return "committed_prefix_collision", None
        if self.solver_mode == "async":
            valid = any(
                compatible(index, cycle, trial.snake, trial.direction)
                for cycle, index in result["witnesses"]
            )
            self._last_validation.update(
                status="solvable" if valid else "unknown",
                reason="current_body_verified" if valid else "certificate_invalidated",
                revalidated_ns=time.monotonic_ns(),
            )
            return (
                ("certified_async_cycle", perturbed)
                if valid
                else ("async_certificate_invalidated", None)
            )
        verdict = verify_solvability(
            trial.width,
            trial.height,
            trial.obstacle_cells,
            trial.snake,
            trial.direction,
            trial.food,
            budget=self.solver_budget,
        )
        self._last_validation = verdict.to_info()
        self._last_validation["budget"] = asdict(self.solver_budget)
        if verdict.status != "solvable":
            return verdict.status + ":" + verdict.reason, None
        return "certified_" + verdict.reason, perturbed

    def step(self, action):
        self._poll_certificate()
        report = gamepad.digital_report(action)
        events_start = len(self.audit_events)
        tick = self.state.logic_steps
        if self.pending and tick >= self.pending["activate_tick"]:
            proposal = self.pending.copy()
            reason, updated = self._activate(report)
            if updated is not None:
                # Preserve the real cutoff; certification used an unlimited clone.
                updated.set_time_limits(
                    max_logic_steps=self.kernel.max_logic_steps,
                    max_episode_seconds=self.kernel.max_episode_seconds,
                )
                self.apply_kernel_update(updated)
                self._map_revision += 1
            if self.solver_mode == "async" and self._last_validation is not None:
                self._last_validation.update(
                    reason=reason, revalidated_ns=time.monotonic_ns()
                )
            self.audit_events.append(
                dict(
                    proposal,
                    tick=tick,
                    status="applied" if updated else "rejected",
                    reason=reason,
                    score_at_event=self.state.score,
                    validation=self._last_validation,
                )
            )
        self._committed = None
        _, reward, terminated, truncated, info = super().step(report)
        if (
            not (terminated or truncated)
            and self.pending is None
            and self.state.logic_steps >= self.next_attempt
        ):
            self._automatic_proposal()
            self.next_attempt = self.state.logic_steps + self._interval()
        info = self._info(info)
        info["perturbation_events"] = copy.deepcopy(self.audit_events[events_start:])
        return self.scene(), reward, terminated, truncated, info

    def _info(self, info):
        state = self.state
        return dict(
            info,
            scene_version=SCENE_VERSION,
            obstacle_sampler=(
                ASYNC_OBSTACLE_SAMPLER
                if self.solver_mode == "async"
                else SYNC_OBSTACLE_SAMPLER
            ),
            obstacle_size_filter=OBSTACLE_SIZE_FILTER,
            obstacle_batch_size=self.obstacle_batch_size,
            solver_budget=asdict(self.solver_budget),
            solver_mode=self.solver_mode,
            random_rotation=self.random_rotation,
            solver_clock_id=self._clock_id,
            stale_solver_results=self._stale_results,
            width=state.width,
            height=state.height,
            obstacle_count=len(self.kernel.obstacle_cells),
        )

    def close(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        super().close()

    def scene(self):
        scene = np.zeros(self.observation_space.shape, dtype=np.float32)
        state = self.state
        h, w = state.height, state.width
        scene[:9, :h, :w] = self.kernel.observation()
        scene[10, :h, :w] = 1
        for x, y in self.kernel.obstacle_cells:
            scene[9, y, x] = 1
        if self.pending:
            for x, y in self.pending.get("cells", ()):
                scene[11, y, x] = 1
            scene[12, :h, :w] = self.pending.get("quarters", 0) / 3
            scene[13, :h, :w] = (
                max(0, self.pending["activate_tick"] - state.logic_steps)
                / self.notice_ticks
            )
        return scene

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            board = [list(row) for row in self.kernel.render_ansi().splitlines()]
            for x, y in self.kernel.obstacle_cells:
                board[y][x] = "#"
            return "\n".join("".join(row) for row in board)
        frame = self.kernel.render_rgb_array()
        size = self.kernel.cell_size
        for cells, color in (
            (self.kernel.obstacle_cells, (160, 165, 175)),
            (self.pending.get("cells", ()) if self.pending else (), (225, 175, 45)),
        ):
            for x, y in cells:
                frame[y * size : (y + 1) * size, x * size : (x + 1) * size] = color
        return frame

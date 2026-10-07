"""Bounded, nonblocking certificate service. Private environment infrastructure."""

import multiprocessing as mp
import queue
import time
from contextlib import suppress
from dataclasses import replace

from resnake_gym.certificates import VECTORS
from resnake_gym.solvability import SolverBudget, verify_solvability


def occupancy_envelope(width, height, snake, steps):
    """Conservative union of possible body occupancy; assumes no board rotation."""
    hx, hy = snake[0]
    return set(snake) | {
        (x, y)
        for y in range(height)
        for x in range(width)
        if abs(x - hx) + abs(y - hy) <= steps
    }


def compatible(index, cycle, snake, heading):
    """O(body length), for a structurally verified private cycle witness."""
    if not snake or any(cell not in index for cell in snake):
        return False
    size = len(cycle)
    span = sum(
        (index[a] - index[b]) % size for a, b in zip(snake, snake[1:], strict=False)
    )
    target = cycle[(index[snake[0]] + 1) % size]
    delta = (target[0] - snake[0][0], target[1] - snake[0][1])
    return span < size and VECTORS.index(delta) != (heading + 2) % 4


def _solve(job):
    started = time.monotonic_ns()
    remaining_ms = max(0, (job["solve_deadline_ns"] - started) / 1e6)
    configured = SolverBudget(**job["budget"])
    budget = replace(
        configured,
        exact_cell_limit=0,
        time_limit_ms=min(remaining_ms, configured.time_limit_ms)
        if configured.time_limit_ms is not None
        else remaining_ms,
    )
    witness = {}
    verdict = verify_solvability(*job["state"], budget=budget, _witness=witness)
    cycle = witness.get("cycle")
    # Both orientations are available for future-body revalidation. A negative
    # snapshot verdict is not a proof about an evolved state.
    cycles = (cycle, tuple(reversed(cycle))) if cycle else ()
    witnesses = tuple((c, {v: i for i, v in enumerate(c)}) for c in cycles)
    return {
        "request_id": job["request_id"],
        "world": job["world"],
        "submitted_ns": job["submitted_ns"],
        "solver_started_ns": started,
        "solver_completed_ns": time.monotonic_ns(),
        "solve_deadline_ns": job["solve_deadline_ns"],
        "snapshot_verdict": verdict.to_info(),
        "witnesses": witnesses,
    }


def _worker(inbox, outbox, stop):
    try:
        while not stop.is_set():
            try:
                job = inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                result = _solve(job)
            except Exception as error:
                result = {
                    "request_id": job["request_id"],
                    "world": job["world"],
                    "error": type(error).__name__ + ": " + str(error),
                }
            if stop.is_set():
                break
            with suppress(queue.Full):
                outbox.put_nowait(result)
    finally:
        outbox.cancel_join_thread()


class CertificateWorker:
    """One worker and bounded queues; no waits in submit/poll. Close may join."""

    def __init__(self):
        context = mp.get_context("spawn")
        self.inbox = context.Queue(maxsize=1)
        self.outbox = context.Queue(maxsize=2)
        self.stop = context.Event()
        self.process = context.Process(
            target=_worker, args=(self.inbox, self.outbox, self.stop), daemon=True
        )
        self.process.start()
        self.closed = False

    def submit(self, job):
        if self.closed or not self.process.is_alive():
            return False
        try:
            self.inbox.put_nowait(job)
            return True
        except queue.Full:
            return False

    def poll(self):
        results = []
        for _ in range(2):
            try:
                result = self.outbox.get_nowait()
            except queue.Empty:
                break
            result["result_received_ns"] = time.monotonic_ns()
            results.append(result)
        return results

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        for channel in (self.inbox, self.outbox):
            channel.cancel_join_thread()
            channel.close()

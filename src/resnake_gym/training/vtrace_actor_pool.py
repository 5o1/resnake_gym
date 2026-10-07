"""Lifecycle management for V-trace actor processes.

The trainer owns the learner and its optimizer.  This module owns the shared
parameter snapshot, actor processes, queue, and orderly shutdown.  Keeping
those resources together makes it possible to test failure cleanup without
running the command-line entry point.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from resnake_gym.gamepad_vtrace_actor import (
    actor_worker,
    parameter_count,
    publish_parameters,
)
from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.gamepad_vtrace_credit import (
    CreditTraceAssembler,
    collect_fresh_credit_traces,
)


class VTraceActorPool:
    """Own a fixed set of actors and their shared parameter transport."""

    def __init__(
        self,
        config: VTraceConfig,
        model: torch.nn.Module,
        *,
        run_generation: int,
    ) -> None:
        self.config = config
        self.model = model
        self.run_generation = run_generation
        self._context = torch.multiprocessing.get_context("spawn")
        self._shared_parameters = torch.empty(
            parameter_count(model), dtype=torch.float32
        ).share_memory_()
        # The version is the learner optimizer-step count represented by the
        # shared vector.  Publishing does not invent a separate counter.
        self._shared_version = self._context.Value("q", -1)
        self._parameter_lock = self._context.Lock()
        self._stop_event = self._context.Event()
        self._output_queue = self._context.Queue(maxsize=config.queue_capacity)
        config_payload = asdict(config)
        self.actors = [
            self._context.Process(
                target=actor_worker,
                args=(
                    actor_id,
                    run_generation,
                    config_payload,
                    self._shared_parameters,
                    self._shared_version,
                    self._parameter_lock,
                    self._output_queue,
                    self._stop_event,
                ),
                name=f"resnake-vtrace-actor-{actor_id}",
            )
            for actor_id in range(config.actor_processes)
        ]
        self._started_actors: list[Any] = []
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self, *, policy_version: int) -> int:
        """Publish the initial learner snapshot, then start every actor."""
        if self._closed:
            raise RuntimeError("cannot start a closed V-trace actor pool")
        if self._started:
            raise RuntimeError("V-trace actor pool is already started")
        published_version = publish_parameters(
            self.model,
            self._shared_parameters,
            self._shared_version,
            self._parameter_lock,
            policy_version,
        )
        try:
            for actor in self.actors:
                actor.start()
                self._started_actors.append(actor)
        except BaseException:
            self.close()
            raise
        self._started = True
        return published_version

    def publish(self, *, policy_version: int) -> int:
        """Atomically expose the current learner parameters to actors."""
        if not self._started or self._closed:
            raise RuntimeError("V-trace actor pool is not running")
        return publish_parameters(
            self.model,
            self._shared_parameters,
            self._shared_version,
            self._parameter_lock,
            policy_version,
        )

    def collect(
        self, assembler: CreditTraceAssembler
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Collect one fresh FIFO learner batch from the running actors."""
        if not self._started or self._closed:
            raise RuntimeError("V-trace actor pool is not running")
        return collect_fresh_credit_traces(
            self._output_queue,
            self._stop_event,
            self.config,
            assembler,
            actor_processes=self.actors,
        )

    def close(self) -> None:
        """Stop actors and release queue resources; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        for actor in self._started_actors:
            actor.join(timeout=5)
        for actor in self._started_actors:
            if actor.is_alive():
                actor.terminate()
        for actor in self._started_actors:
            actor.join(timeout=5)
        self._output_queue.close()
        self._output_queue.join_thread()


__all__ = ["VTraceActorPool"]

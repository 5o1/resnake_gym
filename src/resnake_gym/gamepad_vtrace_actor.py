"""Actor processes and coherent policy-parameter publication."""

from __future__ import annotations

import queue
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.synchronize import Event, Lock
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from resnake_gym.gamepad_runtime import build_model, make_env, stack_observations
from resnake_gym.gamepad_vtrace_contract import VTraceConfig
from resnake_gym.gamepad_vtrace_fragments import (
    _append_transition,
    _copy_observation,
    _finish_fragment,
    _new_buffer,
)
from resnake_gym.models import (
    distribution_from_observation,
    policy_action_log_prob,
    reports_from_policy_action,
    sample_policy_action,
)


def _put_until_stopped(output_queue, payload, stop_event: Event) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(payload, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _publish_error(output_queue, stop_event: Event, actor_id: int) -> None:
    payload = {
        "kind": "actor_error",
        "actor_id": actor_id,
        "traceback": traceback.format_exc(),
    }
    _put_until_stopped(output_queue, payload, stop_event)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def publish_parameters(
    model: torch.nn.Module,
    shared_parameters: Tensor,
    shared_version,
    parameter_lock: Lock,
    policy_version: int,
) -> int:
    """Atomically publish one coherent, explicitly versioned parameter vector.

    The version is the learner optimizer-step count represented by ``model``.
    It must not be inferred by incrementing a separate publication counter.
    """
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError("policy_version must be a non-negative int")
    vector = parameters_to_vector(model.parameters()).detach().cpu()
    if vector.shape != shared_parameters.shape:
        raise ValueError("shared parameter vector shape mismatch")
    with parameter_lock:
        shared_parameters.copy_(vector)
        shared_version.value = policy_version
        return policy_version


def load_published_parameters(
    model: torch.nn.Module,
    shared_parameters: Tensor,
    shared_version,
    parameter_lock: Lock,
    current_version: int,
) -> int:
    """Load a coherent snapshot only when the learner version has changed."""
    if int(shared_version.value) == current_version:
        return current_version
    with parameter_lock:
        version = int(shared_version.value)
        if version == current_version:
            return current_version
        vector = shared_parameters.clone()
    vector_to_parameters(vector, model.parameters())
    return version


def _rebuild_actor_hidden(model, context, fallback_hidden):
    """Recompute one actor state after a weight sync using bounded burn-in."""
    if not context:
        return fallback_hidden.detach()
    hidden = torch.as_tensor(context[0][1], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for observation, _ in context:
            _, _, hidden = distribution_from_observation(
                model,
                stack_observations([observation], torch.device("cpu")),
                hidden,
            )
    return hidden[0].detach()


@dataclass
class _ActorRuntime:
    actor_id: int
    run_generation: int
    config: VTraceConfig
    model: torch.nn.Module
    local_version: int
    environments: list[Any]
    observations: list[dict[str, np.ndarray]]
    hidden: Tensor
    contexts: list[deque]
    episode_ids: list[int]
    episode_scores: list[int]
    fragment_sequences: list[int]
    episode_decisions: list[int]
    buffers: list[dict[str, Any]]
    vector_steps: int = 0


@dataclass(frozen=True)
class _ActorDecision:
    hidden_before: Tensor
    next_hidden: Tensor
    policy_action: Tensor
    behavior_log_prob: Tensor
    behavior_slot_log_probs: Tensor
    report_arrays: np.ndarray


def _new_actor_buffer(runtime: _ActorRuntime, env_id: int) -> dict[str, Any]:
    return _new_buffer(
        actor_id=runtime.actor_id,
        env_id=env_id,
        episode_id=runtime.episode_ids[env_id],
        initial_hidden=runtime.hidden[env_id],
        score=runtime.episode_scores[env_id],
        action_head=runtime.config.action_head,
        burn_context=runtime.contexts[env_id],
        run_generation=runtime.run_generation,
        fragment_sequence=runtime.fragment_sequences[env_id],
        decision_start=runtime.episode_decisions[env_id],
    )


def _start_actor_runtime(
    actor_id: int,
    run_generation: int,
    config_dict: dict[str, Any],
    shared_parameters: Tensor,
    shared_version,
    parameter_lock: Lock,
) -> _ActorRuntime:
    """Initialize one actor without changing its historical RNG order."""
    torch.set_num_threads(1)
    config = VTraceConfig(**config_dict)
    generation_seed = config.seed + run_generation * 10_000_019
    torch.manual_seed(generation_seed + 1_000_003 * (actor_id + 1))
    model = build_model(config).cpu().eval()
    local_version = load_published_parameters(
        model,
        shared_parameters,
        shared_version,
        parameter_lock,
        -1,
    )
    environments = [make_env(config) for _ in range(config.envs_per_actor)]
    observations = [
        environment.reset(seed=generation_seed + actor_id * 100_000 + env_id)[0]
        for env_id, environment in enumerate(environments)
    ]
    hidden = torch.zeros(config.envs_per_actor, config.dim)
    runtime = _ActorRuntime(
        actor_id=actor_id,
        run_generation=run_generation,
        config=config,
        model=model,
        local_version=local_version,
        environments=environments,
        observations=observations,
        hidden=hidden,
        contexts=[
            deque(maxlen=config.recurrent_burn_in) for _ in range(config.envs_per_actor)
        ],
        episode_ids=[0] * config.envs_per_actor,
        episode_scores=[0] * config.envs_per_actor,
        fragment_sequences=[0] * config.envs_per_actor,
        episode_decisions=[0] * config.envs_per_actor,
        buffers=[],
    )
    runtime.buffers = [
        _new_actor_buffer(runtime, env_id) for env_id in range(config.envs_per_actor)
    ]
    return runtime


def _close_actor_environments(environments: list[Any]) -> None:
    for environment in environments:
        environment.close()


def _finish_and_enqueue_fragment(
    runtime: _ActorRuntime,
    env_id: int,
    bootstrap_observation: dict[str, np.ndarray],
    *,
    won: bool,
    termination_reason: str | None,
    output_queue,
    stop_event: Event,
) -> bool:
    fragment = _finish_fragment(
        runtime.buffers[env_id],
        bootstrap_observation,
        won=won,
        termination_reason=termination_reason,
    )
    if not _put_until_stopped(output_queue, fragment, stop_event):
        return False
    runtime.fragment_sequences[env_id] += 1
    return True


def _sync_actor_parameters_if_due(
    runtime: _ActorRuntime,
    shared_parameters: Tensor,
    shared_version,
    parameter_lock: Lock,
    output_queue,
    stop_event: Event,
) -> bool:
    published_version = int(shared_version.value)
    should_sync = (
        runtime.vector_steps > 0
        and runtime.vector_steps % runtime.config.actor_sync_steps == 0
        and published_version != runtime.local_version
    )
    if not should_sync:
        return True

    # Seal every non-empty fragment under its old behavior version before the
    # model changes.  Credit traces may then cross versions without losing
    # exact behavior provenance.
    for env_id, buffer in enumerate(runtime.buffers):
        if not buffer["rewards"]:
            continue
        if not _finish_and_enqueue_fragment(
            runtime,
            env_id,
            runtime.observations[env_id],
            won=False,
            termination_reason=None,
            output_queue=output_queue,
            stop_event=stop_event,
        ):
            return False

    runtime.local_version = load_published_parameters(
        runtime.model,
        shared_parameters,
        shared_version,
        parameter_lock,
        runtime.local_version,
    )
    for env_id in range(runtime.config.envs_per_actor):
        runtime.hidden[env_id] = _rebuild_actor_hidden(
            runtime.model,
            runtime.contexts[env_id],
            runtime.hidden[env_id],
        )
        runtime.buffers[env_id] = _new_actor_buffer(runtime, env_id)
    return True


def _sample_actor_decision(runtime: _ActorRuntime) -> _ActorDecision:
    observation_batch = stack_observations(runtime.observations, torch.device("cpu"))
    with torch.no_grad():
        hidden_before = runtime.hidden.detach().clone()
        distribution, _, next_hidden = distribution_from_observation(
            runtime.model,
            observation_batch,
            runtime.hidden,
        )
        policy_action = sample_policy_action(distribution, stochastic=True)
        behavior_log_prob = policy_action_log_prob(distribution, policy_action)
        behavior_slot_log_probs = distribution.logits
        reports = reports_from_policy_action(distribution, policy_action)
    return _ActorDecision(
        hidden_before=hidden_before,
        next_hidden=next_hidden,
        policy_action=policy_action,
        behavior_log_prob=behavior_log_prob,
        behavior_slot_log_probs=behavior_slot_log_probs,
        report_arrays=reports.cpu().numpy(),
    )


def _record_actor_transition(
    runtime: _ActorRuntime,
    env_id: int,
    decision: _ActorDecision,
    transition,
) -> tuple[dict[str, np.ndarray], bool, bool, bool, dict[str, Any]]:
    next_observation, reward, terminated, truncated, info = transition
    _append_transition(
        runtime.buffers[env_id],
        observation=runtime.observations[env_id],
        policy_action=decision.policy_action[env_id],
        requested_report=decision.report_arrays[env_id],
        behavior_log_prob=float(decision.behavior_log_prob[env_id]),
        behavior_slot_log_probs=decision.behavior_slot_log_probs[env_id]
        .detach()
        .cpu()
        .numpy(),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        policy_version=runtime.local_version,
    )
    if runtime.config.recurrent_burn_in:
        runtime.contexts[env_id].append(
            (
                _copy_observation(runtime.observations[env_id]),
                decision.hidden_before[env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            )
        )
    runtime.episode_scores[env_id] = int(info["score"])
    runtime.episode_decisions[env_id] += 1
    boundary = (
        terminated
        or truncated
        or bool(info["food_events"])
        or len(runtime.buffers[env_id]["rewards"]) >= runtime.config.unroll_length
    )
    return next_observation, terminated, truncated, boundary, info


def _advance_actor_environment(
    runtime: _ActorRuntime,
    env_id: int,
    decision: _ActorDecision,
    next_observation: dict[str, np.ndarray],
    *,
    terminated: bool,
    truncated: bool,
    boundary: bool,
) -> None:
    if terminated or truncated:
        next_observation = runtime.environments[env_id].reset()[0]
        runtime.episode_ids[env_id] += 1
        runtime.episode_scores[env_id] = 0
        runtime.fragment_sequences[env_id] = 0
        runtime.episode_decisions[env_id] = 0
        runtime.hidden[env_id].zero_()
        runtime.contexts[env_id].clear()
    else:
        runtime.hidden[env_id] = decision.next_hidden[env_id]
    if boundary:
        runtime.buffers[env_id] = _new_actor_buffer(runtime, env_id)
    runtime.observations[env_id] = next_observation


def _run_actor_vector_step(
    runtime: _ActorRuntime,
    output_queue,
    stop_event: Event,
) -> bool:
    decision = _sample_actor_decision(runtime)
    transitions = [
        environment.step(report)
        for environment, report in zip(
            runtime.environments,
            decision.report_arrays,
            strict=True,
        )
    ]
    for env_id, transition in enumerate(transitions):
        next_observation, terminated, truncated, boundary, info = (
            _record_actor_transition(runtime, env_id, decision, transition)
        )
        if boundary and not _finish_and_enqueue_fragment(
            runtime,
            env_id,
            next_observation,
            won=info["won"],
            termination_reason=info["termination_reason"],
            output_queue=output_queue,
            stop_event=stop_event,
        ):
            return False
        _advance_actor_environment(
            runtime,
            env_id,
            decision,
            next_observation,
            terminated=terminated,
            truncated=truncated,
            boundary=boundary,
        )
    runtime.vector_steps += 1
    return True


def actor_worker(
    actor_id: int,
    run_generation: int,
    config_dict: dict[str, Any],
    shared_parameters: Tensor,
    shared_version,
    parameter_lock: Lock,
    output_queue,
    stop_event: Event,
) -> None:
    """CPU actor entry point.  All emitted fragments are episode-local."""
    environments = []
    try:
        runtime = _start_actor_runtime(
            actor_id,
            run_generation,
            config_dict,
            shared_parameters,
            shared_version,
            parameter_lock,
        )
        environments = runtime.environments
        while not stop_event.is_set():
            if not _sync_actor_parameters_if_due(
                runtime,
                shared_parameters,
                shared_version,
                parameter_lock,
                output_queue,
                stop_event,
            ):
                return
            if not _run_actor_vector_step(runtime, output_queue, stop_event):
                return
    except BaseException:  # actor failures must be visible to the learner
        _publish_error(output_queue, stop_event, actor_id)
        stop_event.set()
    finally:
        _close_actor_environments(environments)

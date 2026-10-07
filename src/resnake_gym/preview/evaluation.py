"""Deterministic policy evaluation for live checkpoint previews."""

import datetime as dt
import hashlib
import json

from resnake_gym.artifact_io import sha256_file as sha256_file
from resnake_gym.evaluation_protocol import (
    DETERMINISTIC_SOLVER_BUDGET,
    gamepad_environment_protocol_fields,
)
from resnake_gym.policy_checkpoint import (
    load_policy_config as _load_policy_config,
)
from resnake_gym.policy_checkpoint import load_policy_model

from .archive import SCHEMA_VERSION, EpisodeArchiveWriter, snapshot


def protocol_fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluation_protocol(config, seed):
    """Frozen causal conditions; training-only hyperparameters are excluded."""
    return {
        "protocol": "complete-checkpoint-episode-v4",
        "seed": seed,
        "action_selection": "deterministic",
        "width": config.width,
        "height": config.height,
        **gamepad_environment_protocol_fields(config, random_rotation=False),
        "solver_budget": DETERMINISTIC_SOLVER_BUDGET,
        "frame_granularity": "logic_tick",
    }


def make_preview_env(config, tick_sink):
    """Frozen evaluation env whose private copy emits every game logic tick."""
    from resnake_gym.envs.reward_gamepad import RewardGamepadEnv
    from resnake_gym.wrappers.gamepad_async import AsyncGamepad

    class CapturingRewardGamepadEnv(RewardGamepadEnv):
        preview_decision = 0

        def step(self, action):
            transition = super().step(action)
            tick_sink(self, transition[4], self.preview_decision)
            return transition

    game = CapturingRewardGamepadEnv(
        reward_gamma=config.gamma,
        shaping_scale=config.shaping_scale,
        death_cost=config.death_cost,
        width=config.width,
        height=config.height,
        initial_length=config.initial_length,
        logic_fps=config.logic_fps,
        max_logic_steps=config.max_logic_steps,
        perturbation_interval=(config.perturbation_min, config.perturbation_max),
        obstacle_probability=0.5,
        random_rotation=False,
        notice_ticks=6,
        protected_ticks=3,
        max_obstacle_cells=32,
        obstacle_batch_size=(1, 6),
        stick_deadzone=0.2,
        solver_budget=DETERMINISTIC_SOLVER_BUDGET,
    )
    return AsyncGamepad(
        game,
        chunk_length=config.chunk_length,
        decision_ticks=(config.decision_min, config.decision_max),
        observation_delay=(0, config.observation_delay_max),
        command_delay=(0, config.command_delay_max),
        frame_drop_probability=config.drop_probability,
        gamma=config.gamma,
        time_features=True,
    )


def load_policy(path, device):
    """Compatibility name for the shared validated model loader."""

    return load_policy_model(path, device)


def load_policy_config(path):
    """Compatibility name for the shared validated config loader."""

    return _load_policy_config(path)


def cache_key_for(checkpoint_sha256, seed, protocol_sha256):
    identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "evaluation_protocol_sha256": protocol_sha256,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _event_common(frame, frame_index, decision, policy_version):
    return {
        "episode_tick": frame["episode_tick"],
        "simulation_seconds": frame["simulation_seconds"],
        "decision": decision,
        "frame_index": frame_index,
        "policy_version": policy_version,
    }


def events_from_tick(frame, frame_index, decision, policy_version, tick_info):
    """Normalize environment audit records to the visible post-step frame."""
    common = _event_common(frame, frame_index, decision, policy_version)
    before, after = [], []
    for event in tick_info.get("perturbation_events", ()):
        status = event.get("status")
        if event.get("kind") != "obstacle" or status not in (
            "announced",
            "applied",
            "rejected",
        ):
            continue
        raw_tick = (
            event.get("tick", frame["episode_tick"])
            if status in ("applied", "rejected")
            else event.get("announced_tick", event.get("tick", frame["episode_tick"]))
        )
        normalized = {
            **common,
            "type": f"obstacle_{status}",
            "cells": [list(cell) for cell in event.get("cells", ())],
            "reason": event.get("reason", "pending"),
            "environment_event_tick": int(raw_tick),
        }
        if "activate_tick" in event:
            normalized["activate_game_tick"] = int(event["activate_tick"])
        # Activation is checked before movement; proposals happen after movement.
        (
            before
            if status in ("applied", "rejected") and "activate_tick" in event
            else after
        ).append(normalized)
    food = tick_info.get("food_event")
    middle = []
    if food is not None:
        middle.append(
            {
                **common,
                "type": "food_eaten",
                "score": frame["score"],
                "position": frame["snake"][0],
                "ticks_since_previous_food": int(food["ticks"]),
                "distance": int(food["distance"]),
                "free": int(food["free"]),
                "cells": int(food["cells"]),
                "efficiency": float(food["efficiency"]),
                "reward": float(food["reward"]),
            }
        )
    end = []
    reason = tick_info.get("termination_reason")
    if reason:
        end.append(
            {
                **common,
                "type": "game_end",
                "reason": reason,
                "score": frame["score"],
                "won": bool(tick_info.get("won")),
                "terminated": reason != "time_limit",
                "truncated": reason == "time_limit",
            }
        )
    return [*before, *middle, *after, *end]


def rollout_episode(
    path,
    update,
    seed,
    device,
    evaluation_config,
    site,
    checkpoint_sha256,
    protocol,
    protocol_sha256,
    chunk_frames,
    max_archive_bytes,
):
    import torch

    from resnake_gym.gamepad_runtime import stack_observations
    from resnake_gym.models import (
        distribution_from_observation,
        reports_from_policy_action,
        sample_policy_action,
    )

    payload, policy_config, model = load_policy(path, device)
    for name in ("width", "height", "chunk_length"):
        if getattr(policy_config, name) != getattr(evaluation_config, name):
            raise ValueError(
                f"checkpoint {name} is incompatible with evaluation protocol"
            )
    if evaluation_config.max_logic_steps is None:
        raise ValueError("complete preview requires a finite formal episode limit")
    cache_key = cache_key_for(checkpoint_sha256, seed, protocol_sha256)
    identifier = f"u{update:06d}-{cache_key}"
    writer = EpisodeArchiveWriter(site, identifier, chunk_frames, max_archive_bytes)
    torch.manual_seed(seed)
    events, decision = [], 0
    policy_version = int(payload.get("policy_version", update))

    def capture_tick(game, tick_info, tick_decision):
        frame = snapshot(game, tick_info, tick_decision)
        writer.add(frame)
        events.extend(
            events_from_tick(
                frame,
                writer.frame_count - 1,
                tick_decision,
                policy_version,
                tick_info,
            )
        )

    env = make_preview_env(evaluation_config, capture_tick)
    hidden = None
    observation, info = env.reset(seed=seed)
    writer.add(snapshot(env.env, info, decision))
    events.append(
        {
            "type": "game_start",
            "episode_tick": 0,
            "simulation_seconds": 0.0,
            "decision": 0,
            "frame_index": 0,
            "policy_version": policy_version,
            "seed": seed,
            "width": evaluation_config.width,
            "height": evaluation_config.height,
        }
    )
    try:
        terminated = truncated = False
        with torch.inference_mode():
            while not (terminated or truncated):
                decision += 1
                env.env.preview_decision = decision
                distribution, _, hidden = distribution_from_observation(
                    model, stack_observations([observation], device), hidden
                )
                action = sample_policy_action(distribution, stochastic=False)
                report = (
                    reports_from_policy_action(
                        distribution,
                        action,
                        continuous_buttons=policy_config.chunk_rho is not None,
                    )[0]
                    .cpu()
                    .numpy()
                )
                before = writer.frame_count
                observation, _, terminated, truncated, info = env.step(report)
                if writer.frame_count - before != int(info["ticks_advanced"]):
                    raise RuntimeError(
                        "per-tick capture did not match environment advance"
                    )
        writer.finalize()
        episode_ticks = int(info["logic_steps"])
        if writer.frame_count != episode_ticks + 1:
            raise RuntimeError("episode must contain reset plus every logic tick")
        if not events or events[-1]["type"] != "game_end":
            raise RuntimeError("complete episode is missing its terminal event")
        generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "independent_complete_checkpoint_episode",
                "id": identifier,
                "cache_key": cache_key,
                "checkpoint": path.name,
                "checkpoint_size": path.stat().st_size,
                "checkpoint_mtime_ns": path.stat().st_mtime_ns,
                "checkpoint_sha256": checkpoint_sha256,
                "source_update": update,
                "policy_version": policy_version,
                "source_logic_ticks": int(payload.get("logic_ticks", 0)),
                "source_logic_ticks_semantics": "trained",
                "source_trained_logic_ticks": int(
                    payload.get("trained_logic_ticks", payload.get("logic_ticks", 0))
                ),
                "source_trained_transitions": int(
                    payload.get("trained_transitions", payload.get("transitions", 0))
                ),
                "source_received_logic_ticks": (
                    int(payload["received_logic_ticks"])
                    if "received_logic_ticks" in payload
                    else None
                ),
                "source_received_transitions": (
                    int(payload["received_transitions"])
                    if "received_transitions" in payload
                    else None
                ),
                "action_head": policy_config.action_head,
                "action_encoding": payload.get("action_encoding"),
                "training_objective": payload.get("training_objective"),
                "checkpoint_format": payload.get("format"),
                "collection_semantics": payload.get("collection_semantics"),
                "recurrent_state": payload.get("recurrent_state"),
                "credit_trace_max_transitions": getattr(
                    policy_config, "credit_trace_max_transitions", None
                ),
                "bptt_window": getattr(policy_config, "bptt_window", None),
                "action_variables_per_decision": (
                    1
                    if policy_config.action_head == "held_dpad5"
                    else policy_config.chunk_length
                    if policy_config.action_head == "dpad5"
                    else None
                ),
                "seed": seed,
                "mode": "deterministic",
                "logic_fps": evaluation_config.logic_fps,
                "episode_ticks": episode_ticks,
                "decision_count": decision,
                "final_score": int(info["score"]),
                "termination": info["termination_reason"],
                "won": bool(info["won"]),
                "truncated": bool(truncated),
                "episode_complete": True,
                "frame_count": writer.frame_count,
                "frame_granularity": "logic_tick",
                "chunks": writer.descriptors,
                "events": events,
                "evaluation_protocol": protocol,
                "evaluation_protocol_sha256": protocol_sha256,
                "generated_at": generated_at,
                "timestamp_semantics": (
                    "generated_at is evaluation publication time; event and frame "
                    "times are simulation ticks"
                ),
                "hidden_logic_ticks": 0,
                "frame_note": "reset state plus every bottom-level game logic tick",
            },
            writer,
        )
    except Exception:
        writer.rollback()
        raise
    finally:
        env.close()

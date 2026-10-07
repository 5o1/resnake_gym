"""Evaluation tooling, never training data or a policy teacher.

Motivation: Machado et al. arXiv:1709.06009 (open-loop memorization);
Packer et al. arXiv:1810.12282 (held-out conditions). Frozen-observation,
fixed-report replay and intervention/recovery metrics below are project tests,
not claims of reproducing either paper's exact protocol.
"""

import copy
import hashlib
from dataclasses import dataclass

import numpy as np

from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv
from resnake_gym.wrappers.gamepad_async import AsyncGamepad


@dataclass
class EvaluationConfig:
    width: int = 31
    height: int = 20
    logic_fps: float = 10.0
    chunk_length: int = 8
    decision_ticks: tuple = (2, 5)
    observation_delay: tuple = (0, 3)
    command_delay: tuple = (0, 2)
    frame_drop_probability: float = 0.1
    perturbation_interval: tuple = (20, 40)
    notice_ticks: int = 6
    protected_ticks: int = 3
    max_logic_steps: int = 3000
    recovery_ticks: int = 20
    obstacle_batch_size: tuple = (1, 6)
    solver_time_limit_ms: float = 50.0
    time_features: bool = False


def make_evaluation_env(config, condition):
    if condition not in ("none", "obstacle", "rotation", "combined"):
        raise ValueError("unknown disturbance condition")
    env = PerturbedGamepadEnv(
        width=config.width,
        height=config.height,
        logic_fps=config.logic_fps,
        perturbation_interval=None
        if condition == "none"
        else config.perturbation_interval,
        obstacle_probability={
            "none": 0.0,
            "obstacle": 1.0,
            "rotation": 0.0,
            "combined": 0.5,
        }[condition],
        random_rotation=condition in ("rotation", "combined"),
        notice_ticks=config.notice_ticks,
        protected_ticks=config.protected_ticks,
        max_logic_steps=config.max_logic_steps,
        obstacle_batch_size=config.obstacle_batch_size,
        solver_budget={"time_limit_ms": config.solver_time_limit_ms},
    )
    return AsyncGamepad(
        env,
        chunk_length=config.chunk_length,
        decision_ticks=config.decision_ticks,
        observation_delay=config.observation_delay,
        command_delay=config.command_delay,
        frame_drop_probability=config.frame_drop_probability,
        time_features=config.time_features,
    )


def recovery_metrics(events, scores, final_tick, termination, horizon):
    """Do not count an episode cut short by evaluation cutoff as recovery."""
    results = []
    for event in events:
        if event["status"] != "applied":
            continue
        start = event["tick"]
        deadline = start + horizon
        death = termination in (
            "wall_collision",
            "self_collision",
            "obstacle_collision",
        )
        censored = final_tick < deadline and not death and termination != "board_filled"
        survived = not censored and (
            final_tick > deadline
            or (final_tick == deadline and not death)
            or termination == "board_filled"
        )
        ate = any(
            start < tick <= deadline and score > event["score_at_event"]
            for tick, score in scores
        )
        results.append(
            {
                "kind": event["kind"],
                "event_tick": start,
                "deadline_tick": deadline,
                "censored": censored,
                "survived": survived,
                "ate_within_window": ate,
                "other_event_in_window": any(
                    e["status"] == "applied" and start < e["tick"] <= deadline
                    for e in events
                ),
            }
        )
    return results


def evaluate_episode(policy, config, *, seed, condition, mode="closed", replay=None):
    """Policy protocol: reset(seed=...), act(observation)->numpy[L,20].

    frozen supplies the initial observation on EVERY call (all feedback frozen).
    replay uses actual donor reports indexed by absolute logic tick, never by
    number of policy calls; after donor ends, neutral is explicit fallback.
    """
    if mode not in ("closed", "frozen", "replay"):
        raise ValueError("unknown feedback ablation")
    if mode == "replay" and replay is None:
        raise ValueError("replay requires an independently collected donor trace")
    env = make_evaluation_env(config, condition)
    obs, _ = env.reset(seed=seed)
    frozen = copy.deepcopy(obs)
    policy.reset(seed=seed)
    actions, events, scores = [], [], []
    input_hash = hashlib.sha256()
    try:
        done = False
        while not done:
            if mode == "replay":
                reports = np.zeros((config.chunk_length, 20), np.float32)
                for j in range(config.chunk_length):
                    target = env.tick + j
                    if target < len(replay):
                        reports[j] = replay[target]
            else:
                visible = frozen if mode == "frozen" else obs
                input_hash.update(visible["board"].tobytes())
                reports = policy.act(copy.deepcopy(visible))
            obs, _, terminated, truncated, info = env.step(reports)
            done = terminated or truncated
            events.extend(info["perturbation_events"])
            controls = obs["history"][obs["history_mask"].astype(bool)]
            actions.extend(controls.tolist())
            scores.extend(
                (tick + 1, score)
                for tick, score in zip(
                    info["execution_ticks"], info["tick_scores"], strict=True
                )
            )
        action_array = np.asarray(actions, np.float32)
        return {
            "seed": seed,
            "condition": condition,
            "mode": mode,
            "ticks": info["logic_steps"],
            "score": info["score"],
            "won": info["won"],
            "termination": info["termination_reason"],
            "applied": {
                kind: sum(
                    e["status"] == "applied" and e["kind"] == kind for e in events
                )
                for kind in ("obstacle", "rotation")
            },
            "rejected": sum(e["status"] == "rejected" for e in events),
            "timebase": "simulation_tick",
            "validation_counts": {
                status: sum(
                    (e.get("validation") or {}).get("status") == status for e in events
                )
                for status in ("solvable", "unsolvable", "unknown")
            },
            "events": events,
            "recovery": recovery_metrics(
                events,
                scores,
                info["logic_steps"],
                info["termination_reason"],
                config.recovery_ticks,
            ),
            "action_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
            "input_sha256": input_hash.hexdigest(),
            "actual_reports": actions,
        }
    finally:
        env.close()


def paired_bootstrap_difference(closed, baseline, *, seed=0, samples=2000):
    if len(closed) != len(baseline) or not len(closed):
        raise ValueError("paired samples must have equal nonzero length")
    differences = np.asarray(closed, float) - np.asarray(baseline, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(
        differences, size=(samples, len(differences)), replace=True
    ).mean(1)
    return {
        "pairs": len(differences),
        "mean": float(differences.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def summarize(records, *, min_episodes=20, min_applied_per_kind=5):
    """Evidence gate only. 'ready' is not a paper-level capability claim."""
    comparisons = []
    for condition in ("none", "obstacle", "rotation", "combined"):
        closed = {
            r["seed"]: r
            for r in records
            if r["condition"] == condition and r["mode"] == "closed"
        }
        for mode in ("frozen", "replay"):
            baseline = {
                r["seed"]: r
                for r in records
                if r["condition"] == condition and r["mode"] == mode
            }
            seeds = sorted(closed.keys() & baseline.keys())
            if seeds:
                comparisons.append(
                    {
                        "condition": condition,
                        "baseline": mode,
                        **paired_bootstrap_difference(
                            [closed[s]["score"] for s in seeds],
                            [baseline[s]["score"] for s in seeds],
                        ),
                    }
                )
    closed_records = [row for row in records if row["mode"] == "closed"]
    counts = {
        kind: sum(row["applied"][kind] for row in closed_records)
        for kind in ("obstacle", "rotation")
    }
    reasons = []
    if len(comparisons) != 8 or any(c["pairs"] < min_episodes for c in comparisons):
        reasons.append("insufficient_paired_conditions_or_episodes")
    if any(counts[kind] < min_applied_per_kind for kind in counts):
        reasons.append("insufficient_actual_perturbations")
    perturbed = [c for c in comparisons if c["condition"] != "none"]
    if len(perturbed) != 6 or any(c["ci95"][0] <= 0 for c in perturbed):
        reasons.append("no_positive_feedback_advantage_over_both_controls")
    recovery = [
        item
        for row in closed_records
        for item in row["recovery"]
        if not item["censored"]
    ]
    if not recovery or not any(
        item["survived"] and item["ate_within_window"] for item in recovery
    ):
        reasons.append("no_observed_survival_and_food_recovery")
    return {
        "evidence_gate": "insufficient" if reasons else "ready_for_review",
        "reasons": reasons,
        "paired_score_differences": comparisons,
        "actual_perturbations": counts,
        "validation_counts_closed": {
            status: sum(
                row.get("validation_counts", {}).get(status, 0)
                for row in closed_records
            )
            for status in ("solvable", "unsolvable", "unknown")
        },
        "recovery_windows": len(recovery),
        "survived_windows": sum(item["survived"] for item in recovery),
        "food_recovery_windows": sum(item["ate_within_window"] for item in recovery),
        "note": (
            "Positive action sensitivity alone is not success; "
            "review recovery and task performance."
        ),
    }


def counterfactual_probe(policy, history, variants, seed):
    """Same policy memory/history, different last delivered scene. Diagnostic only.

    policy must implement fork() that clones recurrent state but can share frozen
    weights; no deepcopy of a multi-billion-parameter VLA is imposed by this API.
    Variant validity is the caller's responsibility, typically a rotated snapshot.
    """
    policy.reset(seed=seed)
    for observation in history:
        policy.act(copy.deepcopy(observation))
    outputs = [
        np.asarray(policy.fork().act(copy.deepcopy(obs)), np.float32)
        for obs in variants
    ]
    if len(outputs) < 2 or any(out.shape != outputs[0].shape for out in outputs):
        raise ValueError("need at least two equal-shaped action outputs")
    return {
        "mean_absolute_difference": [
            float(np.abs(out - outputs[0]).mean()) for out in outputs[1:]
        ],
        "changed": [bool(not np.array_equal(out, outputs[0])) for out in outputs[1:]],
        "interpretation": "sensitivity only; not correctness or recovery evidence",
    }

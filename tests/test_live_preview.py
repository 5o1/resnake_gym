import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest


def preview_module():
    path = Path(__file__).parents[1] / "scripts/live_preview.py"
    spec = importlib.util.spec_from_file_location("live_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(tick, *, score=0, food=(8, 4), snake=None, ended=False):
    snake = snake or [[5 + tick, 4], [4 + tick, 4], [3 + tick, 4]]
    return {
        "episode_tick": tick,
        "simulation_seconds": tick / 10,
        "decision": (tick + 1) // 2,
        "duration_ticks": 1,
        "width": 31,
        "height": 20,
        "snake": snake,
        "food": list(food) if food is not None else None,
        "obstacles": [],
        "pending_obstacles": [],
        "pending_rotation": 0,
        "direction": 1,
        "score": score,
        "ate": False,
        "terminated": ended,
        "truncated": False,
        "actual_report": [0.0] * 20,
    }


def prepared_episode(module, site, update, token, *, chunk_frames=2):
    protocol = {"protocol": "fixture-v2", "seed": 730000}
    protocol_sha = module.protocol_fingerprint(protocol)
    checkpoint_sha = (token * 64)[:64]
    cache_key = module.cache_key_for(checkpoint_sha, 730000, protocol_sha)
    identifier = f"u{update:06d}-{cache_key[:16]}"
    writer = module.EpisodeArchiveWriter(site, identifier, chunk_frames, 1_000_000)
    frames = [frame(0), frame(1), frame(2), frame(3, ended=True)]
    for value in frames:
        writer.add(value)
    writer.finalize()
    episode = {
        "schema_version": 2,
        "kind": "independent_complete_checkpoint_episode",
        "id": identifier,
        "cache_key": cache_key,
        "checkpoint": f"policy-{update:06d}.pt",
        "checkpoint_sha256": checkpoint_sha,
        "source_update": update,
        "policy_version": update + 60,
        "source_logic_ticks": update * 100,
        "seed": 730000,
        "mode": "deterministic",
        "logic_fps": 10.0,
        "episode_ticks": 3,
        "decision_count": 2,
        "final_score": 0,
        "termination": "wall_collision",
        "won": False,
        "truncated": False,
        "episode_complete": True,
        "frame_count": 4,
        "frame_granularity": "logic_tick",
        "chunks": writer.descriptors,
        "events": [
            {"type": "game_start", "episode_tick": 0, "frame_index": 0},
            {"type": "game_end", "episode_tick": 3, "frame_index": 3},
        ],
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": protocol_sha,
        "generated_at": "2026-10-06T00:00:00+00:00",
    }
    return episode, writer, frames


def decode_episode(module, site, metadata):
    result = []
    for descriptor in metadata["chunks"]:
        chunk = json.loads((site / descriptor["url"]).read_text())
        state = chunk["initial"]
        values = [state]
        for delta in chunk["deltas"]:
            state = module.apply_frame_delta(state, delta)
            values.append(state)
        assert len(values) == descriptor["frames"]
        result.extend(values)
    return result


def test_schema_v2_archive_round_trip_and_retention(tmp_path):
    module = preview_module()
    expected = None
    for update, token in ((10, "a"), (20, "b"), (30, "c")):
        episode, writer, expected = prepared_episode(module, tmp_path, update, token)
        manifest = module.publish_episode(
            tmp_path, episode, writer, retention=2, max_storage_bytes=1_000_000
        )
    assert [clip["source_update"] for clip in manifest["clips"]] == [20, 30]
    assert not list(tmp_path.glob("segment-*.json"))
    assert not list(tmp_path.glob(".*.stage"))
    metadata = json.loads((tmp_path / manifest["clips"][-1]["url"]).read_text())
    assert "frames" not in metadata
    assert decode_episode(module, tmp_path, metadata) == expected
    assert manifest["storage_bytes"] == sum(
        (tmp_path / asset).stat().st_size
        for clip in manifest["clips"]
        for asset in clip["assets"]
    )
    assert all(
        not path.name.startswith("episode-u000010")
        for path in tmp_path.glob("episode-*.json")
    )


def test_byte_cap_expires_oldest_whole_archive(tmp_path):
    module = preview_module()
    protocol_sha = None
    for update, token in ((10, "a"), (20, "b"), (30, "c")):
        episode, writer, _ = prepared_episode(module, tmp_path, update, token)
        protocol_sha = episode["evaluation_protocol_sha256"]
        module.publish_episode(
            tmp_path, episode, writer, retention=3, max_storage_bytes=1_000_000
        )
    manifest = module.read_manifest(tmp_path, 3, 1_000_000)
    newest_two = sum(clip["bytes"] for clip in manifest["clips"][-2:])
    manifest = module.prune_incompatible_segments(tmp_path, 3, protocol_sha, newest_two)
    assert [clip["source_update"] for clip in manifest["clips"]] == [20, 30]
    assert manifest["storage_bytes"] <= newest_two
    existing = {path.name for path in tmp_path.glob("episode-*.json")}
    assert existing == {asset for clip in manifest["clips"] for asset in clip["assets"]}


def test_protocol_change_expires_metadata_and_chunks(tmp_path):
    module = preview_module()
    episode, writer, _ = prepared_episode(module, tmp_path, 10, "a")
    module.publish_episode(tmp_path, episode, writer, 2, 1_000_000)
    assert list(tmp_path.glob("episode-*.json"))
    manifest = module.prune_incompatible_segments(
        tmp_path, 2, "different-protocol", 1_000_000
    )
    assert manifest["clips"] == []
    assert manifest["latest"] is None
    assert manifest["storage_bytes"] == 0
    assert not list(tmp_path.glob("episode-*.json"))


def test_oversized_episode_never_replaces_last_complete_archive(tmp_path):
    module = preview_module()
    first, first_writer, _ = prepared_episode(module, tmp_path, 10, "a")
    before = module.publish_episode(tmp_path, first, first_writer, 2, 1_000_000)
    second, second_writer, _ = prepared_episode(module, tmp_path, 20, "b")
    with pytest.raises(module.EpisodeTooLarge):
        module.publish_episode(tmp_path, second, second_writer, 2, 1)
    after = module.read_manifest(tmp_path, 2, 1_000_000)
    assert [clip["id"] for clip in after["clips"]] == [
        clip["id"] for clip in before["clips"]
    ]
    assert not list(tmp_path.glob(".*.stage"))


def test_delta_round_trip_covers_growth_collision_and_full_fallback():
    module = preview_module()
    initial = frame(0, snake=[[5, 4], [4, 4], [3, 4]])
    moved = frame(1, snake=[[6, 4], [5, 4], [4, 4]])
    grown = frame(2, score=1, snake=[[7, 4], [6, 4], [5, 4], [4, 4]])
    collision = frame(3, score=1, snake=grown["snake"], ended=True)
    fallback = frame(4, score=1, snake=[[1, 1], [1, 2], [1, 3], [2, 3]])
    previous = initial
    for current in (moved, grown, collision, fallback):
        delta = module.frame_delta(previous, current)
        assert module.apply_frame_delta(previous, delta) == current
        previous = current
    assert "snake" in module.frame_delta(collision, fallback)


def test_normalized_event_stream_has_causal_order():
    module = preview_module()
    current = frame(6, score=1)
    info = {
        "score": 1,
        "won": False,
        "termination_reason": "time_limit",
        "food_event": {
            "ticks": 4,
            "distance": 3,
            "free": 610,
            "cells": 620,
            "efficiency": 0.75,
            "reward": 1.25,
        },
        "perturbation_events": [
            {
                "kind": "obstacle",
                "status": "applied",
                "tick": 5,
                "announced_tick": 1,
                "activate_tick": 5,
                "cells": ((2, 2),),
                "reason": "certified_cycle",
            },
            {
                "kind": "obstacle",
                "status": "announced",
                "announced_tick": 6,
                "activate_tick": 12,
                "cells": ((3, 3),),
            },
            {
                "kind": "obstacle",
                "status": "rejected",
                "tick": 6,
                "reason": "proposal_geometry_or_limit",
            },
        ],
    }
    events = module.events_from_tick(current, 6, 2, 17, info)
    assert [event["type"] for event in events] == [
        "obstacle_applied",
        "food_eaten",
        "obstacle_announced",
        "obstacle_rejected",
        "game_end",
    ]
    assert all(event["episode_tick"] == 6 for event in events)
    assert events[0]["environment_event_tick"] == 5
    assert events[2]["activate_game_tick"] == 12
    assert events[3]["cells"] == []
    assert events[-1]["truncated"] is True


def test_rollout_is_one_complete_real_episode(tmp_path):
    torch = pytest.importorskip("torch")
    module = preview_module()
    from resnake_gym.gamepad_ppo import (
        CHECKPOINT_FORMAT,
        COLLECTION_SEMANTICS_VERSION,
        RECURRENT_UPDATE_VERSION,
        REPLAY_OBJECTIVE_VERSION,
        TRAINING_OBJECTIVE_VERSION,
        PPOConfig,
        build_model,
    )

    config = PPOConfig(
        max_logic_steps=3,
        dim=16,
        decision_min=2,
        decision_max=2,
        perturbation_min=100,
        perturbation_max=100,
    )
    policy = tmp_path / "policy-000010.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "training_objective": TRAINING_OBJECTIVE_VERSION,
            "collection_semantics": COLLECTION_SEMANTICS_VERSION,
            "recurrent_update": RECURRENT_UPDATE_VERSION,
            "replay_objective": REPLAY_OBJECTIVE_VERSION,
            "config": asdict(config),
            "model": build_model(config).state_dict(),
            "policy_version": 17,
            "logic_ticks": 123,
        },
        policy,
    )
    protocol = module.evaluation_protocol(config, 730000)
    protocol_sha = module.protocol_fingerprint(protocol)
    checkpoint_sha = module.sha256_file(policy)
    episode, writer = module.rollout_episode(
        policy,
        10,
        730000,
        "cpu",
        config,
        tmp_path,
        checkpoint_sha,
        protocol,
        protocol_sha,
        2,
        1_000_000,
    )
    assert episode["episode_ticks"] == 3
    assert episode["frame_count"] == 4
    assert episode["termination"] == "time_limit"
    assert episode["truncated"] is True
    assert episode["won"] is False
    assert "episodes" not in episode and "segment_ticks" not in episode
    assert [event["type"] for event in episode["events"]] == [
        "game_start",
        "game_end",
    ]
    manifest = module.publish_episode(tmp_path, episode, writer, 2, 1_000_000)
    metadata = json.loads((tmp_path / manifest["clips"][0]["url"]).read_text())
    decoded = decode_episode(module, tmp_path, metadata)
    assert [value["episode_tick"] for value in decoded] == [0, 1, 2, 3]
    assert decoded[-1]["truncated"] is True
    assert decoded[-1]["terminated"] is False


@pytest.mark.parametrize("action_head", ["dpad5", "held_dpad5"])
def test_vtrace_checkpoint_loads_for_preview(tmp_path, action_head):
    torch = pytest.importorskip("torch")
    module = preview_module()
    from resnake_gym.gamepad_vtrace import GamepadVTraceLearner, VTraceConfig

    config = VTraceConfig(
        max_logic_steps=3,
        dim=16,
        decision_min=2,
        decision_max=2,
        perturbation_min=100,
        perturbation_max=100,
        actor_processes=1,
        envs_per_actor=1,
        unroll_length=2,
        recurrent_burn_in=2,
        batch_min_transitions=2,
        batch_max_transitions=2,
        replay_capacity=8,
        replay_batch_fragments=1,
        replay_warmup_fragments=1,
        action_head=action_head,
    )
    learner = GamepadVTraceLearner(config)
    payload = learner.checkpoint()
    payload.update(training_iteration=7, policy_version=0, run_generation=0)
    policy = tmp_path / "policy-000007.pt"
    torch.save(payload, policy)

    metadata, loaded_config = module.load_policy_config(policy)
    _, model_config, model = module.load_policy(policy, "cpu")

    assert metadata["format"] == "gamepad-vtrace-v3"
    assert loaded_config == config
    assert model_config == config
    assert model.action_head == action_head

    protocol = module.evaluation_protocol(config, 731000)
    protocol_sha = module.protocol_fingerprint(protocol)
    episode, writer = module.rollout_episode(
        policy,
        7,
        731000,
        "cpu",
        config,
        tmp_path,
        module.sha256_file(policy),
        protocol,
        protocol_sha,
        2,
        1_000_000,
    )
    try:
        assert episode["episode_ticks"] == 3
        assert episode["frame_count"] == 4
        assert episode["termination"] == "time_limit"
        assert episode["truncated"] is True
        assert episode["action_head"] == action_head
        assert episode["action_encoding"] == metadata["action_encoding"]
        assert episode["training_objective"] == metadata["training_objective"]
        assert episode["checkpoint_format"] == "gamepad-vtrace-v3"
        assert episode["collection_semantics"] == metadata["collection_semantics"]
        assert episode["recurrent_state"] == metadata["recurrent_state"]
        assert episode["credit_trace_max_transitions"] == 2048
        assert episode["bptt_window"] == 128
        assert episode["action_variables_per_decision"] == (
            1 if action_head == "held_dpad5" else config.chunk_length
        )
    finally:
        writer.rollback()


def test_cache_identity_uses_content_seed_and_protocol():
    module = preview_module()
    key = module.cache_key_for("a" * 64, 7, "b" * 64)
    assert key == module.cache_key_for("a" * 64, 7, "b" * 64)
    assert key != module.cache_key_for("c" * 64, 7, "b" * 64)
    assert key != module.cache_key_for("a" * 64, 8, "b" * 64)
    assert key != module.cache_key_for("a" * 64, 7, "d" * 64)


def test_only_stable_lightweight_policies_are_observed(tmp_path):
    module = preview_module()
    old = tmp_path / "policy-000010.pt"
    new = tmp_path / "policy-000020.pt"
    full = tmp_path / "checkpoint.pt"
    for path in (old, new, full):
        path.write_bytes(b"fixture")
    timestamp = time.time() - 30
    os.utime(old, (timestamp, timestamp))
    assert module.stable_policies(tmp_path, stable_seconds=10) == [(10, old)]


def test_page_states_the_question_and_disables_unbounded_cache():
    module = preview_module()
    assert "color-scheme:light" in module.INDEX_HTML
    assert "固定评估协议与同一初始随机流" in module.INDEX_HTML
    assert "不是训练当时采集的轨迹" in module.INDEX_HTML
    assert "同一 seed 只固定初始随机流" in module.INDEX_HTML
    assert "cache:'no-store'" in module.INDEX_HTML
    assert "while(chunkCache.size>2)" in module.INDEX_HTML
    assert "自适应（约90秒/局）" in module.INDEX_HTML
    assert "playbackSpeed()" in module.INDEX_HTML
    assert "episode-end" in module.INDEX_HTML
    assert "<video" not in module.INDEX_HTML
    assert "<canvas" in module.INDEX_HTML


def test_manifest_recovers_from_partial_or_v1_json(tmp_path):
    module = preview_module()
    (tmp_path / "manifest.json").write_text('{"partial":')
    manifest = module.read_manifest(tmp_path, retention=4)
    assert manifest["clips"] == []
    assert manifest["schema_version"] == 2
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    assert module.read_manifest(tmp_path, 4)["clips"] == []


def protocol_config():
    return SimpleNamespace(
        width=31,
        height=20,
        initial_length=3,
        logic_fps=10.0,
        max_logic_steps=3000,
        perturbation_min=20,
        perturbation_max=40,
        chunk_length=8,
        decision_min=2,
        decision_max=5,
        observation_delay_max=3,
        command_delay_max=2,
        drop_probability=0.1,
        gamma=0.99,
        shaping_scale=0.25,
        death_cost=3.0,
    )


def worker_args(tmp_path):
    run = tmp_path / "run"
    site = tmp_path / "site"
    run.mkdir()
    site.mkdir()
    policy = run / "policy-000020.pt"
    policy.write_bytes(b"fixture")
    return (
        SimpleNamespace(
            run=run,
            site=site,
            retention=2,
            training_pid=None,
            max_storage_bytes=1_000_000,
            seed=730000,
            device="cpu",
            chunk_frames=2,
        ),
        policy,
    )


def test_protocol_serializes_every_effective_environment_default():
    module = preview_module()
    protocol = module.evaluation_protocol(protocol_config(), 730000)
    assert protocol["random_rotation"] is False
    assert protocol["solver_budget"]["time_limit_ms"] is None
    assert protocol["stick_deadzone"] == 0.2
    assert protocol["notice_ticks"] == 6
    assert protocol["protected_ticks"] == 3
    assert protocol["obstacle_probability"] == 0.5
    assert protocol["obstacle_batch_size"] == [1, 6]
    assert protocol["obstacle_sampler"] == "mixed-shapes-certificate-domain-v2"
    assert protocol["obstacle_size_filter"] == "cycle-certificate-count-v1"
    assert protocol["max_obstacle_cells"] == 32
    assert protocol["time_features"] is True
    assert protocol["gamepad_layout"]
    assert protocol["scene_version"]


def test_storage_evicted_target_is_remembered_instead_of_retried(tmp_path):
    module = preview_module()
    newest, newest_writer, _ = prepared_episode(module, tmp_path, 500, "a")
    manifest = module.publish_episode(tmp_path, newest, newest_writer, 2, 1_000_000)
    older, older_writer, _ = prepared_episode(module, tmp_path, 490, "b")
    older_filename = f"episode-{older['id']}.json"
    older_bytes = older_writer.staged_bytes + len(
        module.json_bytes({**older, "url": older_filename})
    )
    cap = max(manifest["clips"][0]["bytes"], older_bytes)
    manifest = module.publish_episode(tmp_path, older, older_writer, 2, cap)
    assert [clip["source_update"] for clip in manifest["clips"]] == [500]
    skipped = manifest["skipped_evaluations"]
    assert [(item["source_update"], item["reason"]) for item in skipped] == [
        (490, "storage_budget")
    ]
    clips, skipped = module._evaluation_cache_state(
        tmp_path,
        manifest,
        older["evaluation_protocol_sha256"],
        cap,
        {newest["cache_key"], older["cache_key"]},
    )
    assert {clip["source_update"] for clip in clips} == {500}
    assert {item["source_update"] for item in skipped} == {490}


def test_worker_keeps_previous_archive_if_replacement_evaluation_fails(
    tmp_path, monkeypatch
):
    module = preview_module()
    args, policy = worker_args(tmp_path)
    old, writer, _ = prepared_episode(module, args.site, 10, "a")
    before = module.publish_episode(args.site, old, writer, 2, 1_000_000)
    old_assets = set(before["clips"][0]["assets"])
    monkeypatch.setattr(module, "stable_policies", lambda _run: [(20, policy)])
    monkeypatch.setattr(
        module, "load_policy_config", lambda _path: ({}, protocol_config())
    )
    monkeypatch.setattr(module, "_checkpoint_sha", lambda _path, _cache: "b" * 64)
    monkeypatch.setattr(
        module,
        "rollout_episode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        module.worker_once(args, {})
    after = module.read_manifest(args.site, 2, 1_000_000)
    assert [clip["id"] for clip in after["clips"]] == [
        clip["id"] for clip in before["clips"]
    ]
    assert all((args.site / asset).is_file() for asset in old_assets)


def test_worker_records_oversized_target_and_does_not_retry(tmp_path, monkeypatch):
    module = preview_module()
    args, policy = worker_args(tmp_path)
    calls = []
    monkeypatch.setattr(module, "stable_policies", lambda _run: [(20, policy)])
    monkeypatch.setattr(
        module, "load_policy_config", lambda _path: ({}, protocol_config())
    )
    monkeypatch.setattr(module, "_checkpoint_sha", lambda _path, _cache: "b" * 64)

    def too_large(*_args, **_kwargs):
        calls.append(1)
        raise module.EpisodeTooLarge("fixture")

    monkeypatch.setattr(module, "rollout_episode", too_large)
    module.worker_once(args, {})
    module.worker_once(args, {})
    manifest = module.read_manifest(args.site, 2, 1_000_000)
    assert len(calls) == 1
    assert manifest["status"] == "limited"
    assert len(manifest["skipped_evaluations"]) == 1


def test_cleanup_failure_cannot_break_committed_manifest(tmp_path, monkeypatch):
    module = preview_module()
    episode, writer, _ = prepared_episode(module, tmp_path, 10, "a")
    orphan = tmp_path / "episode-orphan.json"
    orphan.write_text("orphan")
    original_unlink = Path.unlink

    def guarded_unlink(path, *args, **kwargs):
        if path == orphan:
            raise PermissionError("fixture")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    manifest = module.publish_episode(tmp_path, episode, writer, 2, 1_000_000)
    assert manifest["clips"][0]["id"] == episode["id"]
    assert all((tmp_path / asset).is_file() for asset in manifest["clips"][0]["assets"])
    assert orphan.exists()


def test_chunk_commit_repairs_partial_archive_with_same_cache_key(tmp_path):
    module = preview_module()
    identifier = "u000010-" + "a" * 64
    stale = tmp_path / f"episode-{identifier}-c0000.json"
    stale.write_text("partial")
    writer = module.EpisodeArchiveWriter(tmp_path, identifier, 2, 1_000_000)
    writer.add(frame(0))
    writer.add(frame(1))
    writer.finalize()
    writer.commit()
    payload = json.loads(stale.read_text())
    assert payload["episode_id"] == identifier
    assert payload["initial"]["episode_tick"] == 0


def test_embedded_javascript_parses_and_has_single_poll_guard():
    module = preview_module()
    script = re.search(r"<script>(.*)</script>", module.INDEX_HTML, re.S).group(1)
    assert "request===refreshRequest" in script
    assert "request===frameRequest" in script
    assert "scheduleRefresh(500)" in script
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    subprocess.run([node, "--check"], input=script, text=True, check=True)

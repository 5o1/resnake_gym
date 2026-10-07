"""Checkpoint discovery and background preview evaluation."""

import time

from resnake_gym.process_status import process_command_matches

from .archive import (
    EpisodeTooLarge,
    _clip_assets,
    _matching_skips,
    prune_incompatible_segments,
    publish_episode,
    read_manifest,
    record_skipped_evaluation,
    stable_policies,
    update_status,
)
from .evaluation import (
    cache_key_for,
    evaluation_protocol,
    load_policy_config,
    protocol_fingerprint,
    rollout_episode,
    sha256_file,
)


def training_alive(pid, run):
    if not pid:
        return None
    return process_command_matches(
        pid,
        (b"train_gamepad_ppo.py", b"train_gamepad_vtrace.py"),
        run,
    )


def _checkpoint_sha(path, hash_cache):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in hash_cache:
        for old_key in [item for item in hash_cache if item[0] == str(path)]:
            del hash_cache[old_key]
        hash_cache[key] = sha256_file(path)
    return hash_cache[key]


def _evaluation_cache_state(
    site, manifest, protocol_sha256, max_storage_bytes, target_keys
):
    clips = [
        clip
        for clip in manifest.get("clips", [])
        if clip.get("evaluation_protocol_sha256") == protocol_sha256
        and clip.get("cache_key") in target_keys
        and clip.get("episode_complete") is True
        and all((site / name).is_file() for name in _clip_assets(clip))
    ]
    skipped = _matching_skips(manifest, protocol_sha256, max_storage_bytes, target_keys)
    return clips, skipped


def worker_once(args, hash_cache):
    policies = stable_policies(args.run)[-args.retention :]
    alive = training_alive(args.training_pid, args.run)
    if not policies:
        update_status(
            args.site,
            args.retention,
            args.max_storage_bytes,
            status="waiting",
            message="尚无稳定策略快照",
            training_alive=alive,
        )
        return

    latest_update, latest_policy = policies[-1]
    _, evaluation_config = load_policy_config(latest_policy)
    protocol = evaluation_protocol(evaluation_config, args.seed)
    protocol_sha256 = protocol_fingerprint(protocol)
    targets = []
    for update, policy in policies:
        checkpoint_sha256 = _checkpoint_sha(policy, hash_cache)
        targets.append(
            {
                "update": update,
                "path": policy,
                "checkpoint_sha256": checkpoint_sha256,
                "cache_key": cache_key_for(
                    checkpoint_sha256, args.seed, protocol_sha256
                ),
            }
        )
    target_keys = {target["cache_key"] for target in targets}
    # Keep the previous complete comparison playable until the first archive
    # under a new protocol/checkpoint set has been published successfully.
    manifest = read_manifest(args.site, args.retention, args.max_storage_bytes)
    current_clips, skipped = _evaluation_cache_state(
        args.site,
        manifest,
        protocol_sha256,
        args.max_storage_bytes,
        target_keys,
    )
    completed = {clip["cache_key"] for clip in current_clips}
    completed.update(item["cache_key"] for item in skipped)
    missing = [
        target for target in reversed(targets) if target["cache_key"] not in completed
    ]
    if missing:
        target = missing[0]
        update_status(
            args.site,
            args.retention,
            args.max_storage_bytes,
            status="evaluating",
            message=(
                f"正在独立评估 update {target['update']} 的完整单局；"
                f"待补 {len(missing)} 个 checkpoint"
            ),
            training_alive=alive,
            training_update=latest_update,
            pending_evaluation_protocol_sha256=protocol_sha256,
        )
        try:
            episode, writer = rollout_episode(
                target["path"],
                target["update"],
                args.seed,
                args.device,
                evaluation_config,
                args.site,
                target["checkpoint_sha256"],
                protocol,
                protocol_sha256,
                args.chunk_frames,
                args.max_storage_bytes,
            )
            # Lightweight snapshots are immutable by convention; verify that
            # the exact bytes evaluated still name the file before publication.
            if sha256_file(target["path"]) != target["checkpoint_sha256"]:
                writer.rollback()
                raise RuntimeError("policy snapshot changed during evaluation")
            manifest = publish_episode(
                args.site,
                episode,
                writer,
                args.retention,
                args.max_storage_bytes,
            )
        except EpisodeTooLarge as exc:
            manifest = record_skipped_evaluation(
                args.site,
                args.retention,
                args.max_storage_bytes,
                cache_key=target["cache_key"],
                source_update=target["update"],
                protocol_sha256=protocol_sha256,
                reason=f"storage_budget: {exc}",
            )

    current_clips, skipped = _evaluation_cache_state(
        args.site,
        manifest,
        protocol_sha256,
        args.max_storage_bytes,
        target_keys,
    )
    completed = {clip["cache_key"] for clip in current_clips}
    completed.update(item["cache_key"] for item in skipped)
    missing = [target for target in targets if target["cache_key"] not in completed]
    if not missing and current_clips:
        manifest = prune_incompatible_segments(
            args.site,
            args.retention,
            protocol_sha256,
            args.max_storage_bytes,
            target_keys,
        )
        current_clips, skipped = _evaluation_cache_state(
            args.site,
            manifest,
            protocol_sha256,
            args.max_storage_bytes,
            target_keys,
        )
    limited = bool(skipped)
    values = {
        "status": "evaluating" if missing else "limited" if limited else "ready",
        "message": (
            f"待补 {len(missing)} 个 checkpoint"
            if missing
            else f"{len(skipped)} 个 checkpoint 因存储上限未发布"
            if limited
            else ""
        ),
        "training_alive": alive,
        "training_update": latest_update,
        "pending_evaluation_protocol_sha256": protocol_sha256 if missing else None,
    }
    if current_clips:
        values.update(
            evaluation_protocol=protocol,
            evaluation_protocol_sha256=protocol_sha256,
        )
    update_status(
        args.site,
        args.retention,
        args.max_storage_bytes,
        **values,
    )


def worker(args):
    import torch

    torch.set_num_threads(1)
    hash_cache = {}
    while True:
        try:
            worker_once(args, hash_cache)
        except Exception as exc:  # Keep serving the last complete archive.
            update_status(
                args.site,
                args.retention,
                args.max_storage_bytes,
                status="error",
                message=f"{type(exc).__name__}: {exc}",
            )
        time.sleep(args.poll_seconds)

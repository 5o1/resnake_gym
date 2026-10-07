"""Serve bounded, controlled full-episode comparisons of recent checkpoints.

The worker only reads stable lightweight ``policy-*.pt`` snapshots. It never
attaches to the trainer, reads the full checkpoint, or writes into the run.
Every retained item uses the same environment seed and deterministic actions;
the only intended varying factor is the checkpoint. These are independent
evaluations, never reconstructions of training trajectories.

Business logic lives in :mod:`resnake_gym.preview`; imports below intentionally
preserve this script's historical test and tooling API.
"""

# ruff: noqa: F401 -- Imports are the script's compatibility re-export surface.

import argparse
import functools
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from resnake_gym.preview import worker as _worker_module
from resnake_gym.preview.archive import (
    DEFAULT_STORAGE_BYTES,
    SCHEMA_VERSION,
    EpisodeArchiveWriter,
    EpisodeTooLarge,
    _clip_assets,
    _matching_skips,
    _skip_record,
    _stored_bytes,
    _write_clips,
    apply_frame_delta,
    atomic_bytes,
    atomic_json,
    atomic_text,
    empty_manifest,
    frame_delta,
    json_bytes,
    prune_incompatible_segments,
    publish_episode,
    read_manifest,
    record_skipped_evaluation,
    snapshot,
    stable_policies,
    update_status,
)
from resnake_gym.preview.assets import INDEX_HTML, NoStoreHandler
from resnake_gym.preview.evaluation import (
    _event_common,
    cache_key_for,
    evaluation_protocol,
    events_from_tick,
    load_policy,
    load_policy_config,
    make_preview_env,
    protocol_fingerprint,
    rollout_episode,
    sha256_file,
)
from resnake_gym.preview.worker import (
    _checkpoint_sha,
    _evaluation_cache_state,
    training_alive,
)


def worker_once(args, hash_cache):
    """Compatibility adapter that keeps script-level monkeypatching effective."""
    names = (
        "stable_policies",
        "training_alive",
        "load_policy_config",
        "evaluation_protocol",
        "protocol_fingerprint",
        "_checkpoint_sha",
        "cache_key_for",
        "read_manifest",
        "_evaluation_cache_state",
        "update_status",
        "rollout_episode",
        "sha256_file",
        "publish_episode",
        "record_skipped_evaluation",
        "prune_incompatible_segments",
        "EpisodeTooLarge",
    )
    previous = {name: getattr(_worker_module, name) for name in names}
    try:
        for name in names:
            setattr(_worker_module, name, globals()[name])
        return _worker_module.worker_once(args, hash_cache)
    finally:
        for name, value in previous.items():
            setattr(_worker_module, name, value)


worker = _worker_module.worker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--retention", type=int, default=6)
    parser.add_argument("--seed", type=int, default=730000)
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--max-storage-mib", type=float, default=128.0)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--training-pid", type=int)
    args = parser.parse_args()
    if (
        not 1 <= args.retention <= 50
        or not 2 <= args.chunk_frames <= 4096
        or not 0 < args.max_storage_mib < float("inf")
        or not 1 <= args.port <= 65535
        or args.poll_seconds <= 0
    ):
        parser.error("invalid retention, chunk size, storage, port, or poll interval")
    args.max_storage_bytes = int(args.max_storage_mib * 1024 * 1024)
    args.run, args.site = args.run.resolve(), args.site.resolve()
    return args


def main():
    args = parse_args()
    args.site.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(NoStoreHandler, directory=str(args.site))
    # Reserve the public endpoint before touching its currently valid index.
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    atomic_text(args.site / "index.html", INDEX_HTML)
    prune_incompatible_segments(
        args.site,
        args.retention,
        max_storage_bytes=args.max_storage_bytes,
    )
    thread = threading.Thread(
        target=worker, args=(args,), daemon=True, name="preview-worker"
    )
    thread.start()
    print(
        json.dumps(
            {"url": f"http://{args.host}:{args.port}/", "retention": args.retention}
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

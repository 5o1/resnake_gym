"""Atomic storage, archive chunks, and manifest management."""

import contextlib
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(
        path, json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


SCHEMA_VERSION = 2
DEFAULT_STORAGE_BYTES = 128 * 1024 * 1024


class EpisodeTooLarge(RuntimeError):
    """A complete episode cannot fit inside the configured archive budget."""


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def empty_manifest(retention, max_storage_bytes):
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": 0,
        "retention": retention,
        "max_storage_bytes": max_storage_bytes,
        "clips": [],
        "skipped_evaluations": [],
        "status": "starting",
        "latest": None,
    }


def read_manifest(site, retention, max_storage_bytes=DEFAULT_STORAGE_BYTES):
    path = site / "manifest.json"
    if path.exists():
        try:
            value = json.loads(path.read_text())
            if value.get("schema_version") != SCHEMA_VERSION:
                return empty_manifest(retention, max_storage_bytes)
            value["retention"] = retention
            value["max_storage_bytes"] = max_storage_bytes
            return value
        except (OSError, json.JSONDecodeError):
            pass
    return empty_manifest(retention, max_storage_bytes)


def _clip_assets(clip):
    return [str(name) for name in clip.get("assets", ())]


def _skip_record(
    cache_key,
    source_update,
    protocol_sha256,
    max_storage_bytes,
    reason,
    archive_bytes=None,
):
    return {
        "cache_key": cache_key,
        "source_update": source_update,
        "evaluation_protocol_sha256": protocol_sha256,
        "max_storage_bytes": max_storage_bytes,
        "reason": reason,
        "archive_bytes": archive_bytes,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _matching_skips(manifest, protocol_sha256, max_storage_bytes, cache_keys=None):
    return [
        item
        for item in manifest.get("skipped_evaluations", [])
        if item.get("evaluation_protocol_sha256") == protocol_sha256
        and item.get("max_storage_bytes") == max_storage_bytes
        and (cache_keys is None or item.get("cache_key") in cache_keys)
    ]


def record_skipped_evaluation(
    site,
    retention,
    max_storage_bytes,
    *,
    cache_key,
    source_update,
    protocol_sha256,
    reason,
    archive_bytes=None,
):
    manifest = read_manifest(site, retention, max_storage_bytes)
    skipped = [
        item
        for item in manifest.get("skipped_evaluations", [])
        if item.get("cache_key") != cache_key
    ]
    skipped.append(
        _skip_record(
            cache_key,
            source_update,
            protocol_sha256,
            max_storage_bytes,
            reason,
            archive_bytes,
        )
    )
    manifest["skipped_evaluations"] = skipped[-retention:]
    atomic_json(site / "manifest.json", manifest)
    return manifest


def _stored_bytes(site, clips):
    return sum(
        (site / name).stat().st_size
        for clip in clips
        for name in _clip_assets(clip)
        if (site / name).is_file()
    )


def _cleanup_unreferenced(site, clips):
    keep = {name for clip in clips for name in _clip_assets(clip)}
    for pattern in ("segment-*.json", "episode-*.json"):
        for path in site.glob(pattern):
            if path.name not in keep:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                # The manifest is already committed. A later poll retries
                # failed orphan cleanup; never break references to new assets.
    for pattern in (".*.tmp", ".*.stage"):
        for path in site.glob(pattern):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)


def _write_clips(site, manifest, clips, *, status=None, message=None):
    clips = sorted(clips, key=lambda clip: clip["source_update"])
    latest = clips[-1] if clips else None
    manifest.update(
        clips=clips,
        latest=latest["id"] if latest else None,
        latest_update=latest["source_update"] if latest else None,
        storage_bytes=_stored_bytes(site, clips),
    )
    if status is not None:
        manifest["status"] = status
    if message is not None:
        manifest["message"] = message
    atomic_json(site / "manifest.json", manifest)
    _cleanup_unreferenced(site, clips)
    return manifest


def prune_incompatible_segments(
    site,
    retention,
    protocol_sha256=None,
    max_storage_bytes=DEFAULT_STORAGE_BYTES,
    cache_keys=None,
):
    """Keep only complete schema-v2 archives for the active protocol."""
    manifest = read_manifest(site, retention, max_storage_bytes)
    clips = [
        clip
        for clip in manifest.get("clips", [])
        if clip.get("schema_version") == SCHEMA_VERSION
        and clip.get("episode_complete") is True
        and all((site / name).is_file() for name in _clip_assets(clip))
        and (
            protocol_sha256 is None
            or clip.get("evaluation_protocol_sha256") == protocol_sha256
        )
        and (cache_keys is None or clip.get("cache_key") in cache_keys)
    ]
    clips = sorted(clips, key=lambda clip: clip["source_update"])[-retention:]
    evicted = []
    while clips and _stored_bytes(site, clips) > max_storage_bytes:
        evicted.append(clips.pop(0))
    if protocol_sha256 is None:
        skipped = manifest.get("skipped_evaluations", [])[-retention:]
    else:
        skipped = _matching_skips(
            manifest, protocol_sha256, max_storage_bytes, cache_keys
        )
        skipped.extend(
            _skip_record(
                clip["cache_key"],
                clip["source_update"],
                protocol_sha256,
                max_storage_bytes,
                "storage_budget",
                clip.get("bytes"),
            )
            for clip in evicted
        )
        skipped = list({item["cache_key"]: item for item in skipped}.values())[
            -retention:
        ]
    values = dict(
        schema_version=SCHEMA_VERSION,
        retention=retention,
        max_storage_bytes=max_storage_bytes,
        skipped_evaluations=skipped,
        status="ready" if clips else "starting",
        message="" if clips else "正在生成完整单局评估",
    )
    if protocol_sha256 is not None:
        values["evaluation_protocol_sha256"] = protocol_sha256
    manifest.update(values)
    return _write_clips(site, manifest, clips)


def update_status(site, retention, max_storage_bytes=DEFAULT_STORAGE_BYTES, **values):
    manifest = read_manifest(site, retention, max_storage_bytes)
    manifest.update(values, checked_at=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(site / "manifest.json", manifest)


def stable_policies(run, stable_seconds=10):
    now = time.time()
    result = []
    for path in run.glob("policy-*.pt"):
        try:
            update = int(path.stem.rsplit("-", 1)[1])
            if now - path.stat().st_mtime >= stable_seconds:
                result.append((update, path))
        except (OSError, ValueError):
            continue
    return sorted(result)


def snapshot(game, info, decision):
    state = game.state
    pending = game.pending or {}
    return {
        "episode_tick": int(state.logic_steps),
        "simulation_seconds": state.elapsed_seconds,
        "decision": decision,
        "duration_ticks": 1,
        "width": state.width,
        "height": state.height,
        "snake": [list(cell) for cell in state.snake],
        "food": list(state.food) if state.food is not None else None,
        "obstacles": [list(cell) for cell in sorted(game.kernel.obstacle_cells)],
        "pending_obstacles": [list(cell) for cell in pending.get("cells", ())],
        "pending_rotation": int(pending.get("quarters", 0)),
        "direction": state.direction,
        "score": int(info.get("score", state.score)),
        "ate": info.get("food_event") is not None,
        "terminated": bool(info.get("termination_reason"))
        and info.get("termination_reason") != "time_limit",
        "truncated": info.get("termination_reason") == "time_limit",
        "actual_report": [round(float(value), 4) for value in game.last_report],
    }


def frame_delta(previous, current):
    """Encode one logic-frame transition; omitted keys remain unchanged."""
    delta = {
        "episode_tick": current["episode_tick"],
        "simulation_seconds": current["simulation_seconds"],
        "ate": current["ate"],
        "terminated": current["terminated"],
        "truncated": current["truncated"],
    }
    old_snake, new_snake = previous["snake"], current["snake"]
    if new_snake != old_snake:
        shifted = len(new_snake) == len(old_snake) and new_snake[1:] == old_snake[:-1]
        grown = len(new_snake) == len(old_snake) + 1 and new_snake[1:] == old_snake
        if shifted or grown:
            delta["head_added"] = new_snake[0]
            delta["tail_removed"] = shifted
        else:
            delta["snake"] = new_snake
    for key in (
        "decision",
        "width",
        "height",
        "food",
        "obstacles",
        "pending_obstacles",
        "pending_rotation",
        "direction",
        "score",
        "actual_report",
    ):
        if current[key] != previous[key]:
            delta[key] = current[key]
    return delta


def apply_frame_delta(previous, delta):
    """Reference decoder; the browser implements this same small format."""
    current = {**previous}
    if "snake" in delta:
        current["snake"] = delta["snake"]
    elif "head_added" in delta:
        snake = [delta["head_added"], *previous["snake"]]
        if delta.get("tail_removed"):
            snake.pop()
        current["snake"] = snake
    for key, value in delta.items():
        if key not in ("snake", "head_added", "tail_removed"):
            current[key] = value
    return current


class EpisodeArchiveWriter:
    """Stream independent frame chunks without retaining a whole episode."""

    def __init__(self, site, identifier, chunk_frames, max_archive_bytes):
        self.site = site
        self.identifier = identifier
        self.chunk_frames = chunk_frames
        self.max_archive_bytes = max_archive_bytes
        self.frame_count = 0
        self.staged_bytes = 0
        self.descriptors = []
        self._initial = None
        self._previous = None
        self._deltas = []
        self._chunk_start = 0
        self._staged = []
        self._final = []

    def add(self, frame):
        if self._initial is not None and 1 + len(self._deltas) >= self.chunk_frames:
            self._flush()
        if self._initial is None:
            self._initial = frame
            self._previous = frame
            self._chunk_start = self.frame_count
        else:
            self._deltas.append(frame_delta(self._previous, frame))
            self._previous = frame
        self.frame_count += 1

    def _flush(self):
        if self._initial is None:
            return
        index = len(self.descriptors)
        filename = f"episode-{self.identifier}-c{index:04d}.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.identifier,
            "index": index,
            "initial": self._initial,
            "deltas": self._deltas,
        }
        encoded = json_bytes(payload)
        if self.staged_bytes + len(encoded) > self.max_archive_bytes:
            raise EpisodeTooLarge(
                f"complete episode exceeds {self.max_archive_bytes} bytes"
            )
        stage = self.site / f".{filename}.stage"
        atomic_bytes(stage, encoded)
        count = 1 + len(self._deltas)
        self.descriptors.append(
            {
                "url": filename,
                "start_frame": self._chunk_start,
                "end_frame": self._chunk_start + count - 1,
                "start_tick": self._initial["episode_tick"],
                "end_tick": self._previous["episode_tick"],
                "frames": count,
                "bytes": len(encoded),
            }
        )
        self._staged.append((stage, self.site / filename))
        self.staged_bytes += len(encoded)
        self._initial = self._previous = None
        self._deltas = []

    def finalize(self):
        self._flush()
        return self.descriptors

    def commit(self):
        try:
            for stage, final in self._staged:
                # The content-addressed name may survive a prior partial
                # publication. Atomic replacement repairs that unplayable
                # archive; a complete referenced key is never reevaluated.
                os.replace(stage, final)
                self._final.append(final)
            self._staged.clear()
        except Exception:
            self.rollback()
            raise

    def rollback(self):
        for stage, _ in self._staged:
            stage.unlink(missing_ok=True)
        for final in self._final:
            final.unlink(missing_ok=True)
        self._staged.clear()
        self._final.clear()


def publish_episode(site, episode, writer, retention, max_storage_bytes):
    """Expose one completed archive, then atomically update and prune its index."""
    filename = f"episode-{episode['id']}.json"
    episode = {**episode, "url": filename}
    encoded = json_bytes(episode)
    archive_bytes = writer.staged_bytes + len(encoded)
    if archive_bytes > max_storage_bytes:
        writer.rollback()
        raise EpisodeTooLarge(
            "complete episode needs "
            f"{archive_bytes} bytes; budget is {max_storage_bytes}"
        )
    writer.commit()
    try:
        atomic_bytes(site / filename, encoded)
        assets = [filename, *(chunk["url"] for chunk in episode["chunks"])]
        summary = {
            key: episode[key]
            for key in (
                "id",
                "cache_key",
                "source_update",
                "policy_version",
                "seed",
                "mode",
                "episode_ticks",
                "final_score",
                "termination",
                "won",
                "truncated",
                "episode_complete",
                "evaluation_protocol_sha256",
                "generated_at",
            )
        }
        summary.update(
            schema_version=SCHEMA_VERSION,
            url=filename,
            assets=assets,
            bytes=archive_bytes,
            action_encoding=episode.get("action_encoding"),
            training_objective=episode.get("training_objective"),
            checkpoint_format=episode.get("checkpoint_format"),
            collection_semantics=episode.get("collection_semantics"),
            recurrent_state=episode.get("recurrent_state"),
            credit_trace_max_transitions=episode.get("credit_trace_max_transitions"),
            bptt_window=episode.get("bptt_window"),
            action_variables_per_decision=episode.get("action_variables_per_decision"),
            source_received_logic_ticks=episode.get("source_received_logic_ticks"),
            source_trained_logic_ticks=episode.get("source_trained_logic_ticks"),
        )
        manifest = read_manifest(site, retention, max_storage_bytes)
        if manifest.get("evaluation_protocol_sha256") not in (
            None,
            episode["evaluation_protocol_sha256"],
        ):
            clips = []
            skipped = _matching_skips(
                manifest,
                episode["evaluation_protocol_sha256"],
                max_storage_bytes,
            )
        else:
            clips = [
                clip
                for clip in manifest.get("clips", [])
                if clip.get("source_update") != episode["source_update"]
                and clip.get("cache_key") != episode["cache_key"]
            ]
            skipped = [
                item
                for item in _matching_skips(
                    manifest,
                    episode["evaluation_protocol_sha256"],
                    max_storage_bytes,
                )
                if item.get("cache_key") != episode["cache_key"]
            ]
        clips.append(summary)
        clips = sorted(clips, key=lambda clip: clip["source_update"])[-retention:]
        capacity_evicted = []
        while len(clips) > 1 and _stored_bytes(site, clips) > max_storage_bytes:
            capacity_evicted.append(clips.pop(0))
        if _stored_bytes(site, clips) > max_storage_bytes:
            raise EpisodeTooLarge("published episode exceeds total storage budget")
        skipped.extend(
            _skip_record(
                clip["cache_key"],
                clip["source_update"],
                episode["evaluation_protocol_sha256"],
                max_storage_bytes,
                "storage_budget",
                clip.get("bytes"),
            )
            for clip in capacity_evicted
        )
        skipped = list({item["cache_key"]: item for item in skipped}.values())[
            -retention:
        ]
        manifest.update(
            schema_version=SCHEMA_VERSION,
            generation=int(manifest.get("generation", 0)) + 1,
            evaluation_protocol=episode["evaluation_protocol"],
            evaluation_protocol_sha256=episode["evaluation_protocol_sha256"],
            skipped_evaluations=skipped,
            updated_at=episode["generated_at"],
        )
        manifest = _write_clips(site, manifest, clips, status="ready", message="")
        writer._final.clear()
        return manifest
    except Exception:
        (site / filename).unlink(missing_ok=True)
        writer.rollback()
        raise

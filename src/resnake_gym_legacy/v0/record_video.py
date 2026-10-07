"""Record a deterministic, time-compressed ReSnake episode as an MP4."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from resnake_gym.baselines import (
    HamiltonianOracle,
    absolute_action,
    cycle_aligned_snake,
)
from resnake_gym.envs import SnakeEnv

DEFAULT_WIDTH = 31
DEFAULT_HEIGHT = 20
DEFAULT_SEED = 0
DEFAULT_MAX_LOGIC_STEPS = 200_000
DEFAULT_CELL_SIZE = 8
DEFAULT_FPS = 30.0
DEFAULT_DENSE_START_TICKS = 300
DEFAULT_DENSE_END_TICKS = 300
DEFAULT_MIDDLE_FRAMES = 1_200

ActionFunction = Callable[[np.ndarray, SnakeEnv], int]
FrameFunction = Callable[[int, SnakeEnv], None]


@dataclass(frozen=True)
class Outcome:
    """Terminal episode data plus a digest of the complete trajectory."""

    terminated: bool
    truncated: bool
    won: bool
    reason: str | None
    score: int
    length: int
    ticks: int
    trajectory_sha256: str

    @property
    def strict_win(self) -> bool:
        return self.terminated and self.won and self.reason == "board_filled"


@dataclass(frozen=True)
class FrameSchedule:
    """Chronologically ordered logic ticks to include in the video."""

    ticks: tuple[int, ...]
    dense_start_ticks: int
    dense_end_ticks: int
    middle_target_frames: int
    middle_frames: int


@dataclass(frozen=True)
class FfmpegInfo:
    """The selected ffmpeg executable and its identifying information."""

    executable: str
    source: str
    version: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def build_schedule(
    total_ticks: int,
    *,
    dense_start_ticks: int = DEFAULT_DENSE_START_TICKS,
    dense_end_ticks: int = DEFAULT_DENSE_END_TICKS,
    middle_frames: int = DEFAULT_MIDDLE_FRAMES,
) -> FrameSchedule:
    """Build a duplicate-free schedule containing the initial and final states."""

    if total_ticks < 0:
        raise ValueError("total_ticks must be non-negative")
    if dense_start_ticks < 0 or dense_end_ticks < 0 or middle_frames < 0:
        raise ValueError("schedule counts must be non-negative")

    first_dense_end = min(total_ticks, dense_start_ticks)
    selected = set(range(first_dense_end + 1))

    last_dense_start = max(1, total_ticks - dense_end_ticks + 1)
    if dense_end_ticks:
        selected.update(range(last_dense_start, total_ticks + 1))

    middle_start = first_dense_end + 1
    middle_end = min(total_ticks - 1, total_ticks - dense_end_ticks)
    middle_count = max(0, middle_end - middle_start + 1)
    sampled_middle: tuple[int, ...]
    if middle_frames == 0 or middle_count == 0:
        sampled_middle = ()
    elif middle_count <= middle_frames:
        sampled_middle = tuple(range(middle_start, middle_end + 1))
    elif middle_frames == 1:
        sampled_middle = ((middle_start + middle_end) // 2,)
    else:
        span = middle_count - 1
        denominator = middle_frames - 1
        sampled_middle = tuple(
            middle_start + index * span // denominator for index in range(middle_frames)
        )
    selected.update(sampled_middle)
    selected.add(0)
    selected.add(total_ticks)

    ticks = tuple(sorted(selected))
    if len(ticks) != len(set(ticks)):
        raise AssertionError("frame schedule contains duplicate ticks")
    if not ticks or ticks[0] != 0 or ticks[-1] != total_ticks:
        raise AssertionError("frame schedule must include initial and final states")
    return FrameSchedule(
        ticks=ticks,
        dense_start_ticks=dense_start_ticks,
        dense_end_ticks=dense_end_ticks,
        middle_target_frames=middle_frames,
        middle_frames=len(sampled_middle),
    )


def _oracle_initial_options(
    oracle: HamiltonianOracle,
) -> dict[str, object]:
    snake = cycle_aligned_snake(oracle.cycle, length=3, head_index=2)
    direction = absolute_action(snake[1], snake[0])
    return {"snake": snake, "direction": direction}


def _update_trajectory_digest(
    digest: Any,
    observation: np.ndarray,
    *,
    action: int | None,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
) -> None:
    action_value = -1 if action is None else action
    digest.update(
        struct.pack(
            "<qd??",
            action_value,
            reward,
            terminated,
            truncated,
        )
    )
    contiguous = np.ascontiguousarray(observation)
    digest.update(memoryview(contiguous).cast("B"))


def _rollout(
    *,
    width: int,
    height: int,
    seed: int,
    action_mode: str,
    max_logic_steps: int,
    cell_size: int,
    action_function: ActionFunction,
    reset_options: dict[str, object] | None,
    frame_ticks: frozenset[int] = frozenset(),
    frame_function: FrameFunction | None = None,
) -> Outcome:
    env = SnakeEnv(
        width=width,
        height=height,
        action_mode=action_mode,
        frame_skip=1,
        max_logic_steps=max_logic_steps,
        render_mode="rgb_array",
        cell_size=cell_size,
    )
    digest = hashlib.sha256()
    try:
        observation, info = env.reset(seed=seed, options=reset_options)
        _update_trajectory_digest(digest, observation, action=None)
        if 0 in frame_ticks:
            if frame_function is None:
                raise AssertionError("frame ticks require a frame function")
            frame_function(0, env)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = action_function(observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
            _update_trajectory_digest(
                digest,
                observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            tick = int(info["logic_steps"])
            if tick in frame_ticks:
                if frame_function is None:
                    raise AssertionError("frame ticks require a frame function")
                frame_function(tick, env)

        return Outcome(
            terminated=bool(terminated),
            truncated=bool(truncated),
            won=bool(info["won"]),
            reason=info["termination_reason"],
            score=int(info["score"]),
            length=int(info["length"]),
            ticks=int(info["logic_steps"]),
            trajectory_sha256=digest.hexdigest(),
        )
    finally:
        env.close()


def _resolve_model_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_file() and not candidate.suffix:
        zip_candidate = Path(f"{candidate}.zip")
        if zip_candidate.is_file():
            candidate = zip_candidate
    if not candidate.is_file():
        raise ValueError(f"model checkpoint is not a file: {candidate}")
    return candidate.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_action_function(
    controller: str,
    *,
    width: int,
    height: int,
    action_mode: str,
    model_path: Path | None,
    device: str,
) -> tuple[ActionFunction, dict[str, object] | None, Path | None, str | None]:
    if controller == "oracle":
        if model_path is not None:
            raise ValueError("--model is only valid with a learned controller")
        oracle = HamiltonianOracle(width, height)

        def oracle_action(_observation: np.ndarray, env: SnakeEnv) -> int:
            return oracle.act_head(
                env.head,
                env.direction,
                action_mode=action_mode,
            )

        return oracle_action, _oracle_initial_options(oracle), None, None

    if model_path is None:
        raise ValueError(f"--model is required with --controller {controller}")
    resolved_model = _resolve_model_path(model_path)
    model_sha256 = _sha256_file(resolved_model)

    try:
        if controller == "ppo":
            from stable_baselines3 import PPO

            model = PPO.load(resolved_model, device=device)
        elif controller == "qrdqn":
            from sb3_contrib import QRDQN

            model = QRDQN.load(resolved_model, device=device)
        else:
            raise ValueError(f"unsupported controller: {controller}")
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            'learned controllers require `pip install -e ".[rl]"`'
        ) from exc

    def learned_action(observation: np.ndarray, _env: SnakeEnv) -> int:
        prediction, _ = model.predict(observation, deterministic=True)
        values = np.asarray(prediction)
        if values.size != 1:
            raise RuntimeError(
                f"model returned {values.size} actions for one observation"
            )
        return int(values.item())

    return learned_action, None, resolved_model, model_sha256


def _ffmpeg_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"could not execute ffmpeg at {executable}: {exc}") from exc
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg version probe exited with status {completed.returncode}: {output}"
        )
    return output.splitlines()[0] if output else "unknown"


def _find_ffmpeg() -> FfmpegInfo:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        executable = str(Path(system_ffmpeg).resolve())
        return FfmpegInfo(
            executable=executable,
            source="system",
            version=_ffmpeg_version(executable),
        )

    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError(
            "ffmpeg was not found; install ffmpeg or the imageio-ffmpeg package"
        ) from exc
    try:
        executable = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve())
    except Exception as exc:  # pragma: no cover - dependency-specific failure
        raise RuntimeError(f"imageio-ffmpeg could not provide ffmpeg: {exc}") from exc
    return FfmpegInfo(
        executable=executable,
        source="imageio_ffmpeg",
        version=_ffmpeg_version(executable),
    )


def _format_fps(fps: float) -> str:
    return format(fps, ".12g")


def _ffmpeg_command(
    ffmpeg: FfmpegInfo,
    *,
    input_width: int,
    input_height: int,
    fps: float,
    output: Path,
) -> list[str]:
    return [
        ffmpeg.executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{input_width}x{input_height}",
        "-framerate",
        _format_fps(fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output),
    ]


def _decode_stderr(stderr: bytes | None) -> str:
    return "" if stderr is None else stderr.decode("utf-8", errors="replace").strip()


def _stop_encoder(process: subprocess.Popen[bytes]) -> str:
    if process.poll() is None:
        process.kill()
    if process.stdin is not None:
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
        process.stdin = None
    _, stderr = process.communicate()
    return _decode_stderr(stderr)


def _encode_second_pass(
    *,
    temporary_output: Path,
    ffmpeg: FfmpegInfo,
    fps: float,
    schedule: FrameSchedule,
    expected: Outcome,
    width: int,
    height: int,
    seed: int,
    action_mode: str,
    max_logic_steps: int,
    cell_size: int,
    action_function: ActionFunction,
    reset_options: dict[str, object] | None,
) -> tuple[Outcome, int]:
    input_width = width * cell_size
    input_height = height * cell_size
    command = _ffmpeg_command(
        ffmpeg,
        input_width=input_width,
        input_height=input_height,
        fps=fps,
        output=temporary_output,
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        raise RuntimeError(f"could not start ffmpeg: {exc}") from exc
    if process.stdin is None:
        _stop_encoder(process)
        raise RuntimeError("ffmpeg did not provide a stdin pipe")

    encoder_input = process.stdin
    emitted_ticks: list[int] = []

    def write_frame(tick: int, env: SnakeEnv) -> None:
        frame = env.render()
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("rgb_array rendering did not return an array")
        expected_shape = (input_height, input_width, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise RuntimeError(
                "unexpected RGB frame: "
                f"got shape={frame.shape}, dtype={frame.dtype}; "
                f"expected shape={expected_shape}, dtype=uint8"
            )
        contiguous = np.ascontiguousarray(frame)
        raw_frame = memoryview(contiguous).cast("B")
        try:
            pending = raw_frame
            while pending:
                written = encoder_input.write(pending)
                if written is None or written <= 0:
                    raise RuntimeError("ffmpeg did not accept the complete frame")
                pending = pending[written:]
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg closed its input while encoding") from exc
        emitted_ticks.append(tick)

    try:
        actual = _rollout(
            width=width,
            height=height,
            seed=seed,
            action_mode=action_mode,
            max_logic_steps=max_logic_steps,
            cell_size=cell_size,
            action_function=action_function,
            reset_options=reset_options,
            frame_ticks=frozenset(schedule.ticks),
            frame_function=write_frame,
        )
        encoder_input.close()
        process.stdin = None
        _, stderr_bytes = process.communicate()
    except BaseException:
        _stop_encoder(process)
        raise

    stderr = _decode_stderr(stderr_bytes)
    if process.returncode != 0:
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"ffmpeg exited with status {process.returncode}{detail}")
    if actual != expected:
        raise RuntimeError(
            "second rollout did not exactly match the first "
            f"(first={expected}, second={actual})"
        )
    if tuple(emitted_ticks) != schedule.ticks:
        raise RuntimeError(
            "encoded frame ticks did not match the planned schedule "
            f"(expected {len(schedule.ticks)}, emitted {len(emitted_ticks)})"
        )
    return actual, len(emitted_ticks)


def _write_json_atomic(path: Path, value: dict[str, object]) -> str:
    contents = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return contents


def record_video(
    *,
    output: Path,
    controller: str,
    model_path: Path | None,
    width: int,
    height: int,
    seed: int,
    action_mode: str,
    max_logic_steps: int,
    cell_size: int,
    fps: float,
    dense_start_ticks: int,
    dense_end_ticks: int,
    middle_frames: int,
    device: str,
) -> tuple[dict[str, object], Path]:
    """Run two identical episodes and atomically publish an MP4 and sidecar."""

    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must have an .mp4 suffix")
    output = output.expanduser().resolve()
    sidecar = Path(f"{output}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    action_function, reset_options, resolved_model, model_sha256 = (
        _load_action_function(
            controller,
            width=width,
            height=height,
            action_mode=action_mode,
            model_path=model_path,
            device=device,
        )
    )

    first = _rollout(
        width=width,
        height=height,
        seed=seed,
        action_mode=action_mode,
        max_logic_steps=max_logic_steps,
        cell_size=cell_size,
        action_function=action_function,
        reset_options=reset_options,
    )
    schedule = build_schedule(
        first.ticks,
        dense_start_ticks=dense_start_ticks,
        dense_end_ticks=dense_end_ticks,
        middle_frames=middle_frames,
    )
    ffmpeg = _find_ffmpeg()

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.stem}.",
            suffix=".mp4",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        temporary_output = Path(temporary_name)
        _, encoded_frames = _encode_second_pass(
            temporary_output=temporary_output,
            ffmpeg=ffmpeg,
            fps=fps,
            schedule=schedule,
            expected=first,
            width=width,
            height=height,
            seed=seed,
            action_mode=action_mode,
            max_logic_steps=max_logic_steps,
            cell_size=cell_size,
            action_function=action_function,
            reset_options=reset_options,
        )
        os.replace(temporary_output, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    report: dict[str, object] = {
        "schema_version": 1,
        "controller": controller,
        "model_path": str(resolved_model) if resolved_model is not None else None,
        "model_sha256": model_sha256,
        "seed": seed,
        "width": width,
        "height": height,
        "action_mode": action_mode,
        "max_logic_steps": max_logic_steps,
        "cell_size": cell_size,
        "strict_win": first.strict_win,
        "terminated": first.terminated,
        "truncated": first.truncated,
        "won": first.won,
        "score": first.score,
        "length": first.length,
        "ticks": first.ticks,
        "reason": first.reason,
        "trajectory_sha256": first.trajectory_sha256,
        "schedule": {
            "dense_start_ticks": schedule.dense_start_ticks,
            "dense_end_ticks": schedule.dense_end_ticks,
            "middle_target_frames": schedule.middle_target_frames,
            "middle_frames": schedule.middle_frames,
            "ticks": list(schedule.ticks),
        },
        "frames": encoded_frames,
        "fps": fps,
        "duration_seconds": encoded_frames / fps,
        "video_path": str(output),
        "video_width": width * cell_size + (width * cell_size) % 2,
        "video_height": height * cell_size + (height * cell_size) % 2,
        "ffmpeg": {
            "executable": ffmpeg.executable,
            "source": ffmpeg.source,
            "version": ffmpeg.version,
            "video_codec": "libx264",
            "pixel_format": "yuv420p",
        },
    }
    rendered = _write_json_atomic(sidecar, report)
    sys.stdout.write(rendered)
    return report, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--controller",
        choices=("oracle", "ppo", "qrdqn"),
        default="oracle",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=_nonnegative_int, default=DEFAULT_SEED)
    parser.add_argument(
        "--action-mode",
        choices=("relative", "absolute"),
        default="relative",
    )
    parser.add_argument(
        "--max-logic-steps",
        type=_positive_int,
        default=DEFAULT_MAX_LOGIC_STEPS,
    )
    parser.add_argument(
        "--cell-size",
        type=_positive_int,
        default=DEFAULT_CELL_SIZE,
    )
    parser.add_argument("--fps", type=_positive_float, default=DEFAULT_FPS)
    parser.add_argument(
        "--dense-start-ticks",
        type=_nonnegative_int,
        default=DEFAULT_DENSE_START_TICKS,
    )
    parser.add_argument(
        "--dense-end-ticks",
        type=_nonnegative_int,
        default=DEFAULT_DENSE_END_TICKS,
    )
    parser.add_argument(
        "--middle-frames",
        type=_nonnegative_int,
        default=DEFAULT_MIDDLE_FRAMES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.controller != "oracle" and args.model is None:
        parser.error(f"--model is required with --controller {args.controller}")
    try:
        record_video(
            output=args.output,
            controller=args.controller,
            model_path=args.model,
            width=args.width,
            height=args.height,
            seed=args.seed,
            action_mode=args.action_mode,
            max_logic_steps=args.max_logic_steps,
            cell_size=args.cell_size,
            fps=args.fps,
            dense_start_ticks=args.dense_start_ticks,
            dense_end_ticks=args.dense_end_ticks,
            middle_frames=args.middle_frames,
            device=args.device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Small Linux process-command probes shared by background utilities."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


def _output_directories(arguments: list[str], cwd: Path) -> list[Path]:
    """Resolve ``--output`` values exactly as the target process does."""
    values = []
    for index, argument in enumerate(arguments):
        if argument == "--output" and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif argument.startswith("--output="):
            values.append(argument.split("=", 1)[1])
    result = []
    for value in values:
        path = Path(value)
        result.append((path if path.is_absolute() else cwd / path).resolve())
    return result


def process_command_matches(
    pid: int | None,
    needles: Iterable[bytes],
    directory: Path,
) -> bool:
    """Match a trainer name and its resolved ``--output`` directory."""

    if not pid:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
    except OSError:
        return False
    if not any(needle in command for needle in needles):
        return False
    arguments = [os.fsdecode(argument) for argument in command.split(b"\0") if argument]
    return directory.resolve() in _output_directories(arguments, cwd)

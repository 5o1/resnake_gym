"""Small Linux process-command probes shared by background utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def process_command_matches(
    pid: int | None,
    needles: Iterable[bytes],
    directory: Path,
) -> bool:
    """Match a live process command against a name and exact directory string."""

    if not pid:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return (
        any(needle in command for needle in needles)
        and str(directory).encode() in command
    )

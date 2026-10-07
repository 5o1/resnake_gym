"""Small, algorithm-independent helpers for experiment provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def python_source_hashes(root: Path) -> dict[str, str]:
    """Hash every Python source shipped by the repository invocation."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("src", "scripts")
        for path in sorted((root / folder).rglob("*.py"))
    }


__all__ = ["python_source_hashes"]

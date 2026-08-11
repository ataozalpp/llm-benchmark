from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        process = subprocess.run(["git", *args], capture_output=True, text=True, check=False, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def environment_snapshot() -> dict[str, Any]:
    status = _git(["status", "--porcelain"])
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "project_version": __version__,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "runtime_packages": _runtime_packages(),
        "pyproject_sha256": _file_hash(Path("pyproject.toml")),
    }


def _runtime_packages() -> list[str]:
    try:
        from importlib.metadata import distributions
        return sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata.get("Name"))
    except Exception:
        return []


def _file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

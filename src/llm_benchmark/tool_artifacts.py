"""Incremental tool artifacts with atomic metadata replacement.

Each JSONL event is flushed and fsynced. A killed process may leave an incomplete
last line; readers must ignore that line. Filesystem and DB are not transactional.
Raw prompts, arguments, results, credentials and exception messages are omitted.
"""

import hashlib
import json
import os
from pathlib import Path


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ToolArtifactWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("results.jsonl", "trace.jsonl"):
            (directory / name).touch(exist_ok=False)

    def write_json(self, name: str, value: object) -> None:
        if name not in {"manifest.json", "summary.json"}:
            raise ValueError("Unknown metadata artifact.")
        target = self.directory / name
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)

    def append(self, name: str, value: object) -> None:
        if name not in {"results.jsonl", "trace.jsonl"}:
            raise ValueError("Unknown append artifact.")
        line = canonical_json(value) + "\n"
        with (self.directory / name).open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

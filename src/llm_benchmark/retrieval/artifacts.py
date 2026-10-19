"""Retrieval metadata replacement and incremental result persistence."""

import json
import os
from pathlib import Path


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


class RetrievalArtifactWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "results.jsonl").touch(exist_ok=False)

    def write_json(self, name: str, value: object) -> None:
        if name not in {"manifest.json", "summary.json"}:
            raise ValueError("Unknown retrieval artifact.")
        target = self.directory / name
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)

    def append_result(self, value: object) -> None:
        line = _json(value)
        with (self.directory / "results.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

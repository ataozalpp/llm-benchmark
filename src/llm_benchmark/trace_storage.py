"""Append sanitized execution events to a JSON trace artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .trace import TraceEvent, serialize_trace_event


class JsonlTraceWriter:
    """Write events to a new trace file using a single writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._failed = False

        # The run service owns directory creation.
        # Never overwrite a trace from an existing run.
        with self.path.open("x", encoding="utf-8", newline="\n"):
            pass

    def append(self, event: TraceEvent) -> None:
        if self._failed:
            raise RuntimeError("Trace writer is unavailable after an I/O failure.")

        # Serialize before opening the file so invalid events do not
        # leave partially serialized JSON behind.
        serialized = json.dumps(
            serialize_trace_event(event),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # A failed append may leave an incomplete final line.
            # Do not append further events to that potentially damaged line.
            self._failed = True
            raise

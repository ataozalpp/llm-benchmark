import json
from pathlib import Path

import pytest

from llm_benchmark.trace import (
    InMemoryTraceRecorder,
    TraceEventData,
    TraceEventType,
)
from llm_benchmark.trace_storage import JsonlTraceWriter


def test_writer_creates_empty_trace(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"

    JsonlTraceWriter(path)

    assert path.read_text(encoding="utf-8") == ""


def test_writer_refuses_to_overwrite_existing_trace(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("existing content\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        JsonlTraceWriter(path)

    assert path.read_text(encoding="utf-8") == "existing content\n"


def test_writer_requires_existing_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "trace.jsonl"

    with pytest.raises(FileNotFoundError):
        JsonlTraceWriter(path)

    assert not path.parent.exists()


def test_writer_appends_events_in_call_order(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(path)
    recorder = InMemoryTraceRecorder(clock=lambda: "2026-11-24T10:00:00+00:00")

    first = recorder.record(
        TraceEventType.RETRIEVAL_QUERY,
        run_id="run-1",
        sample_id="query-1",
        data=TraceEventData(requested_k=5),
    )
    second = recorder.record(
        TraceEventType.RETRIEVAL_RESULT,
        run_id="run-1",
        sample_id="query-1",
        data=TraceEventData(retrieved_chunk_count=3),
    )

    writer.append(first)
    writer.append(second)

    content = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]

    assert content.endswith("\n")
    assert len(rows) == 2
    assert [row["sequence"] for row in rows] == [1, 2]
    assert [row["event_type"] for row in rows] == [
        "retrieval_query",
        "retrieval_result",
    ]
    assert rows[0]["data"]["requested_k"] == 5
    assert rows[1]["data"]["retrieved_chunk_count"] == 3


def test_writer_persists_only_sanitized_event_data(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(path)
    event = InMemoryTraceRecorder().record(
        TraceEventType.ERROR,
        run_id="run-1",
        sample_id="soru-ç",
        model=r"C:\private\model.gguf",
        data=TraceEventData(
            provider="api_key=private-credential",
        ),
    )

    writer.append(event)

    content = path.read_text(encoding="utf-8")
    row = json.loads(content)

    assert "private-credential" not in content
    assert row["model"] == "[REDACTED_PATH]"
    assert row["sample_id"] == "soru-ç"
    assert "soru-ç" in content


@pytest.mark.parametrize(
    "data",
    [
        TraceEventData(requested_k=-1),
        TraceEventData(latency_ms=float("nan")),
        TraceEventData(latency_ms=float("inf")),
    ],
)
def test_invalid_event_does_not_change_existing_trace(
    tmp_path: Path,
    data: TraceEventData,
) -> None:
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(path)
    recorder = InMemoryTraceRecorder()

    valid = recorder.record(
        TraceEventType.SCENARIO_STARTED,
        run_id="run-1",
    )
    writer.append(valid)
    before = path.read_bytes()

    invalid = recorder.record(
        TraceEventType.RETRIEVAL_RESULT,
        run_id="run-1",
        data=data,
    )

    with pytest.raises(ValueError):
        writer.append(invalid)

    assert path.read_bytes() == before


def test_io_failure_is_reported_and_disables_further_appends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(path)
    event = InMemoryTraceRecorder().record(
        TraceEventType.SCENARIO_STARTED,
        run_id="run-1",
    )
    original_open = Path.open

    def failing_open(self, mode="r", *args, **kwargs):
        if self == path and mode == "a":
            raise OSError("Simulated storage failure.")
        return original_open(self, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing_open)

        with pytest.raises(OSError, match="Simulated storage failure"):
            writer.append(event)

    with pytest.raises(RuntimeError, match="I/O failure"):
        writer.append(event)

    assert path.read_text(encoding="utf-8") == ""

import json
from dataclasses import FrozenInstanceError

import pytest

from llm_benchmark.storage import write_jsonl
from llm_benchmark.trace import (
    MAX_TRACE_STRING_LENGTH,
    TRACE_EVENT_DATA_FIELDS,
    TRACE_EVENT_FIELDS,
    InMemoryTraceRecorder,
    TraceEvent,
    TraceEventData,
    TraceEventType,
    serialize_trace_event,
)


def test_in_memory_trace_recorder_preserves_event_order() -> None:
    recorder = InMemoryTraceRecorder(
        clock=lambda: "2026-09-01T10:00:00+00:00"
    )

    first = recorder.record(
        TraceEventType.SCENARIO_STARTED,
        run_id="run-1",
        sample_id="sample-1",
        model="model-1",
    )
    second = recorder.record(
        TraceEventType.MODEL_REQUEST,
        run_id="run-1",
        sample_id="sample-1",
        model="model-1",
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert recorder.events() == (first, second)


def test_trace_recorder_uses_injected_clock() -> None:
    recorder = InMemoryTraceRecorder(
        clock=lambda: "2026-09-01T10:00:00+00:00"
    )

    event = recorder.record(
        TraceEventType.SCENARIO_STARTED,
        run_id="run-1",
    )

    assert event.timestamp == "2026-09-01T10:00:00+00:00"


def test_trace_event_and_typed_data_are_frozen() -> None:
    data = TraceEventData(request_status="succeeded")
    event = InMemoryTraceRecorder().record(
        TraceEventType.MODEL_RESPONSE,
        run_id="run-1",
        data=data,
    )

    with pytest.raises(FrozenInstanceError):
        event.sequence = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        data.request_status = "failed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("identifier", "field"),
    [
        (r"C:\Users\person\sample.jsonl", "sample_id"),
        ("/home/person/model.gguf", "model"),
    ],
)
def test_trace_redacts_absolute_path_identifiers(
    identifier: str,
    field: str,
) -> None:
    event = InMemoryTraceRecorder().record(
        TraceEventType.SCENARIO_STARTED,
        run_id="run-1",
        sample_id=identifier if field == "sample_id" else "sample-1",
        model=identifier if field == "model" else "model-1",
    )

    assert getattr(event, field) == "[REDACTED_PATH]"
    assert identifier not in json.dumps(event.to_dict())


def test_trace_event_serializes_typed_data() -> None:
    event = InMemoryTraceRecorder().record(
        TraceEventType.ERROR,
        run_id="run-1",
        data=TraceEventData(
            stage="provider_execution",
            error_type="RuntimeError",
        ),
    )

    serialized = event.to_dict()

    assert serialized["event_type"] == "error"
    assert serialized["data"] == {
        "error_type": "RuntimeError",
        "evaluation_status": None,
        "final_output_tokens": None,
        "input_tokens": None,
        "is_correct": None,
        "latency_ms": None,
        "output_budget_provenance": None,
        "parse_status": None,
        "provider": None,
        "reasoning_mode": None,
        "reasoning_output_tokens": None,
        "request_status": None,
        "stage": "provider_execution",
        "task_type": None,
        "time_to_first_token_ms": None,
        "tokens_per_second": None,
        "total_output_tokens": None,
        "total_tokens": None,
    }


def test_safe_serializer_uses_exact_allowlists_and_preserves_normal_values() -> None:
    event = TraceEvent(
        sequence=1,
        event_type=TraceEventType.TASK_EVALUATION,
        timestamp="2026-09-01T10:00:00+00:00",
        run_id="run-1",
        sample_id="sample-1",
        model="model-1",
        data=TraceEventData(
            provider="mock",
            task_type="multiple_choice",
            request_status="succeeded",
            parse_status="normalized_label",
            evaluation_status="correct",
            is_correct=True,
        ),
    )

    serialized = serialize_trace_event(event)

    assert set(serialized) == TRACE_EVENT_FIELDS
    assert set(serialized["data"]) == TRACE_EVENT_DATA_FIELDS
    assert serialized["sample_id"] == "sample-1"
    assert serialized["model"] == "model-1"
    assert serialized["data"]["provider"] == "mock"
    assert serialized["data"]["request_status"] == "succeeded"
    assert serialized["data"]["evaluation_status"] == "correct"


@pytest.mark.parametrize(
    "unsafe_value",
    [
        r"C:\Users\person\private\model.gguf",
        "/home/person/private/model.gguf",
        "failed at C:\\Users\\person\\private\\data.jsonl",
        "failed at /home/person/private/data.jsonl",
    ],
)
def test_directly_constructed_event_paths_are_redacted(unsafe_value: str) -> None:
    event = TraceEvent(
        sequence=1,
        event_type=TraceEventType.ERROR,
        timestamp="2026-09-01T10:00:00+00:00",
        run_id="run-1",
        sample_id=unsafe_value,
        model=unsafe_value,
        data=TraceEventData(error_type=unsafe_value),
    )

    serialized = serialize_trace_event(event)
    serialized_text = json.dumps(serialized)

    assert unsafe_value not in serialized_text
    assert serialized["sample_id"] == "[REDACTED_PATH]"
    assert serialized["model"] == "[REDACTED_PATH]"
    assert serialized["data"]["error_type"] == "[REDACTED_PATH]"


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "api_key=credential-value",
        "authorization: credential-value",
        "Authorization: Bearer credential-value",
        "Bearer credential-value",
        "token = credential-value",
        "password: credential-value",
        "secret=credential-value",
    ],
)
def test_directly_constructed_event_credentials_are_redacted(
    unsafe_value: str,
) -> None:
    event = TraceEvent(
        sequence=1,
        event_type=TraceEventType.ERROR,
        timestamp="2026-09-01T10:00:00+00:00",
        run_id=unsafe_value,
        sample_id=unsafe_value,
        model=unsafe_value,
        data=TraceEventData(
            provider=unsafe_value,
            task_type=unsafe_value,
            error_type=unsafe_value,
        ),
    )

    serialized_text = json.dumps(serialize_trace_event(event)).lower()

    assert "credential-value" not in serialized_text


def test_safe_serializer_bounds_every_persisted_string() -> None:
    oversized = "x" * (MAX_TRACE_STRING_LENGTH + 100)
    event = TraceEvent(
        sequence=1,
        event_type=TraceEventType.ERROR,
        timestamp=oversized,
        run_id=oversized,
        sample_id=oversized,
        model=oversized,
        data=TraceEventData(provider=oversized, error_type=oversized),
    )

    serialized = serialize_trace_event(event)

    assert len(serialized["timestamp"]) == MAX_TRACE_STRING_LENGTH
    assert len(serialized["run_id"]) == MAX_TRACE_STRING_LENGTH
    assert len(serialized["sample_id"]) == MAX_TRACE_STRING_LENGTH
    assert len(serialized["model"]) == MAX_TRACE_STRING_LENGTH
    assert len(serialized["data"]["provider"]) == MAX_TRACE_STRING_LENGTH
    assert len(serialized["data"]["error_type"]) == MAX_TRACE_STRING_LENGTH
    assert serialized["run_id"].endswith("...")


def test_safe_serializer_rejects_non_trace_values() -> None:
    with pytest.raises(TypeError, match="TraceEvent and TraceEventData"):
        serialize_trace_event(object())  # type: ignore[arg-type]


def test_write_jsonl_preserves_utf8_and_cleans_temporary_file(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"

    write_jsonl(path, [{"message": "Türkçe doğrulama"}])

    assert path.read_text(encoding="utf-8") == (
        '{"message": "Türkçe doğrulama"}\n'
    )
    assert not path.with_suffix(".jsonl.tmp").exists()

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

MAX_TRACE_STRING_LENGTH = 256
TRACE_EVENT_FIELDS = frozenset(
    {"sequence", "event_type", "timestamp", "run_id", "sample_id", "model", "data"}
)
TRACE_EVENT_DATA_FIELDS = frozenset(
    {
        "provider",
        "task_type",
        "reasoning_mode",
        "output_budget_provenance",
        "request_status",
        "input_tokens",
        "total_output_tokens",
        "reasoning_output_tokens",
        "final_output_tokens",
        "total_tokens",
        "latency_ms",
        "time_to_first_token_ms",
        "tokens_per_second",
        "parse_status",
        "evaluation_status",
        "is_correct",
        "stage",
        "error_type",
    }
)

_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)\bauthorization\b(?:\s*[:=]\s*|\s+)(?:bearer\s+)?[^\s,;]+"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:api[-_ ]?key|token|password|secret)\b\s*[:=]\s*[^\s,;]+"
)
_WINDOWS_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\)"
)
_POSIX_PATH_PATTERN = re.compile(r"(?:^|[\s\"'])/(?!/)")


class TraceEventType(StrEnum):
    SCENARIO_STARTED = "scenario_started"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TASK_EVALUATION = "task_evaluation"
    SCENARIO_COMPLETED = "scenario_completed"
    ERROR = "error"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TraceEventData:
    provider: str | None = None
    task_type: str | None = None
    reasoning_mode: str | None = None
    output_budget_provenance: str | None = None
    request_status: str | None = None
    input_tokens: int | None = None
    total_output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    final_output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    tokens_per_second: float | None = None
    parse_status: str | None = None
    evaluation_status: str | None = None
    is_correct: bool | None = None
    stage: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    event_type: TraceEventType
    timestamp: str
    run_id: str
    sample_id: str | None
    model: str | None
    data: TraceEventData

    def to_dict(self) -> dict[str, object]:
        return serialize_trace_event(self)


class TraceRecorder(Protocol):
    def record(
        self,
        event_type: TraceEventType,
        *,
        run_id: str,
        sample_id: str | None = None,
        model: str | None = None,
        data: TraceEventData | None = None,
    ) -> TraceEvent:
        """Record and return one ordered trace event."""
        ...

    def events(self) -> tuple[TraceEvent, ...]:
        """Return recorded events in deterministic sequence order."""
        ...


TraceRecorderFactory = Callable[[], TraceRecorder]


class InMemoryTraceRecorder:
    def __init__(
        self,
        *,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self._clock = clock
        self._events: list[TraceEvent] = []

    def record(
        self,
        event_type: TraceEventType,
        *,
        run_id: str,
        sample_id: str | None = None,
        model: str | None = None,
        data: TraceEventData | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            sequence=len(self._events) + 1,
            event_type=event_type,
            timestamp=self._clock(),
            run_id=run_id,
            sample_id=_sanitize_identifier(sample_id),
            model=_sanitize_identifier(model),
            data=data or TraceEventData(),
        )
        self._events.append(event)
        return event

    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)


def _sanitize_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
        return "[REDACTED_PATH]"
    return value


def serialize_trace_event(event: TraceEvent) -> dict[str, object]:
    """Return the fixed, sanitized trace artifact representation."""

    if type(event) is not TraceEvent or type(event.data) is not TraceEventData:
        raise TypeError("trace artifact values must use TraceEvent and TraceEventData")
    if type(event.sequence) is not int or event.sequence < 1:
        raise TypeError("trace event sequence must be a positive integer")
    if not isinstance(event.event_type, TraceEventType):
        raise TypeError("trace event type is invalid")

    data = event.data
    serialized_data: dict[str, object] = {
        "provider": _sanitize_optional_string(data.provider),
        "task_type": _sanitize_optional_string(data.task_type),
        "reasoning_mode": _sanitize_optional_string(data.reasoning_mode),
        "output_budget_provenance": _sanitize_optional_string(
            data.output_budget_provenance
        ),
        "request_status": _sanitize_optional_string(data.request_status),
        "input_tokens": _optional_integer(data.input_tokens),
        "total_output_tokens": _optional_integer(data.total_output_tokens),
        "reasoning_output_tokens": _optional_integer(data.reasoning_output_tokens),
        "final_output_tokens": _optional_integer(data.final_output_tokens),
        "total_tokens": _optional_integer(data.total_tokens),
        "latency_ms": _optional_number(data.latency_ms),
        "time_to_first_token_ms": _optional_number(data.time_to_first_token_ms),
        "tokens_per_second": _optional_number(data.tokens_per_second),
        "parse_status": _sanitize_optional_string(data.parse_status),
        "evaluation_status": _sanitize_optional_string(data.evaluation_status),
        "is_correct": _optional_boolean(data.is_correct),
        "stage": _sanitize_optional_string(data.stage),
        "error_type": _sanitize_optional_string(data.error_type),
    }
    if set(serialized_data) != TRACE_EVENT_DATA_FIELDS:
        raise AssertionError("trace event data allowlist is inconsistent")

    serialized: dict[str, object] = {
        "sequence": event.sequence,
        "event_type": _sanitize_string(event.event_type.value),
        "timestamp": _sanitize_string(event.timestamp),
        "run_id": _sanitize_string(event.run_id),
        "sample_id": _sanitize_optional_string(event.sample_id),
        "model": _sanitize_optional_string(event.model),
        "data": serialized_data,
    }
    if set(serialized) != TRACE_EVENT_FIELDS:
        raise AssertionError("trace event allowlist is inconsistent")
    return serialized


def _sanitize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("trace string fields must contain strings")
    return _sanitize_string(value)


def _sanitize_string(value: str) -> str:
    sanitized = _AUTHORIZATION_PATTERN.sub("[REDACTED]", value)
    sanitized = _SECRET_ASSIGNMENT_PATTERN.sub("[REDACTED]", sanitized)
    sanitized = _BEARER_PATTERN.sub("[REDACTED]", sanitized)
    if (
        PureWindowsPath(sanitized).is_absolute()
        or PurePosixPath(sanitized).is_absolute()
        or _WINDOWS_PATH_PATTERN.search(sanitized)
        or _POSIX_PATH_PATTERN.search(sanitized)
    ):
        return "[REDACTED_PATH]"
    if len(sanitized) > MAX_TRACE_STRING_LENGTH:
        return sanitized[: MAX_TRACE_STRING_LENGTH - 3] + "..."
    return sanitized


def _optional_integer(value: int | None) -> int | None:
    if value is not None and type(value) is not int:
        raise TypeError("trace integer fields must contain integers")
    return value


def _optional_number(value: float | None) -> int | float | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise TypeError("trace numeric fields must contain numbers")
    return value


def _optional_boolean(value: bool | None) -> bool | None:
    if value is not None and type(value) is not bool:
        raise TypeError("trace boolean fields must contain booleans")
    return value

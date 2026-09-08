"""Pure normalization of supplied OpenAI-compatible tool-call responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .tool_runtime import InvalidJsonValueError, ToolCall


class ToolCallNormalizationErrorCode(StrEnum):
    INVALID_RESPONSE = ("invalid_response",)
    INVALID_CONTENT = ("invalid_content",)
    INVALID_FINISH_REASON = ("invalid_finish_reason",)
    EMPTY_TURN = ("empty_turn",)
    INVALID_TOOL_CALL = ("invalid_tool_call",)
    UNSUPPORTED_TOOL_TYPE = ("unsupported_tool_type",)
    DUPLICATE_CALL_ID = ("duplicate_call_id",)
    INVALID_ARGUMENTS = ("invalid_arguments",)
    DUPLICATE_ARGUMENT_KEY = ("duplicate_argument_key",)
    ARGUMENTS_TOO_LARGE = ("arguments_too_large",)
    TOTAL_ARGUMENTS_TOO_LARGE = ("total_arguments_too_large",)


class ToolCallNormalizationError(ValueError):
    def __init__(self, code: ToolCallNormalizationErrorCode) -> None:
        if type(code) is not ToolCallNormalizationErrorCode:
            raise TypeError("Invalid normalization error code.")

        self.code = code
        super().__init__(f"Tool-call normalization failed: {code.value}.")


class _Rejected(Exception):
    """Internal control signal; never carries provider payloads."""

    def __init__(self, code: ToolCallNormalizationErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True)
class NormalizationToolTurn:
    content: str | None = field(repr=False)
    tool_calls: tuple[ToolCall, ...] = field(repr=False)
    finish_reason: str | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.content is not None and type(self.content) is not str:
            raise TypeError("Content must be a string or null.")

        if self.finish_reason is not None and type(self.finish_reason) is not str:
            raise TypeError("Finish reason must be a string or null.")

        if type(self.tool_calls) is not tuple:
            raise TypeError("Tool calls must be a tuple.")

        if any(type(call) is not ToolCall for call in self.tool_calls):
            raise TypeError("Tool calls must use the ToolCall contract.")

        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Tool call IDs must be unique within a turn.")

        has_content = self.content is not None and bool(self.content.strip())
        if not has_content and not self.tool_calls:
            raise ValueError("A turn must contain text or tool calls.")


def _argument_byte_size(text: str, limit: int) -> int:
    total = 0

    for start in range(0, len(text), 1024):
        try:
            chunk_size = len(text[start : start + 1024].encode("utf-8"))
        except UnicodeError:
            raise _Rejected(ToolCallNormalizationErrorCode.INVALID_ARGUMENTS) from None

        total += chunk_size

        if total > limit:
            raise _Rejected(ToolCallNormalizationErrorCode.ARGUMENTS_TOO_LARGE)

    return total


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}

    for key, value in pairs:
        if key in result:
            raise _Rejected(ToolCallNormalizationErrorCode.DUPLICATE_ARGUMENT_KEY)
        result[key] = value

    return result


def _reject_constant(value: str) -> object:
    del value
    raise _Rejected(ToolCallNormalizationErrorCode.INVALID_ARGUMENTS)


def _parse_arguments(text: str) -> dict[str, object]:
    try:
        arguments = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError, OverflowError):
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_ARGUMENTS) from None

    if type(arguments) is not dict:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_ARGUMENTS)

    return arguments


def _normalize_call(
    raw_call: object,
    *,
    max_arguments_bytes: int,
    remaining_arguments_bytes: int,
) -> tuple[ToolCall, int]:
    if type(raw_call) is not dict:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_TOOL_CALL)

    tool_type = raw_call.get("type")

    if type(tool_type) is not str:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_TOOL_CALL)
    if tool_type != "function":
        raise _Rejected(ToolCallNormalizationErrorCode.UNSUPPORTED_TOOL_TYPE)

    function = raw_call.get("function")
    if type(function) is not dict:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_TOOL_CALL)

    call_id = raw_call.get("id")
    name = function.get("name")
    arguments_text = function.get("arguments")

    if type(call_id) is not str or type(name) is not str:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_TOOL_CALL)

    if type(arguments_text) is not str:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_ARGUMENTS)

    size = _argument_byte_size(arguments_text, max_arguments_bytes)

    if size > remaining_arguments_bytes:
        raise _Rejected(ToolCallNormalizationErrorCode.TOTAL_ARGUMENTS_TOO_LARGE)

    arguments = _parse_arguments(arguments_text)

    try:
        call = ToolCall(
            call_id=call_id,
            tool_name=name,
            arguments=arguments,
        )
    except InvalidJsonValueError:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_ARGUMENTS) from None
    except (TypeError, ValueError):
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_TOOL_CALL) from None

    return call, size


def _normalize_response(
    body: object,
    *,
    max_arguments_bytes: int,
    max_total_arguments_bytes: int,
) -> NormalizationToolTurn:
    if type(body) is not dict:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_RESPONSE)

    choices = body.get("choices")

    if type(choices) is not list or len(choices) != 1:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_RESPONSE)

    choice = choices[0]
    if type(choice) is not dict:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_RESPONSE)

    message = choice.get("message")
    if type(message) is not dict:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_RESPONSE)

    role = message.get("role")
    if type(role) is not str or role != "assistant":
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_RESPONSE)

    content = message.get("content")
    if content is not None and type(content) is not str:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_CONTENT)

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and type(finish_reason) is not str:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_FINISH_REASON)

    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []

    if type(raw_calls) is not list:
        raise _Rejected(ToolCallNormalizationErrorCode.INVALID_TOOL_CALL)

    calls: list[ToolCall] = []
    seen_ids: set[str] = set()
    total_size = 0

    for raw_call in raw_calls:
        call, size = _normalize_call(
            raw_call,
            max_arguments_bytes=max_arguments_bytes,
            remaining_arguments_bytes=max_total_arguments_bytes - total_size,
        )

        if call.call_id in seen_ids:
            raise _Rejected(ToolCallNormalizationErrorCode.DUPLICATE_CALL_ID)

        seen_ids.add(call.call_id)
        total_size += size
        calls.append(call)

    if not calls and (content is None or not content.strip()):
        raise _Rejected(ToolCallNormalizationErrorCode.EMPTY_TURN)

    return NormalizationToolTurn(
        content=content,
        tool_calls=tuple(calls),
        finish_reason=finish_reason,
    )


def normalize_openai_tool_response(
    body: object,
    *,
    max_arguments_bytes: int = 65_536,
    max_total_arguments_bytes: int = 262_144,
) -> NormalizationToolTurn:
    """Normalize one assistant choice without executing or resolving tools.

    Limits apply to original argument strings encoded as UTF-8, before JSON
    parsing, per call and in aggregate. They do not bound the response body,
    content, or handler execution. Any malformed call rejects the whole turn.
    """
    for limit in (max_arguments_bytes, max_total_arguments_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Argument byte limits must be positive integers.")

    try:
        return _normalize_response(
            body,
            max_arguments_bytes=max_arguments_bytes,
            max_total_arguments_bytes=max_total_arguments_bytes,
        )
    except _Rejected as exc:
        error_code = exc.code

    # Raise outside the handler so the public error retains no parser context.
    raise ToolCallNormalizationError(error_code)

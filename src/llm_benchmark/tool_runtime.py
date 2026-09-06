from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ValidationError

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]

_MAX_TOOL_NAME_LENGTH = 64
_MAX_TOOL_DESCRIPTION_LENGTH = 512
_MAX_TOOL_CALL_ID_LENGTH = 128
_DEFAULT_MAX_OUTPUT_BYTES = 65_536

_TOOL_NAME_PATTERN = re.compile(
    rf"^[A-Za-z][A-Za-z0-9_-]{{0,{_MAX_TOOL_NAME_LENGTH - 1}}}$"
)
_TOOL_CALL_ID_PATTERN = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_-]{{0,{_MAX_TOOL_CALL_ID_LENGTH - 1}}}$"
)


class ToolRuntimeError(Exception):
    """Base error for tool-runtime setup failures."""


class DuplicateToolError(ToolRuntimeError):
    """Raised when a tool name is registered more than once."""


class ToolRegistrationError(ToolRuntimeError):
    """Raised when a tool registration is invalid."""


class InvalidJsonValueError(ValueError):
    """Raised when a value is outside the strict JSON domain."""


class ToolExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_FAILED = "execution_failed"
    INVALID_OUTPUT = "invalid_output"
    OUTPUT_TOO_LARGE = "output_too_large"


class ToolErrorCode(StrEnum):
    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    INVALID_TOOL_OUTPUT = "invalid_tool_output"
    TOOL_OUTPUT_TOO_LARGE = "tool_output_too_large"


_ERROR_CODE_BY_STATUS: dict[ToolExecutionStatus, ToolErrorCode | None] = {
    ToolExecutionStatus.SUCCEEDED: None,
    ToolExecutionStatus.TOOL_NOT_FOUND: ToolErrorCode.TOOL_NOT_FOUND,
    ToolExecutionStatus.INVALID_ARGUMENTS: ToolErrorCode.INVALID_ARGUMENTS,
    ToolExecutionStatus.EXECUTION_FAILED: ToolErrorCode.TOOL_EXECUTION_FAILED,
    ToolExecutionStatus.INVALID_OUTPUT: ToolErrorCode.INVALID_TOOL_OUTPUT,
    ToolExecutionStatus.OUTPUT_TOO_LARGE: ToolErrorCode.TOOL_OUTPUT_TOO_LARGE,
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if type(self.description) is not str or not self.description.strip():
            raise ValueError("Tool description must not be empty.")
        if len(self.description) > _MAX_TOOL_DESCRIPTION_LENGTH:
            raise ValueError(
                "Tool description must not exceed "
                f"{_MAX_TOOL_DESCRIPTION_LENGTH} characters."
            )


@dataclass(frozen=True, init=False)
class ToolCall:
    call_id: str
    tool_name: str
    _arguments_json: str = field(repr=False)

    def __init__(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: object,
    ) -> None:
        _validate_tool_call_id(call_id)
        _validate_tool_name(tool_name)
        if type(arguments) is not dict:
            raise TypeError("Tool arguments must be a JSON object.")

        normalized, snapshot, _ = _create_json_snapshot(arguments)
        if type(normalized) is not dict:
            raise TypeError("Tool arguments must be a JSON object.")

        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "_arguments_json", snapshot)

    @property
    def arguments(self) -> JsonObject:
        value = json.loads(self._arguments_json)
        if type(value) is not dict:
            raise AssertionError("Stored tool arguments must be an object.")
        return cast(JsonObject, value)


@dataclass(frozen=True, init=False)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolExecutionStatus
    error_code: ToolErrorCode | None
    _output_json: str = field(repr=False)

    def __init__(
        self,
        *,
        call_id: str,
        tool_name: str,
        status: ToolExecutionStatus,
        output: object = None,
        error_code: ToolErrorCode | None = None,
    ) -> None:
        _validate_tool_call_id(call_id)
        _validate_tool_name(tool_name)
        if type(status) is not ToolExecutionStatus:
            raise TypeError("Invalid tool execution status.")
        if error_code is not None and type(error_code) is not ToolErrorCode:
            raise TypeError("Invalid tool result error code.")
        if error_code is not _ERROR_CODE_BY_STATUS[status]:
            raise ValueError("Tool result status and error code must match.")
        if status is not ToolExecutionStatus.SUCCEEDED and output is not None:
            raise ValueError("Unsuccessful tool results cannot contain output.")

        _, snapshot, _ = _create_json_snapshot(output)
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "_output_json", snapshot)

    @property
    def output(self) -> JsonValue:
        return cast(JsonValue, json.loads(self._output_json))


ToolHandler = Callable[[BaseModel], object]


@dataclass(frozen=True)
class ToolRegistration:
    definition: ToolDefinition
    argument_model: type[BaseModel]
    handler: ToolHandler

    def __post_init__(self) -> None:
        if type(self.definition) is not ToolDefinition:
            raise ToolRegistrationError(
                "Tool definition must use the ToolDefinition contract."
            )
        try:
            is_model = issubclass(self.argument_model, BaseModel)
        except TypeError as exc:
            raise ToolRegistrationError(
                "Tool argument model must be a Pydantic model."
            ) from exc
        if not is_model:
            raise ToolRegistrationError("Tool argument model must be a Pydantic model.")
        if not callable(self.handler):
            raise ToolRegistrationError("Tool handler must be callable.")

        model_config = self.argument_model.model_config
        if model_config.get("strict") is not True:
            raise ToolRegistrationError(
                "Tool argument model must use strict validation."
            )
        if model_config.get("extra") != "forbid":
            raise ToolRegistrationError("Tool argument model must forbid extra fields.")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}

    def register(self, registration: ToolRegistration) -> None:
        if type(registration) is not ToolRegistration:
            raise ToolRegistrationError(
                "Tool registry accepts ToolRegistration values only."
            )
        name = registration.definition.name
        if name in self._tools:
            raise DuplicateToolError("Tool is already registered.")
        self._tools[name] = registration

    def get(self, name: str) -> ToolRegistration | None:
        return self._tools.get(name)

    def list_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name].definition for name in sorted(self._tools))


class ToolRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        if type(registry) is not ToolRegistry:
            raise TypeError("registry must be a ToolRegistry.")
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer.")
        self._registry = registry
        self._max_output_bytes = max_output_bytes

    def execute(self, call: ToolCall) -> ToolResult:
        if type(call) is not ToolCall:
            raise TypeError("Tool runtime accepts ToolCall values only.")

        registration = self._registry.get(call.tool_name)
        if registration is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.TOOL_NOT_FOUND,
                error_code=ToolErrorCode.TOOL_NOT_FOUND,
            )

        try:
            arguments = registration.argument_model.model_validate(call.arguments)
        except ValidationError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.INVALID_ARGUMENTS,
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
            )

        try:
            output = registration.handler(arguments)
        except Exception:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.EXECUTION_FAILED,
                error_code=ToolErrorCode.TOOL_EXECUTION_FAILED,
            )

        try:
            normalized_output, _, encoded_output = _create_json_snapshot(output)
        except InvalidJsonValueError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.INVALID_OUTPUT,
                error_code=ToolErrorCode.INVALID_TOOL_OUTPUT,
            )

        if len(encoded_output) > self._max_output_bytes:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.OUTPUT_TOO_LARGE,
                error_code=ToolErrorCode.TOOL_OUTPUT_TOO_LARGE,
            )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.SUCCEEDED,
            output=normalized_output,
        )


def _validate_tool_name(value: object) -> None:
    if type(value) is not str or not _TOOL_NAME_PATTERN.fullmatch(value):
        raise ValueError("Invalid tool name.")


def _validate_tool_call_id(value: object) -> None:
    if type(value) is not str or not _TOOL_CALL_ID_PATTERN.fullmatch(value):
        raise ValueError("Invalid tool call ID.")


def _normalize_json_value(value: object) -> JsonValue:
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is str:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidJsonValueError("JSON numbers must be finite.")
        return value
    if type(value) is list:
        return [_normalize_json_value(item) for item in value]
    if type(value) is dict:
        normalized: JsonObject = {}
        for key, item in value.items():
            if type(key) is not str:
                raise InvalidJsonValueError("JSON object keys must be strings.")
            normalized[key] = _normalize_json_value(item)
        return normalized
    raise InvalidJsonValueError("Value is not strict JSON-compatible.")


def _create_json_snapshot(value: object) -> tuple[JsonValue, str, bytes]:
    try:
        normalized = _normalize_json_value(value)
        snapshot = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        encoded = snapshot.encode("utf-8")
    except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
        if isinstance(exc, InvalidJsonValueError):
            raise
        raise InvalidJsonValueError("Value is not strict JSON-compatible.") from exc
    return normalized, snapshot, encoded

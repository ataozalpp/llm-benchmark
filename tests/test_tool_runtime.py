from __future__ import annotations

import os
import subprocess
import sys
from collections import UserDict
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from llm_benchmark.tool_runtime import (
    DuplicateToolError,
    InvalidJsonValueError,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolRegistration,
    ToolRegistrationError,
    ToolRegistry,
    ToolResult,
    ToolRuntime,
)


class AddArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    a: int
    b: int


class NonStrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class ExtraAllowedArguments(BaseModel):
    model_config = ConfigDict(strict=True)

    value: int


class FatalToolSignal(BaseException):
    pass


def add_handler(arguments: BaseModel) -> int:
    assert isinstance(arguments, AddArguments)
    return arguments.a + arguments.b


def create_registration(
    *,
    name: str = "add_numbers",
    handler: object = add_handler,
    argument_model: object = AddArguments,
) -> ToolRegistration:
    return ToolRegistration(
        definition=ToolDefinition(
            name=name,
            description=f"Execute {name} deterministically.",
        ),
        argument_model=argument_model,  # type: ignore[arg-type]
        handler=handler,  # type: ignore[arg-type]
    )


def create_runtime(
    *,
    handler: object = add_handler,
    max_output_bytes: int = 65_536,
) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(create_registration(handler=handler))
    return ToolRuntime(registry, max_output_bytes=max_output_bytes)


def valid_call(arguments: object | None = None) -> ToolCall:
    return ToolCall(
        call_id="call-1",
        tool_name="add_numbers",
        arguments={"a": 2, "b": 3} if arguments is None else arguments,
    )


def test_executes_registered_tool() -> None:
    result = create_runtime().execute(valid_call())

    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.output == 5
    assert result.error_code is None


def test_unknown_tool_is_normalized_outcome() -> None:
    result = create_runtime().execute(
        ToolCall(
            call_id="call-1",
            tool_name="missing_tool",
            arguments={},
        )
    )

    assert result.status is ToolExecutionStatus.TOOL_NOT_FOUND
    assert result.output is None
    assert result.error_code == "tool_not_found"


@pytest.mark.parametrize(
    "arguments",
    [
        {"a": 2},
        {"a": "2", "b": 3},
        {"a": 2, "b": 3, "extra": 4},
    ],
)
def test_invalid_arguments_are_rejected_without_calling_handler(
    arguments: dict[str, object],
) -> None:
    call_count = 0

    def recording_handler(validated: BaseModel) -> int:
        nonlocal call_count
        call_count += 1
        return add_handler(validated)

    result = create_runtime(handler=recording_handler).execute(valid_call(arguments))

    assert result.status is ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.output is None
    assert result.error_code == "invalid_arguments"
    assert call_count == 0


@pytest.mark.parametrize(
    "name",
    [
        "",
        "1calculator",
        "tool name",
        "tool/name",
        r"tool\name",
        "tool.name",
        "a" * 65,
    ],
)
def test_invalid_tool_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid tool name"):
        ToolDefinition(name=name, description="Valid description.")


def test_tool_name_accepts_64_characters() -> None:
    name = "a" + ("b" * 63)

    definition = ToolDefinition(name=name, description="Valid description.")

    assert definition.name == name


@pytest.mark.parametrize("description", ["", "   ", None])
def test_empty_or_non_string_description_is_rejected(description: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ToolDefinition(
            name="calculator",
            description=description,  # type: ignore[arg-type]
        )


def test_description_accepts_512_characters() -> None:
    definition = ToolDefinition(name="calculator", description="a" * 512)

    assert len(definition.description) == 512


def test_description_rejects_more_than_512_characters() -> None:
    with pytest.raises(ValueError, match="must not exceed 512"):
        ToolDefinition(name="calculator", description="a" * 513)


def test_tool_definition_is_frozen() -> None:
    definition = ToolDefinition(name="calculator", description="Calculate.")

    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "call_id",
    [
        "",
        "call id",
        r"C:\Users\private\file.txt",
        "/home/private/file.txt",
        "Authorization:Bearer-secret",
        "api_key=secret",
        "a" * 129,
    ],
)
def test_invalid_tool_call_ids_are_rejected(call_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid tool call ID"):
        ToolCall(
            call_id=call_id,
            tool_name="add_numbers",
            arguments={"a": 2, "b": 3},
        )


def test_tool_call_id_accepts_128_characters() -> None:
    call_id = "a" * 128

    call = ToolCall(
        call_id=call_id,
        tool_name="add_numbers",
        arguments={"a": 2, "b": 3},
    )

    assert call.call_id == call_id


def test_tool_call_snapshots_nested_arguments() -> None:
    arguments = {"items": [{"value": 1}]}
    call = ToolCall(
        call_id="call-1",
        tool_name="example_tool",
        arguments=arguments,
    )

    arguments["items"][0]["value"] = 999
    returned = call.arguments
    returned["items"][0]["value"] = 500

    assert call.arguments == {"items": [{"value": 1}]}


def test_tool_call_is_frozen() -> None:
    call = valid_call()

    with pytest.raises(FrozenInstanceError):
        call.call_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "arguments",
    [
        UserDict({"a": 2, "b": 3}),
        {"value": (1, 2)},
        {"value": {1, 2}},
        {"value": b"bytes"},
        {"value": object()},
        {1: "value"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
    ],
)
def test_tool_call_rejects_non_strict_json_arguments(arguments: object) -> None:
    with pytest.raises((InvalidJsonValueError, TypeError)):
        ToolCall(
            call_id="call-1",
            tool_name="example_tool",
            arguments=arguments,
        )


def test_duplicate_tool_is_rejected() -> None:
    registry = ToolRegistry()
    registration = create_registration()
    registry.register(registration)

    with pytest.raises(DuplicateToolError):
        registry.register(registration)


def test_registry_lists_definitions_deterministically_by_name() -> None:
    registry = ToolRegistry()
    for name in ("z_tool", "a_tool", "m_tool"):
        registry.register(create_registration(name=name))

    assert [definition.name for definition in registry.list_definitions()] == [
        "a_tool",
        "m_tool",
        "z_tool",
    ]


@pytest.mark.parametrize(
    ("argument_model", "handler"),
    [
        (object, add_handler),
        (NonStrictArguments, add_handler),
        (ExtraAllowedArguments, add_handler),
        (AddArguments, None),
    ],
)
def test_invalid_tool_registrations_are_rejected(
    argument_model: object,
    handler: object,
) -> None:
    with pytest.raises(ToolRegistrationError):
        create_registration(argument_model=argument_model, handler=handler)


def test_handler_exception_is_normalized_without_sensitive_details() -> None:
    secret = "handler-secret-value"
    local_path = r"C:\Users\private\tool.txt"
    stack_sentinel = "STACK_TRACE_SENTINEL"

    def failing_handler(arguments: BaseModel) -> object:
        del arguments
        raise RuntimeError(f"password={secret} {local_path} {stack_sentinel}")

    result = create_runtime(handler=failing_handler).execute(valid_call())
    serialized = repr(result)

    assert result.status is ToolExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "tool_execution_failed"
    assert result.output is None
    assert secret not in serialized
    assert local_path not in serialized
    assert stack_sentinel not in serialized


def test_handler_base_exception_is_not_caught() -> None:
    def fatal_handler(arguments: BaseModel) -> object:
        del arguments
        raise FatalToolSignal

    with pytest.raises(FatalToolSignal):
        create_runtime(handler=fatal_handler).execute(valid_call())


@pytest.mark.parametrize(
    "output",
    [
        (1, 2),
        {1, 2},
        b"bytes",
        object(),
        {1: "value"},
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_strict_json_output_is_normalized_as_invalid(output: object) -> None:
    def output_handler(arguments: BaseModel) -> object:
        del arguments
        return output

    result = create_runtime(handler=output_handler).execute(valid_call())

    assert result.status is ToolExecutionStatus.INVALID_OUTPUT
    assert result.error_code == "invalid_tool_output"
    assert result.output is None


def test_oversized_output_is_normalized_without_returning_output() -> None:
    def output_handler(arguments: BaseModel) -> str:
        del arguments
        return "abcdef"

    result = create_runtime(
        handler=output_handler,
        max_output_bytes=7,
    ).execute(valid_call())

    assert result.status is ToolExecutionStatus.OUTPUT_TOO_LARGE
    assert result.error_code == "tool_output_too_large"
    assert result.output is None


def test_unicode_output_uses_utf8_byte_limit() -> None:
    def output_handler(arguments: BaseModel) -> str:
        del arguments
        return "ş"

    succeeded = create_runtime(
        handler=output_handler,
        max_output_bytes=4,
    ).execute(valid_call())
    too_large = create_runtime(
        handler=output_handler,
        max_output_bytes=3,
    ).execute(valid_call())

    assert succeeded.status is ToolExecutionStatus.SUCCEEDED
    assert succeeded.output == "ş"
    assert too_large.status is ToolExecutionStatus.OUTPUT_TOO_LARGE


def test_tool_result_snapshots_nested_output() -> None:
    output = {"items": [{"value": 1}]}
    result = ToolResult(
        call_id="call-1",
        tool_name="example_tool",
        status=ToolExecutionStatus.SUCCEEDED,
        output=output,
    )

    output["items"][0]["value"] = 999
    returned = result.output
    assert isinstance(returned, dict)
    returned["items"][0]["value"] = 500

    assert result.output == {"items": [{"value": 1}]}


def test_tool_result_is_frozen() -> None:
    result = ToolResult(
        call_id="call-1",
        tool_name="example_tool",
        status=ToolExecutionStatus.SUCCEEDED,
        output=1,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = ToolExecutionStatus.EXECUTION_FAILED  # type: ignore[misc]


def test_tool_result_rejects_arbitrary_error_code() -> None:
    with pytest.raises(TypeError, match="Invalid tool result error code"):
        ToolResult(
            call_id="call-1",
            tool_name="example_tool",
            status=ToolExecutionStatus.EXECUTION_FAILED,
            error_code="password=secret-value",  # type: ignore[arg-type]
        )


def test_tool_result_rejects_status_error_code_mismatch() -> None:
    with pytest.raises(ValueError, match="status and error code must match"):
        ToolResult(
            call_id="call-1",
            tool_name="example_tool",
            status=ToolExecutionStatus.TOOL_NOT_FOUND,
            error_code=ToolErrorCode.INVALID_ARGUMENTS,
        )


@pytest.mark.parametrize("max_output_bytes", [0, -1, True, False, 1.5, "100"])
def test_invalid_output_byte_limits_are_rejected(max_output_bytes: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ToolRuntime(
            ToolRegistry(),
            max_output_bytes=max_output_bytes,  # type: ignore[arg-type]
        )


def test_importing_tool_runtime_has_no_runtime_side_effects(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("LLM_BENCHMARK_DATABASE_URL", None)

    completed = subprocess.run(
        [sys.executable, "-c", "import llm_benchmark.tool_runtime"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "outputs").exists()
    assert not list(tmp_path.glob("*.db"))
    assert not list(tmp_path.glob("*.sqlite*"))

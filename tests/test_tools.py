from __future__ import annotations

import pytest

from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolRegistration,
    ToolRuntime,
)
from llm_benchmark.tools import create_example_tool_registry


@pytest.fixture
def runtime() -> ToolRuntime:
    return ToolRuntime(create_example_tool_registry())


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected"),
    [
        ("add", 2, 3, 5),
        ("subtract", 2, 3, -1),
        ("multiply", 17, 23, 391),
        ("divide", 7, 2, 3.5),
        ("multiply", -4, 3, -12),
        ("add", 0, 0, 0),
        ("add", 1_000_000, -1_000_000, 0),
    ]
)
def test_calculator_operations(
    runtime:ToolRuntime,
    operation: str,
    left: int,
    right: int,
    expected: int | float,
) -> None:
    result = runtime.execute(
        ToolCall(
            call_id="calculator-1",
            tool_name="calculator",
            arguments={
                "operation": operation,
                "left": left,
                "right": right,
            },
        )
    )

    assert result.call_id == "calculator-1"
    assert result.tool_name == "calculator"
    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.error_code is None
    assert result.output == {"result": expected}


@pytest.mark.parametrize(
    "arguments",
    [
        {"operation": "power", "left": 2, "right": 3},
        {"operation": "add", "left": "2", "right": 3},
        {"operation": "add", "left": True, "right": 3},
        {"operation": "add", "left": 2.5, "right": 3},
        {"operation": "add", "left": 2},
        {"operation": "add", "left": 2, "right": 3, "extra": 1},
        {"operation": "add", "left": 1_000_001, "right": 0},
        {"operation": "add", "left": 0, "right": -1_000_001},
    ],
)
def test_calculator_rejects_invalid_arguments(
    runtime: ToolRuntime,
    arguments: dict[str, object],
) -> None:
    result = runtime.execute(
        ToolCall(
            call_id="calculator-invalid",
            tool_name="calculator",
            arguments=arguments,
        )
    )

    assert result.status is ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert result.output is None


def test_division_by_zero_is_normalized(runtime: ToolRuntime) -> None:
    result = runtime.execute(
        ToolCall(
            call_id="division-zero",
            tool_name="calculator",
            arguments={
                "operation": "divide",
                "left": 7,
                "right": 0,
            },
        )
    )

    assert result.status is ToolExecutionStatus.EXECUTION_FAILED
    assert result.error_code is ToolErrorCode.TOOL_EXECUTION_FAILED
    assert result.output is None
    assert "Division by zero" not in repr(result)


@pytest.mark.parametrize(
    ("city", "expected_code"),
    [
        ("northport", "SYN-001"),
        ("rivergate", "SYN-002"),
        ("sunvale", "SYN-003"),
        ("unknown_city", None),
    ],
)
def test_city_lookup(
    runtime: ToolRuntime,
    city: str,
    expected_code: str | None
) -> None:
    result = runtime.execute(
        ToolCall(
            call_id="lookup-1",
            tool_name="lookup_city_code",
            arguments={"city": city},
        )
    )

    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.error_code is None
    assert result.output == {
        "found": expected_code is not None,
        "code": expected_code,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"city": 123},
        {"city": ""},
        {"city": "Northport"},
        {"city": "north port"},
        {"city": "a" * 65},
        {"city": "northport", "extra": True},
    ],
)
def test_city_lookup_rejects_invalid_arguments(
    runtime: ToolRuntime,
    arguments: dict[str, object],
) -> None:
    result = runtime.execute(
        ToolCall(
            call_id="lookup-invalid",
            tool_name="lookup_city_code",
            arguments=arguments,
        )
    )

    assert result.status is ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert result.output is None


def test_repeated_calls_return_the_same_result(
    runtime: ToolRuntime,
) -> None:
    call = ToolCall(
        call_id="repeat-1",
        tool_name="calculator",
        arguments={
            "operation": "multiply",
            "left": 17,
            "right": 23,
        },
    )

    assert runtime.execute(call) == runtime.execute(call)


def test_registry_factories_are_independent() -> None:
    first = create_example_tool_registry()
    second = create_example_tool_registry()

    assert first is not second
    assert [item.name for item in second.list_definitions()] == [
        "calculator",
        "lookup_city_code",
    ]

    calculator = first.get("calculator")
    assert calculator is not None

    first.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="extra_calculator",
                description="An additional test registration.",
            ),
            argument_model=calculator.argument_model,
            handler=calculator.handler,
        )
    )

    assert first.get("extra_calculator") is not None
    assert second.get("extra_calculator") is None

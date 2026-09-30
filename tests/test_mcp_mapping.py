import pytest

from llm_benchmark.external_tool_schema import (
    ExternalToolSchemaCode,
    ExternalToolSchemaError,
)
from llm_benchmark.mcp_mapping import (
    descriptor_from_mcp_fields,
    result_from_mcp_fields,
)
from llm_benchmark.tool_runtime import ToolCall, ToolExecutionStatus
from llm_benchmark.tools import CalculatorArguments


def make_call():
    return ToolCall(
        call_id="call-1",
        tool_name="calculator",
        arguments={
            "operation": "multiply",
            "left": 17,
            "right": 23,
        },
    )


def test_calculator_descriptor_preserves_schema():
    schema = CalculatorArguments.model_json_schema()

    descriptor = descriptor_from_mcp_fields(
        name="calculator",
        description="Synthetic calculator.",
        input_schema=schema,
    )

    assert descriptor.definition.name == "calculator"
    assert descriptor.parameters == schema


def test_missing_description_uses_fallback():
    descriptor = descriptor_from_mcp_fields(
        name="calculator",
        description=None,
        input_schema={"type": "object"},
    )

    assert descriptor.definition.description == "MCP tool: calculator"


def test_unsupported_schema_is_rejected():
    with pytest.raises(ExternalToolSchemaError) as caught:
        descriptor_from_mcp_fields(
            name="calculator",
            description=None,
            input_schema={
                "type": "object",
                "$ref": "#/$defs/Arguments",
            },
        )

        assert caught.value.code is ExternalToolSchemaCode.UNSUPPORTED_SCHEMA


def test_success_preserves_call_identity_and_output():
    call = make_call()

    result = result_from_mcp_fields(
        call=call,
        is_error=False,
        structured_content={"result": 391},
    )

    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.call_id == call.call_id
    assert result.tool_name == call.tool_name
    assert result.output == {"result": 391}


def test_tool_error_does_not_preserve_error_paylaod():
    result = result_from_mcp_fields(
        call=make_call(),
        is_error=True,
        structured_content={"secret": "must-not-be-returned"},
    )

    assert result.status is ToolExecutionStatus.EXECUTION_FAILED
    assert result.output is None


@pytest.mark.parametrize(
    "output",
    [None, "391", [], {"result": float("nan")}],
)
def unsupported_or_invalid_output_is_rejected(output):
    result = result_from_mcp_fields(
        call=make_call(),
        is_error=False,
        structured_content=output,
    )

    assert result.status is ToolExecutionStatus.INVALID_OUTPUT


def test_output_budget_uses_utf8_bytes():
    result = result_from_mcp_fields(
        call=make_call(),
        is_error=False,
        structured_content={"result": "ş"},
        max_output_bytes=14,
    )

    assert result.status is ToolExecutionStatus.OUTPUT_TOO_LARGE


@pytest.mark.parametrize("budget", [0, -1, True])
def test_invalid_output_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        result_from_mcp_fields(
            call=make_call(),
            is_error=False,
            structured_content={"result": 391},
            max_output_bytes=budget,
        )
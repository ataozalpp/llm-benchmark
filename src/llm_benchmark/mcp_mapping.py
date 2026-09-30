"""Pure mappings for the initial structured-JSON MCP tool profile."""

import json

from .external_tool_schema import validate_external_tool_schema
from .tool_descriptors import ToolDescriptor
from .tool_runtime import (
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolResult,
)


def descriptor_from_mcp_fields(
    *,
    name: str,
    description: str | None,
    input_schema: object,
) -> ToolDescriptor:
    descriptor = ToolDescriptor(
        definition=ToolDefinition(
            name=name,
            description=description or f"MCP tool: {name}",
        ),
        parameters=input_schema,
    )
    validate_external_tool_schema(descriptor)
    return descriptor


def result_from_mcp_fields(
    *,
    call: ToolCall,
    is_error: bool | None,
    structured_content: object,
    max_output_bytes: int = 65_536,
) -> ToolResult:
    """Map already decoded fields; transport limits belong to the client."""

    if type(call) is not ToolCall:
        raise TypeError("Expected a ToolCall.")

    if is_error is not None and type(is_error) is not bool:
        raise TypeError("MCP error flag must be boolean or null.")

    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("Output budget must be a positive integer.")

    if is_error is True:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.EXECUTION_FAILED,
            error_code=ToolErrorCode.TOOL_EXECUTION_FAILED,
        )

    if type(structured_content) is not dict:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.INVALID_OUTPUT,
            error_code=ToolErrorCode.INVALID_TOOL_OUTPUT,
        )

    try:
        result = ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.SUCCEEDED,
            output=structured_content,
        )
    except (TypeError, ValueError):
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.INVALID_OUTPUT,
            error_code=ToolErrorCode.INVALID_TOOL_OUTPUT,
        )

    encoded = json.dumps(
        result.output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    if len(encoded) > max_output_bytes:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.OUTPUT_TOO_LARGE,
            error_code=ToolErrorCode.TOOL_OUTPUT_TOO_LARGE,
        )

    return result
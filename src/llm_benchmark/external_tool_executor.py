"""Validate external tool arguments before delegating execution."""

from __future__ import annotations

from .external_tool_schema import (
    ExternalToolSchemaCode,
    ExternalToolSchemaError,
    validate_external_tool_arguments,
    validate_external_tool_schema,
)
from .tool_descriptors import ToolDescriptor
from .tool_execution import ToolExecutor
from .tool_runtime import (
    ToolCall,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolResult,
)


class ValidatingExternalToolExecutor:
    """Validate selected-tool calls; delegate owns output limits and resources.

    This wrapper neither provides sandboxing nor closes its delegate. Unexpected
    exceptions propagate; only argument-schema mismatch is a normalized outcome.
    """

    def __init__(
        self,
        *,
        descriptors: tuple[ToolDescriptor, ...],
        delegate: ToolExecutor,
    ) -> None:
        if type(descriptors) is not tuple:
            raise TypeError("Tool descriptors must be a tuple.")
        if not descriptors:
            raise ValueError("A non-empty descriptor tuple is required.")

        if not callable(getattr(delegate, "execute", None)):
            raise TypeError("Delegate must support tool execution.")

        by_name: dict[str, ToolDescriptor] = {}

        for descriptor in descriptors:
            validate_external_tool_schema(descriptor)
            name = descriptor.definition.name

            if name in by_name:
                raise ValueError("Tool descriptor names must be unique.")

            by_name[name] = descriptor

        self._descriptors = by_name
        self._delegate = delegate

    def execute(self, call: ToolCall) -> ToolResult:
        if type(call) is not ToolCall:
            raise TypeError("Expected a ToolCall.")

        descriptor = self._descriptors.get(call.tool_name)

        if descriptor is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.TOOL_NOT_FOUND,
                error_code=ToolErrorCode.TOOL_NOT_FOUND,
            )

        try:
            validate_external_tool_arguments(
                descriptor=descriptor,
                call=call,
            )
        except ExternalToolSchemaError as error:
            if error.code is not ExternalToolSchemaCode.INVALID_ARGUMENTS:
                raise

            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.INVALID_ARGUMENTS,
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
            )

        result = self._delegate.execute(call)

        if type(result) is not ToolResult:
            raise TypeError("Delegate returned an invalid result.")

        if result.call_id != call.call_id or result.tool_name != call.tool_name:
            raise ValueError("Delegate result does not match the call.")

        return result

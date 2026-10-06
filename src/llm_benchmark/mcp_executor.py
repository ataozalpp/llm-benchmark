"""Synchronous tool execution backed by an existing async MCP session."""

import math

import anyio
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from .mcp_errors import (
    McpExecutionTimeout,
    MCPProtocolFailure,
    MCPTransportFailure,
)
from .mcp_mapping import result_from_mcp_fields
from .tool_runtime import ToolCall, ToolResult


class MCPToolExecutor:
    """Execute through an existing client from an anyio worker thread.
    
    The caller owns the client lifetime and argument validation.
    Known request failures become infrastructure exceptions.
    """

    def __init__(
        self,
        *,
        client: Client,
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 65_536,
    ) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Timeout must be a finite positive number.")

        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError("Output budget must be a positive integer.")

        self._client = client
        self._timeout_seconds = float(timeout_seconds)
        self._max_output_bytes = max_output_bytes

    def execute(self, call: ToolCall) -> ToolResult:
        if type(call) is not ToolCall:
            raise TypeError("Expected a ToolCall.")

        return anyio.from_thread.run(self._execute_async, call)

    async def _execute_async(self, call: ToolCall) -> ToolResult:
        try:
            with anyio.fail_after(self._timeout_seconds):
                response = await self._client.call_tool(
                    call.tool_name,
                    call.arguments,
                    read_timeout_seconds=self._timeout_seconds,
                )

        except TimeoutError:
            raise McpExecutionTimeout(
                "MCP tool request timed out."
            ) from None

        except MCPError as error:
            if error.code == REQUEST_TIMEOUT:
                raise McpExecutionTimeout(
                    "MCP tool request timed out."
                ) from None

            if error.code == CONNECTION_CLOSED:
                raise MCPTransportFailure(
                    "MCP connection closed during tool execution."
                ) from None

            raise MCPProtocolFailure(code=error.code) from None

        except (
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            anyio.EndOfStream,
            ConnectionError,
        ):
            raise MCPTransportFailure(
                "MCP transport failed during tool execution."
            ) from None

        return result_from_mcp_fields(
            call=call,
            is_error=response.is_error,
            structured_content=response.structured_content,
            max_output_bytes=self._max_output_bytes,
        )
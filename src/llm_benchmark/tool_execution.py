"""Framework independent contract for synchronous tool execution."""

from __future__ import annotations

from typing import Protocol

from .tool_runtime import ToolCall, ToolResult


class ToolExecutor(Protocol):
    """Execute one tool call and return its normalized result.

    Implementations own argument validation, output limits and
    expected execution-error normalization.

    Unexpected exceptions may propagate. This contract does not
    provide sandboxing, cancellation or transport behaviour.
    """

    def execute(self, call: ToolCall) -> ToolResult:
        ...

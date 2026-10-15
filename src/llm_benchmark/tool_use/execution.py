"""Framework independent contract for synchronous tool execution."""

from __future__ import annotations

from typing import Protocol

from llm_benchmark.tool_use.runtime import ToolCall, ToolResult


class ToolExecutor(Protocol):
    """Execute one tool call and return its normalized result.

    Implementations own argument validation, output limits and
    expected execution-error normalization.

    Unexpected exceptions may propagate. This contract does not
    provide sandboxing, cancellation or transport behaviour.
    """

    def execute(self, call: ToolCall) -> ToolResult:
        ...


class ToolExecutionTimeout(RuntimeError):
    """A tool request exceeded its execution time budget."""


class TimedToolExecutor(ToolExecutor, Protocol):
    """Optional support for a per-call time budget."""

    def execute_with_timeout(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float,
    ) -> ToolResult:
        ...
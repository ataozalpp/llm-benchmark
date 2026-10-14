"""Run benchmark scenarios against the local calculator MCP server."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import anyio

from .external_tool_policy import ExternalToolPolicy
from .external_tool_suite import run_external_tool_suite
from .mcp_runtime import prepare_calculator_runtime
from .tool_evaluation import ToolEvaluationResult
from .tool_execution import ToolExecutor
from .tool_loop import ConversationProvider, ToolLoopResult
from .tool_requests import ToolConversationRequest
from .tool_scenarios import ToolScenario
from .tool_suite import ToolSuiteResult


async def run_calculator_mcp_suite(
    *,
    scenarios: tuple[ToolScenario, ...],
    provider_factory: Callable[[], ConversationProvider],
    server_python: Path,
    project_root: Path,
    on_case_completed: (
        Callable[[ToolScenario, ToolLoopResult, ToolEvaluationResult], None] | None
    ) = None,
    observer: Callable[[str, str, dict[str, object]], None] | None = None,
) -> ToolSuiteResult:
    async with prepare_calculator_runtime(
        server_python=server_python,
        project_root=project_root,
    ) as runtime:
        def executor_factory(
            request: ToolConversationRequest,
        ) -> ToolExecutor:
            return runtime.executor

        run_suite = partial(
            run_external_tool_suite,
            scenarios=scenarios,
            descriptors=runtime.descriptors,
            policy=ExternalToolPolicy(
                allowed_tool_names=("calculator",),
                max_tools=1,
                max_total_schema_bytes=65_536,
            ),
            provider_factory=provider_factory,
            executor_factory=executor_factory,
            on_case_completed=on_case_completed,
            observer=observer,
        )

        return await anyio.to_thread.run_sync(run_suite)

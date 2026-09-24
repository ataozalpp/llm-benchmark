"""Run descriptor-based suites after external catalog admission."""

from __future__ import annotations

from collections.abc import Callable

from .external_tool_policy import (
    ExternalToolPolicy,
    admit_tool_descriptors,
)
from .tool_descriptors import ToolDescriptor
from .tool_execution import ToolExecutor
from .tool_loop import ConversationProvider
from .tool_requests import ToolConversationRequest
from .tool_scenarios import ToolScenario
from .tool_suite import ToolSuiteResult, run_tool_suite


def run_external_tool_suite(
    *,
    scenarios: tuple[ToolScenario, ...],
    descriptors: tuple[ToolDescriptor, ...],
    policy: ExternalToolPolicy,
    provider_factory: Callable[[], ConversationProvider],
    executor_factory: Callable[[ToolConversationRequest], ToolExecutor],
) -> ToolSuiteResult:
    """Admit the catalog before the suite calls execution factories.

    This is not schema-semantic validation or a resource-lifecycle manager.
    Provider and executor resource ownership remains with the caller.
    """

    admitted = admit_tool_descriptors(
        descriptors=descriptors,
        policy=policy,
    )

    return run_tool_suite(
        scenarios=scenarios,
        descriptors=admitted,
        provider_factory=provider_factory,
        executor_factory=executor_factory,
    )

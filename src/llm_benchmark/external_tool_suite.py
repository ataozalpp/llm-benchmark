"""Run descriptor-based suites after external catalog admission."""

from __future__ import annotations

from collections.abc import Callable

from .external_tool_executor import ValidatingExternalToolExecutor
from .external_tool_policy import (
    ExternalToolPolicy,
    admit_tool_descriptors,
)
from .external_tool_schema import validate_external_tool_schema
from .tool_descriptors import ToolDescriptor
from .tool_evaluation import ToolEvaluationResult
from .tool_execution import ToolExecutor
from .tool_loop import ConversationProvider, ToolLoopResult
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
    on_case_completed: Callable[[ToolScenario, ToolLoopResult, ToolEvaluationResult], None] | None = None,
    observer: Callable[[str, str, dict[str, object]], None] | None = None,
) -> ToolSuiteResult:
    """Admit the catalog before the suite calls execution factories.

    All admitted schemas are checked against the supported external profile.
    Selected-tool arguments are validated before delegate execution.
    Provider and executor resource ownership remains with the caller.
    """

    admitted = admit_tool_descriptors(
        descriptors=descriptors,
        policy=policy,
    )

    for descriptor in admitted:
        validate_external_tool_schema(descriptor)

    if not callable(executor_factory):
        raise TypeError("Descriptor suites require an executor factory.")

    def validated_executor_factory(request: ToolConversationRequest) -> ToolExecutor:
        delegate = executor_factory(request)
        return ValidatingExternalToolExecutor(
            descriptors=request.descriptors,
            delegate=delegate,
        )

    return run_tool_suite(
        scenarios=scenarios,
        descriptors=admitted,
        provider_factory=provider_factory,
        executor_factory=validated_executor_factory,
        on_case_completed=on_case_completed,
        observer=observer,
    )

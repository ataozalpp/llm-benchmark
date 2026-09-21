"""Sequential execution of synthetic tool-evaluation scenarios."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .tool_descriptors import ToolDescriptor
from .tool_evaluation import (
    ToolEvaluationResult,
    evaluate_tool_loop,
)
from .tool_execution import ToolExecutor
from .tool_loop import ConversationProvider, run_tool_loop
from .tool_reporting import (
    ToolEvaluationSummary,
    aggregate_tool_evaluations,
)
from .tool_requests import ToolConversationRequest
from .tool_runtime import ToolRegistry
from .tool_scenario_requests import (
    build_descriptor_scenario_request,
    build_tool_scenario_request,
)
from .tool_scenarios import ToolScenario


@dataclass(frozen=True)
class ToolSuiteResult:
    evaluations: tuple[ToolEvaluationResult, ...] = field(repr=False)

    def __post_init__(self) -> None:
        aggregate_tool_evaluations(self.evaluations)

    @property
    def summary(self) -> ToolEvaluationSummary:
        return aggregate_tool_evaluations(self.evaluations)


def run_tool_suite(
    *,
    scenarios: tuple[ToolScenario, ...],
    registry: ToolRegistry | None = None,
    provider_factory: Callable[[], ConversationProvider],
    descriptors: tuple[ToolDescriptor, ...] = (),
    executor_factory: (Callable[[ToolConversationRequest], ToolExecutor] | None) = None,
) -> ToolSuiteResult:
    """Prepare all requests, then execute sequentially.

    Normalized loop outcomes are evaluated; unexpected exceptions propagate
    without returning a partial suite. Factories own provider/executor isolation
    and resource lifetimes; this function does not close acquired resources.
    All requests are prepared before either factory is called.
    """

    if type(scenarios) is not tuple:
        raise TypeError("Scenarios must be a tuple.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    if any(type(scenario) is not ToolScenario for scenario in scenarios):
        raise TypeError("Invalid scenario.")

    if type(descriptors) is not tuple:
        raise TypeError("Tool descriptors must be a tuple.")

    if registry is not None:
        if type(registry) is not ToolRegistry:
            raise TypeError("Expected a ToolRegistry.")

        if descriptors:
            raise ValueError("Select exactly one suite tool source.")

        if executor_factory is not None:
            raise ValueError("Local suite execution uses the default runtime.")
    else:
        if not descriptors:
            raise ValueError("A suite tool source is required.")

        if not callable(executor_factory):
            raise TypeError("Descriptor suites require an executor factory.")

    if not callable(provider_factory):
        raise TypeError("Provider factory must be callable.")

    case_ids = tuple(scenario.case_id for scenario in scenarios)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Scenario case IDs must be unique within a suite.")

    if registry is not None:
        requests = tuple(
            build_tool_scenario_request(
                scenario=scenario,
                registry=registry,
            )
            for scenario in scenarios
        )
    else:
        requests = tuple(
            build_descriptor_scenario_request(
                scenario=scenario,
                descriptors=descriptors,
            )
            for scenario in scenarios
        )

    evaluations: list[ToolEvaluationResult] = []

    for scenario, request in zip(
        scenarios,
        requests,
        strict=True,
    ):
        executor: ToolExecutor | None = None

        if registry is None:
            if executor_factory is None:
                raise AssertionError(
                    "Validated descriptor suite requires an executor factory."
                )

            executor = executor_factory(request)

            if not callable(getattr(executor, "execute", None)):
                raise TypeError("Executor must support tool execution.")

        provider = provider_factory()

        if not callable(getattr(provider, "generate_tool_conversation", None)):
            raise TypeError("Provider must support tool conversations.")

        loop_result = run_tool_loop(
            provider=provider,
            request=request,
            policy=scenario.policy,
            executor=executor,
        )

        evaluation = evaluate_tool_loop(
            case=scenario.evaluation,
            result=loop_result,
        )

        evaluations.append(evaluation)

    return ToolSuiteResult(
        evaluations=tuple(evaluations),
    )

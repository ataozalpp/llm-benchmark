"""Sequential execution of synthetic tool-evaluation scenarios."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .tool_evaluation import (
    ToolEvaluationResult,
    evaluate_tool_loop,
)
from .tool_loop import ConversationProvider, run_tool_loop
from .tool_reporting import (
    ToolEvaluationSummary,
    aggregate_tool_evaluations,
)
from .tool_runtime import ToolRegistry
from .tool_scenario_requests import build_tool_scenario_request
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
    registry: ToolRegistry,
    provider_factory: Callable[[], ConversationProvider],
) -> ToolSuiteResult:
    """Prepare all requests, then execute sequentially.

    Normalized loop outcomes are evaluated; unexpected exceptions propagate
    without returning a partial suite. Factories own provider isolation.
    """

    if type(scenarios) is not tuple:
        raise TypeError("Scenarios must be a tuple.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    if any(type(scenario) is not ToolScenario for scenario in scenarios):
        raise TypeError("Invalid scenario.")

    if type(registry) is not ToolRegistry:
        raise TypeError("Expected a ToolRegistry.")

    if not callable(provider_factory):
        raise TypeError("Provider factory must be callable.")

    case_ids = tuple(scenario.case_id for scenario in scenarios)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Scenario case IDs must be unique within a suite.")

    requests = tuple(
        build_tool_scenario_request(
            scenario=scenario,
            registry=registry,
        )
        for scenario in scenarios
    )

    evaluations: list[ToolEvaluationResult] = []

    for scenario, request in zip(
        scenarios,
        requests,
        strict=True,
    ):
        provider = provider_factory()

        if not callable(getattr(provider, "generate_tool_conversation", None)):
            raise TypeError("Provider must support tool conversations.")

        loop_result = run_tool_loop(
            provider=provider,
            request=request,
            policy=scenario.policy,
        )

        evaluation = evaluate_tool_loop(
            case=scenario.evaluation,
            result=loop_result,
        )

        evaluations.append(evaluation)

    return ToolSuiteResult(
        evaluations=tuple(evaluations),
    )

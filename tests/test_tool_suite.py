from dataclasses import FrozenInstanceError, replace

import pytest

from llm_benchmark.tool_calling import NormalizationToolTurn
from llm_benchmark.tool_provider_models import (
    ToolProviderErrorCode,
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_reporting import MetricCount
from llm_benchmark.tool_runtime import ToolCall
from llm_benchmark.tool_scenarios import (
    create_example_tool_scenarios,
)
from llm_benchmark.tool_suite import ToolSuiteResult, run_tool_suite
from llm_benchmark.tools import create_example_tool_registry


class ScriptedProvider:
    def __init__(
        self,
        results: tuple[ToolProviderResult, ...],
    ) -> None:
        self.results = results
        self.requests = []

    def generate_tool_conversation(self, request):
        index = len(self.requests)
        self.requests.append(request)
        return self.results[index]


def successful_turn(
    *,
    content: str | None,
    calls: tuple[ToolCall, ...] = (),
    finish_reason: str = "stop",
) -> ToolProviderResult:
    return ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=NormalizationToolTurn(
            content=content,
            tool_calls=calls,
            finish_reason=finish_reason,
        ),
    )


def test_example_suite_executes_and_reports() -> None:
    scripts = (
        (
            successful_turn(
                content=None,
                calls=(
                    ToolCall(
                        call_id="calculator_1",
                        tool_name="calculator",
                        arguments={
                            "operation": "multiply",
                            "left": 17,
                            "right": 23,
                        },
                    ),
                ),
                finish_reason="tool_calls",
            ),
            successful_turn(content="391"),
        ),
        (successful_turn(content="READY"),),
    )

    created: list[ScriptedProvider] = []

    def factory() -> ScriptedProvider:
        provider = ScriptedProvider(scripts[len(created)])
        created.append(provider)
        return provider

    result = run_tool_suite(
        scenarios=create_example_tool_scenarios(),
        registry=create_example_tool_registry(),
        provider_factory=factory,
    )

    assert len(created) == 2
    assert created[0] is not created[1]

    assert len(created[0].requests) == 2
    assert len(created[1].requests) == 1

    assert len(created[0].requests[0].conversation.messages) == 1
    assert len(created[1].requests[0].conversation.messages) == 1

    assert tuple(item.case_id for item in result.evaluations) == (
        "calculator-multiply",
        "direct-answer",
    )

    summary = result.summary

    assert summary.case_count == 2
    assert summary.completion == MetricCount(2, 2)
    assert summary.tool_sequence_match == MetricCount(2, 2)
    assert summary.argument_match == MetricCount(2, 2)
    assert summary.final_answer_match == MetricCount(2, 2)

    assert summary.requested_call_count == 1
    assert summary.executed_call_count == 1
    assert summary.successful_call_count == 1


def test_all_tools_are_resolved_before_provider_creation() -> None:

    original = create_example_tool_scenarios()

    invalid_second = replace(
        original[1],
        available_tools=("missing_tool",),
    )

    def forbidden_factory():
        pytest.fail("Provider must not be created before all requests are ready.")

    with pytest.raises(ValueError, match="not registered"):
        run_tool_suite(
            scenarios=(original[0], invalid_second),
            registry=create_example_tool_registry(),
            provider_factory=forbidden_factory,
        )


@pytest.mark.parametrize("value", [None, [], (object(),), ()])
def test_invalid_scenarios_do_not_create_provider(value) -> None:
    def forbidden():
        pytest.fail("Unexpected factory call")

    with pytest.raises((TypeError, ValueError)):
        run_tool_suite(
            scenarios=value,
            registry=create_example_tool_registry(),
            provider_factory=forbidden,
        )


def test_duplicate_ids_are_rejected_before_factory() -> None:
    scenario = create_example_tool_scenarios()[0]

    def forbidden():
        pytest.fail("Unexpected factory call")

    with pytest.raises(ValueError, match="unique"):
        run_tool_suite(
            scenarios=(scenario, scenario),
            registry=create_example_tool_registry(),
            provider_factory=forbidden,
        )


@pytest.mark.parametrize("field", ["registry", "provider_factory"])
def test_invalid_dependencies(field) -> None:
    def forbidden():
        pytest.fail("Unexpected factory call")

    values = dict(
        scenarios=create_example_tool_scenarios(),
        registry=create_example_tool_registry(),
        provider_factory=forbidden,
    )
    values[field] = None
    with pytest.raises(TypeError):
        run_tool_suite(**values)


@pytest.mark.parametrize(
    "value",
    [None, object(), type("InvalidProvider", (), {"generate_tool_conversation": 1})()],
)
def test_factory_must_return_conversation_provider(value) -> None:
    attempts = []

    def factory():
        attempts.append(1)
        return value

    with pytest.raises(TypeError, match="tool conversations"):
        run_tool_suite(
            scenarios=create_example_tool_scenarios(),
            registry=create_example_tool_registry(),
            provider_factory=factory,
        )
    assert attempts == [1]


def test_normalized_failure_continues_to_next_case() -> None:
    failure = ToolProviderResult(
        status=ToolProviderStatus.REQUEST_FAILED,
        error_code=ToolProviderErrorCode.TIMEOUT,
    )
    providers = [
        ScriptedProvider((failure,)),
        ScriptedProvider((successful_turn(content="READY"),)),
    ]
    pending = iter(providers)
    result = run_tool_suite(
        scenarios=create_example_tool_scenarios(),
        registry=create_example_tool_registry(),
        provider_factory=lambda: next(pending),
    )
    assert [len(p.requests) for p in providers] == [1, 1]
    assert result.evaluations[0].completed is False
    assert result.evaluations[0].final_answer_match is None
    assert result.evaluations[1].final_answer_match is True
    assert result.summary.completion == MetricCount(1, 2)
    assert result.summary.final_answer_match == MetricCount(1, 1)
    assert result.summary.final_unavailable_count == 1


@pytest.mark.parametrize("source", ["factory", "provider"])
@pytest.mark.parametrize(
    "error", [RuntimeError("private detail"), KeyboardInterrupt(), SystemExit()]
)
def test_unexpected_errors_propagate_without_continuing(source, error) -> None:
    attempts = []

    class BrokenProvider:
        def generate_tool_conversation(self, request):
            raise error

    def factory():
        attempts.append(1)
        if source == "factory":
            raise error
        return BrokenProvider()

    with pytest.raises(type(error)) as caught:
        run_tool_suite(
            scenarios=create_example_tool_scenarios(),
            registry=create_example_tool_registry(),
            provider_factory=factory,
        )
    assert caught.value is error
    assert attempts == [1]


def test_result_is_frozen_and_summary_cannot_drift() -> None:
    scenario = create_example_tool_scenarios()[1]
    result = run_tool_suite(
        scenarios=(scenario,),
        registry=create_example_tool_registry(),
        provider_factory=lambda: ScriptedProvider((successful_turn(content="READY"),)),
    )
    with pytest.raises(FrozenInstanceError):
        result.evaluations = ()
    assert result.summary == result.summary
    assert "direct-answer" not in repr(result)
    with pytest.raises(ValueError):
        ToolSuiteResult(())
    with pytest.raises(TypeError):
        ToolSuiteResult(list(result.evaluations))
    with pytest.raises(TypeError):
        ToolSuiteResult((object(),))
    with pytest.raises(ValueError, match="unique"):
        ToolSuiteResult(result.evaluations * 2)


def test_repeated_suites_preserve_inputs_and_have_fresh_conversations() -> None:
    scenario = create_example_tool_scenarios()[1]
    registry = create_example_tool_registry()
    definitions = registry.list_definitions()
    providers = []

    def factory():
        provider = ScriptedProvider((successful_turn(content="READY"),))
        providers.append(provider)
        return provider

    first = run_tool_suite(
        scenarios=(scenario,), registry=registry, provider_factory=factory
    )
    second = run_tool_suite(
        scenarios=(scenario,), registry=registry, provider_factory=factory
    )
    assert first == second
    assert first is not second
    assert (
        providers[0].requests[0].conversation
        is not providers[1].requests[0].conversation
    )
    assert registry.list_definitions() == definitions
    assert scenario == create_example_tool_scenarios()[1]


def test_late_registry_additions_do_not_expand_selected_tools() -> None:
    from llm_benchmark.tool_runtime import ToolDefinition, ToolRegistration

    scenario = create_example_tool_scenarios()[1]
    scenarios = (
        scenario,
        replace(scenario, evaluation=replace(scenario.evaluation, case_id="second")),
    )
    registry = create_example_tool_registry()
    registrations = []

    def factory():
        if not registry.get("late_tool"):
            calculator = registry.get("calculator")
            registry.register(
                ToolRegistration(
                    ToolDefinition("late_tool", "Added after preparation."),
                    calculator.argument_model,
                    calculator.handler,
                )
            )
        provider = ScriptedProvider((successful_turn(content="READY"),))
        registrations.append(provider)
        return provider

    run_tool_suite(scenarios=scenarios, registry=registry, provider_factory=factory)
    assert all(
        tuple(r.definition.name for r in p.requests[0].registrations) == ("calculator",)
        for p in registrations
    )

from dataclasses import FrozenInstanceError, replace

import pytest

from llm_benchmark.tool_calling import NormalizationToolTurn
from llm_benchmark.tool_descriptors import descriptor_from_registration
from llm_benchmark.tool_provider_models import (
    ToolProviderErrorCode,
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_reporting import MetricCount
from llm_benchmark.tool_runtime import ToolCall, ToolRegistry, ToolRuntime
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
    values[field] = object() if field == "registry" else None
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


def example_descriptors():
    registry = create_example_tool_registry()
    return tuple(
        descriptor_from_registration(registry.get(definition.name))
        for definition in registry.list_definitions()
    )


def local_executor(request):
    available = create_example_tool_registry()
    selected = ToolRegistry()
    for descriptor in request.descriptors:
        selected.register(available.get(descriptor.definition.name))
    return ToolRuntime(selected)


def calculator_script():
    return (
        successful_turn(
            content=None,
            finish_reason="tool_calls",
            calls=(
                ToolCall(
                    call_id="calculator_1",
                    tool_name="calculator",
                    arguments={"operation": "multiply", "left": 17, "right": 23},
                ),
            ),
        ),
        successful_turn(content="391"),
    )


def test_descriptor_suite_matches_local_and_isolates_case_requests():
    scenarios = create_example_tool_scenarios()

    def providers():
        scripts = iter((calculator_script(), (successful_turn(content="READY"),)))
        return lambda: ScriptedProvider(next(scripts))

    local = run_tool_suite(
        scenarios=scenarios,
        registry=create_example_tool_registry(),
        provider_factory=providers(),
    )
    requests, executors = [], []

    def executor_factory(request):
        requests.append(request)
        executor = local_executor(request)
        executors.append(executor)
        return executor

    described = run_tool_suite(
        scenarios=scenarios,
        descriptors=example_descriptors(),
        provider_factory=providers(),
        executor_factory=executor_factory,
    )
    assert described == local
    assert described.summary == local.summary
    assert len(executors) == 2
    assert executors[0] is not executors[1]
    assert requests[0].conversation is not requests[1].conversation
    assert all(r.registrations == () for r in requests)
    assert all(
        tuple(d.definition.name for d in r.descriptors) == ("calculator",)
        for r in requests
    )


def test_later_missing_descriptor_prevents_both_factories():
    first, second = create_example_tool_scenarios()
    second = replace(second, available_tools=("private_missing_name",))

    def forbidden(*args):
        pytest.fail("No factory may run before preparation completes")

    with pytest.raises(ValueError) as caught:
        run_tool_suite(
            scenarios=(first, second),
            descriptors=example_descriptors(),
            provider_factory=forbidden,
            executor_factory=forbidden,
        )
    assert str(caught.value) == "A selected scenario tool is not available."


def test_repeated_descriptor_suites_preserve_inputs_and_results():
    scenario = create_example_tool_scenarios()[1]
    available = example_descriptors()
    requests = []

    def executor_factory(request):
        requests.append(request)
        return local_executor(request)

    def run():
        return run_tool_suite(
            scenarios=(scenario,),
            descriptors=available,
            executor_factory=executor_factory,
            provider_factory=lambda: ScriptedProvider((successful_turn(content="READY"),)),
        )

    first, second = run(), run()
    assert first == second
    assert first is not second
    assert requests[0].conversation is not requests[1].conversation
    assert available == example_descriptors()
    assert scenario == create_example_tool_scenarios()[1]
    with pytest.raises(FrozenInstanceError):
        first.evaluations = ()


@pytest.mark.parametrize(
    "mode",
    [
        "no_source",
        "both",
        "local_executor",
        "invalid_descriptors",
        "invalid_element",
        "duplicate",
        "no_executor",
        "invalid_executor",
        "duplicate_case",
    ],
)
def test_descriptor_setup_fails_before_factories(mode):
    def forbidden(*args):
        pytest.fail("Unexpected factory call")

    values = dict(
        scenarios=create_example_tool_scenarios(),
        descriptors=example_descriptors(),
        executor_factory=forbidden,
        provider_factory=forbidden,
    )
    if mode == "no_source":
        values["descriptors"] = ()
    elif mode == "both":
        values["registry"] = create_example_tool_registry()
    elif mode == "local_executor":
        values.update(registry=create_example_tool_registry(), descriptors=())
    elif mode == "invalid_descriptors":
        values["descriptors"] = []
    elif mode == "invalid_element":
        values["descriptors"] = (object(),)
    elif mode == "duplicate":
        values["descriptors"] *= 2
    elif mode == "duplicate_case":
        values["scenarios"] = (values["scenarios"][0],) * 2
    else:
        values["executor_factory"] = None if mode == "no_executor" else object()
    with pytest.raises((TypeError, ValueError)):
        run_tool_suite(**values)


@pytest.mark.parametrize(
    "value", [None, object(), type("InvalidExecutor", (), {"execute": 1})()]
)
def test_invalid_executor_precedes_provider_factory(value):
    attempts = []

    def executor_factory(request):
        attempts.append(request)
        return value

    def provider_factory():
        pytest.fail("Provider must not be created")

    with pytest.raises(TypeError, match="tool execution"):
        run_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=example_descriptors(),
            provider_factory=provider_factory,
            executor_factory=executor_factory,
        )
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "source", ["executor_factory", "execute", "provider_factory", "provider"]
)
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_descriptor_unexpected_failures_stop_suite(source, error_type):
    error = error_type("private detail")
    factories = []

    class Executor:
        def execute(self, call):
            raise error

    class Provider:
        def generate_tool_conversation(self, request):
            raise error

    def executor_factory(request):
        factories.append("executor")
        if source == "executor_factory":
            raise error
        return Executor()

    def provider_factory():
        factories.append("provider")
        if source == "provider_factory":
            raise error
        return (
            Provider()
            if source == "provider"
            else ScriptedProvider(calculator_script())
        )

    with pytest.raises(error_type) as caught:
        run_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=example_descriptors(),
            executor_factory=executor_factory,
            provider_factory=provider_factory,
        )
    assert caught.value is error
    assert factories == (
        ["executor"] if source == "executor_factory" else ["executor", "provider"]
    )


@pytest.mark.parametrize("failure_source", ["provider", "tool"])
def test_descriptor_normalized_failures_continue(failure_source):
    from llm_benchmark.tool_runtime import (
        ToolErrorCode,
        ToolExecutionStatus,
        ToolResult,
    )

    first = (
        (
            ToolProviderResult(
                status=ToolProviderStatus.REQUEST_FAILED,
                error_code=ToolProviderErrorCode.TIMEOUT,
            ),
        )
        if failure_source == "provider"
        else calculator_script()
    )
    scripts = iter((first, (successful_turn(content="READY"),)))
    attempts = []

    class Executor:
        def execute(self, call):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.EXECUTION_FAILED,
                error_code=ToolErrorCode.TOOL_EXECUTION_FAILED,
            )

    def factory(request):
        attempts.append(request)
        return Executor()

    result = run_tool_suite(
        scenarios=create_example_tool_scenarios(),
        descriptors=example_descriptors(),
        executor_factory=factory,
        provider_factory=lambda: ScriptedProvider(next(scripts)),
    )
    assert len(attempts) == 2
    assert result.evaluations[1].final_answer_match is True
    assert result.summary.successful_call_count == 0

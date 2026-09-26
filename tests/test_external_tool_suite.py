import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llm_benchmark.external_tool_policy import (
    ExternalToolAdmissionCode,
    ExternalToolAdmissionError,
    ExternalToolPolicy,
)
from llm_benchmark.external_tool_schema import (
    ExternalToolSchemaCode,
    ExternalToolSchemaError,
)
from llm_benchmark.external_tool_suite import run_external_tool_suite
from llm_benchmark.tool_calling import NormalizationToolTurn
from llm_benchmark.tool_descriptors import ToolDescriptor, descriptor_from_registration
from llm_benchmark.tool_provider_models import (
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolRegistry,
    ToolRuntime,
)
from llm_benchmark.tool_scenarios import create_example_tool_scenarios
from llm_benchmark.tool_suite import run_tool_suite
from llm_benchmark.tools import create_example_tool_registry


def calculator_descriptor():
    registry = create_example_tool_registry()
    registration = registry.get("calculator")
    assert registration is not None
    return descriptor_from_registration(registration)


def admission_policy(**updates):
    values = {
        "allowed_tool_names": ("calculator",),
        "max_tools": 1,
        "max_total_schema_bytes": 65_536,
    }
    values.update(updates)
    return ExternalToolPolicy(**values)


def forbidden_factory(*args, **kwargs):
    pytest.fail("Factories must not run when admission fails.")


def test_disallowed_catalog_does_not_create_resources():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=(calculator_descriptor(),),
            policy=admission_policy(allowed_tool_names=()),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )

    assert caught.value.code is ExternalToolAdmissionCode.TOOL_NOT_ALLOWED


def test_schema_budget_failure_does_not_create_resources():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=(calculator_descriptor(),),
            policy=admission_policy(max_total_schema_bytes=1),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )

    assert caught.value.code is ExternalToolAdmissionCode.SCHEMA_BUDGET_EXCEEDED


def test_admitted_catalog_executes_and_evaluates():
    scenario = create_example_tool_scenarios()[0]
    requests = []
    executed_calls = []
    selected_tools = []

    class ScriptedProvider:
        def generate_tool_conversation(self, request):
            requests.append(request)

            if len(requests) == 1:
                turn = NormalizationToolTurn(
                    content=None,
                    tool_calls=(
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
                )
            else:
                turn = NormalizationToolTurn(
                    content="391",
                    tool_calls=(),
                    finish_reason="stop",
                )

            return ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=turn,
            )

    def executor_factory(request):
        names = tuple(descriptor.definition.name for descriptor in request.descriptors)
        selected_tools.append(names)

        source = create_example_tool_registry()
        selected_registry = ToolRegistry()

        for name in names:
            registration = source.get(name)
            assert registration is not None
            selected_registry.register(registration)

        runtime = ToolRuntime(selected_registry)

        class RecordingExecutor:
            def execute(self, call):
                executed_calls.append(call)
                return runtime.execute(call)

        return RecordingExecutor()

    result = run_external_tool_suite(
        scenarios=(scenario,),
        descriptors=(calculator_descriptor(),),
        policy=admission_policy(),
        provider_factory=ScriptedProvider,
        executor_factory=executor_factory,
    )

    assert selected_tools == [("calculator",)]
    assert len(requests) == 2
    assert len(executed_calls) == 1
    assert executed_calls[0].tool_name == "calculator"

    assert len(result.evaluations) == 1
    assert result.summary.case_count == 1
    assert result.summary.executed_call_count == 1
    assert result.summary.successful_call_count == 1
    assert result.summary.final_answer_match.rate == 1.0


def city_descriptor():
    registration = create_example_tool_registry().get("lookup_city_code")
    assert registration is not None
    return descriptor_from_registration(registration)


@pytest.mark.parametrize("valid", [True, False])
def test_suite_validates_arguments_before_delegate_and_continues_conversation(valid):
    requests = []
    calls = []
    runtime = ToolRuntime(create_example_tool_registry())

    class Delegate:
        def execute(self, call):
            calls.append(call)
            return runtime.execute(call)

    class Provider:
        def generate_tool_conversation(self, request):
            requests.append(request)
            if len(requests) == 1:
                turn = NormalizationToolTurn(
                    content=None,
                    tool_calls=(
                        ToolCall(
                            call_id="call_1",
                            tool_name="calculator",
                            arguments={
                                "operation": "multiply",
                                "left": 17 if valid else "private-sentinel",
                                "right": 23,
                            },
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            else:
                result = request.conversation.messages[-1].result
                assert result.call_id == "call_1"
                assert result.tool_name == "calculator"
                if valid:
                    assert result.status is ToolExecutionStatus.SUCCEEDED
                else:
                    assert result.status is ToolExecutionStatus.INVALID_ARGUMENTS
                    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS
                    assert result.output is None
                    assert "private-sentinel" not in repr(result)
                turn = NormalizationToolTurn(
                    content="391", tool_calls=(), finish_reason="stop"
                )
            return ToolProviderResult(status=ToolProviderStatus.SUCCEEDED, turn=turn)

    result = run_external_tool_suite(
        scenarios=(create_example_tool_scenarios()[0],),
        descriptors=(calculator_descriptor(),),
        policy=admission_policy(),
        provider_factory=Provider,
        executor_factory=lambda request: Delegate(),
    )
    assert len(requests) == 2
    assert len(calls) == (1 if valid else 0)
    assert result.summary.completion.rate == 1.0
    assert result.summary.successful_call_count == (1 if valid else 0)
    # A normalized rejection is still an attempted loop execution, not a
    # successful delegate invocation; preserve existing reporting semantics.
    assert result.summary.executed_call_count == 1


@pytest.mark.parametrize("value", [None, object(), 1])
def test_noncallable_executor_factory_fails_before_provider(value):
    with pytest.raises(TypeError, match="executor factory"):
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=(calculator_descriptor(),),
            policy=admission_policy(),
            provider_factory=forbidden_factory,
            executor_factory=value,
        )


def test_invalid_delegate_fails_before_provider():
    with pytest.raises(TypeError, match="Delegate"):
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=(calculator_descriptor(),),
            policy=admission_policy(),
            provider_factory=forbidden_factory,
            executor_factory=lambda request: object(),
        )


@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize(
    "fragment,code",
    [
        ({"required": 42}, ExternalToolSchemaCode.INVALID_SCHEMA),
        (
            {"$ref": "https://example.invalid/schema"},
            ExternalToolSchemaCode.UNSUPPORTED_SCHEMA,
        ),
    ],
)
def test_all_catalog_schemas_validated_before_factories(selected, fragment, code):
    original = calculator_descriptor() if selected else city_descriptor()
    invalid = ToolDescriptor(
        definition=original.definition,
        parameters={"type": "object", **fragment},
    )
    catalog = (invalid,) if selected else (calculator_descriptor(), invalid)
    with pytest.raises(ExternalToolSchemaError) as caught:
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=catalog,
            policy=admission_policy(
                allowed_tool_names=("calculator", "lookup_city_code"), max_tools=2
            ),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )
    assert caught.value.code is code


@pytest.mark.parametrize(
    "catalog,code",
    [
        ((), ExternalToolAdmissionCode.EMPTY_COLLECTION),
        (("calculator", "city"), ExternalToolAdmissionCode.TOO_MANY_TOOLS),
        (("calculator", "calculator"), ExternalToolAdmissionCode.DUPLICATE_TOOL_NAME),
    ],
)
def test_catalog_rejections_prevent_both_factories(catalog, code):
    descriptors = tuple(
        calculator_descriptor() if name == "calculator" else city_descriptor()
        for name in catalog
    )
    with pytest.raises(ExternalToolAdmissionError) as caught:
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=descriptors,
            policy=admission_policy(
                max_tools=2
                if code is ExternalToolAdmissionCode.DUPLICATE_TOOL_NAME
                else 1
            ),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )
    assert caught.value.code is code


def test_unselected_disallowed_tool_rejects_whole_catalog():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=(calculator_descriptor(), city_descriptor()),
            policy=admission_policy(max_tools=2),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )
    assert caught.value.code is ExternalToolAdmissionCode.TOOL_NOT_ALLOWED
    assert "lookup_city_code" not in str(caught.value)


def test_missing_tool_in_later_scenario_prevents_all_factories():
    first, second = create_example_tool_scenarios()
    second = replace(second, available_tools=("missing_tool",))
    with pytest.raises(ValueError, match="not available"):
        run_external_tool_suite(
            scenarios=(first, second),
            descriptors=(calculator_descriptor(),),
            policy=admission_policy(),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )


def test_local_city_pattern_is_not_silently_accepted_as_external_schema():
    with pytest.raises(ExternalToolSchemaError) as caught:
        run_external_tool_suite(
            scenarios=create_example_tool_scenarios(),
            descriptors=(calculator_descriptor(), city_descriptor()),
            policy=admission_policy(
                allowed_tool_names=("calculator", "lookup_city_code"), max_tools=2
            ),
            provider_factory=forbidden_factory,
            executor_factory=forbidden_factory,
        )
    assert caught.value.code is ExternalToolSchemaCode.UNSUPPORTED_SCHEMA


def test_wrapper_matches_suite_and_preserves_inputs_across_runs():
    scenarios = (create_example_tool_scenarios()[1],)
    calculator = calculator_descriptor()
    unused = ToolDescriptor(
        definition=replace(calculator.definition, name="unused_calculator"),
        parameters=calculator.parameters,
    )
    descriptors = (unused, calculator)
    snapshots = tuple(item.parameters for item in descriptors)
    policy = admission_policy(
        allowed_tool_names=("calculator", "unused_calculator"), max_tools=2
    )
    events = []
    providers = []
    executors = []

    class ReadyProvider:
        def generate_tool_conversation(self, request):
            assert request.registrations == ()
            assert tuple(d.definition.name for d in request.descriptors) == (
                "calculator",
            )
            return ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=NormalizationToolTurn(
                    content="READY", tool_calls=(), finish_reason="stop"
                ),
            )

    def provider_factory():
        events.append("provider")
        provider = ReadyProvider()
        providers.append(provider)
        return provider

    def executor_factory(request):
        events.append("executor")
        assert tuple(d.definition.name for d in request.descriptors) == ("calculator",)
        executor = ToolRuntime(create_example_tool_registry())
        executors.append(executor)
        return executor

    arguments = dict(
        scenarios=scenarios,
        descriptors=descriptors,
        provider_factory=provider_factory,
        executor_factory=executor_factory,
    )
    expected = run_tool_suite(**arguments)
    first = run_external_tool_suite(**arguments, policy=policy)
    second = run_external_tool_suite(**arguments, policy=policy)
    assert first == second == expected
    assert first.summary == expected.summary
    assert events == ["executor", "provider"] * 3
    assert len({id(item) for item in providers}) == 3
    assert len({id(item) for item in executors}) == 3
    assert tuple(item.parameters for item in descriptors) == snapshots
    assert tuple(item.definition.name for item in descriptors) == (
        "unused_calculator",
        "calculator",
    )
    assert policy.allowed_tool_names == ("calculator", "unused_calculator")


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "phase", ["executor_factory", "provider_factory", "provider", "executor"]
)
def test_unexpected_errors_propagate_unchanged(error_type, phase):
    error = error_type("Synthetic failure.")

    class Provider:
        def generate_tool_conversation(self, request):
            if phase == "provider":
                raise error
            return ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=NormalizationToolTurn(
                    content=None,
                    tool_calls=(
                        ToolCall(
                            call_id="call_1",
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
            )

    class Executor:
        def execute(self, call):
            raise error

    def executor_factory(request):
        if phase == "executor_factory":
            raise error
        return Executor()

    def provider_factory():
        if phase == "provider_factory":
            raise error
        return Provider()

    with pytest.raises(error_type) as caught:
        run_external_tool_suite(
            scenarios=(create_example_tool_scenarios()[0],),
            descriptors=(calculator_descriptor(),),
            policy=admission_policy(),
            provider_factory=provider_factory,
            executor_factory=executor_factory,
        )
    assert caught.value is error


def test_import_does_not_initialize_execution_or_runtime_io(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import os
import sys
sys.path.insert(0, sys.argv[1])
def guard(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.mkdir')):
        raise AssertionError('Unexpected I/O')
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            raise AssertionError('Unexpected write')
sys.addaudithook(guard)
from llm_benchmark.tool_runtime import ToolRegistry, ToolRuntime
def reject(*args, **kwargs):
    raise AssertionError('Unexpected runtime initialization')
ToolRegistry.__init__ = reject
ToolRuntime.__init__ = reject
from llm_benchmark.external_tool_suite import run_external_tool_suite
assert callable(run_external_tool_suite)
from llm_benchmark.external_tool_schema import (
    validate_external_tool_schema, validate_external_tool_arguments,
    ExternalToolSchemaError, ExternalToolSchemaCode,
)
from llm_benchmark.tool_descriptors import ToolDescriptor
from llm_benchmark.tool_runtime import ToolDefinition, ToolCall
definition = ToolDefinition('example', 'Synthetic tool.')
descriptor = ToolDescriptor(definition=definition, parameters={'type': 'object'})
validate_external_tool_schema(descriptor)
validate_external_tool_arguments(
    descriptor=descriptor,
    call=ToolCall(call_id='call_1', tool_name='example', arguments={}),
)
from llm_benchmark.external_tool_executor import ValidatingExternalToolExecutor
from llm_benchmark.tool_runtime import ToolResult, ToolExecutionStatus
class Delegate:
    def execute(self, call):
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name,
                          status=ToolExecutionStatus.SUCCEEDED)
wrapper = ValidatingExternalToolExecutor(descriptors=(descriptor,), delegate=Delegate())
assert wrapper.execute(ToolCall(call_id='call_1', tool_name='example', arguments={})).status is ToolExecutionStatus.SUCCEEDED
for reference in ('https://example.invalid/schema', 'file:///unavailable-schema'):
    descriptor = ToolDescriptor(
        definition=definition, parameters={'type': 'object', '$ref': reference},
    )
    try:
        validate_external_tool_schema(descriptor)
    except ExternalToolSchemaError as error:
        assert error.code is ExternalToolSchemaCode.UNSUPPORTED_SCHEMA
    else:
        raise AssertionError('Reference must be rejected without retrieval')
for name in ('providers', 'runner', 'api', 'worker', 'db', 'tools'):
    assert 'llm_benchmark.' + name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, source_root],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []

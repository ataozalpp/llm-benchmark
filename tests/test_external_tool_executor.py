import pytest

import llm_benchmark.external_tool_executor as executor_module
from llm_benchmark.external_tool_executor import ValidatingExternalToolExecutor
from llm_benchmark.external_tool_schema import (
    ExternalToolSchemaCode,
    ExternalToolSchemaError,
)
from llm_benchmark.tool_descriptors import ToolDescriptor
from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolResult,
)


def descriptor():
    return ToolDescriptor(
        definition=ToolDefinition("calculator", "Synthetic tool."),
        parameters={
            "type": "object",
            "properties": {"left": {"type": "integer"}},
            "required": ["left"],
            "additionalProperties": False,
        },
    )


def call(arguments=None, name="calculator"):
    return ToolCall(
        call_id="call_1",
        tool_name=name,
        arguments={"left": 17} if arguments is None else arguments,
    )


class RecordingExecutor:
    def __init__(self):
        self.calls = []
        self.result = ToolResult(
            call_id="call_1",
            tool_name="calculator",
            status=ToolExecutionStatus.SUCCEEDED,
            output={"value": 34},
        )

    def execute(self, value):
        self.calls.append(value)
        return self.result


def test_valid_call_delegated_once_with_exact_call_and_result():
    delegate = RecordingExecutor()
    selected = descriptor()
    original_schema = selected.parameters
    invocation = call()
    wrapper = ValidatingExternalToolExecutor(descriptors=(selected,), delegate=delegate)
    assert delegate.calls == []
    assert wrapper.execute(invocation) is delegate.result
    assert delegate.calls == [invocation]
    assert delegate.calls[0] is invocation
    assert invocation.arguments == {"left": 17}
    assert selected.parameters == original_schema


@pytest.mark.parametrize(
    "arguments",
    [{}, {"left": "private-value"}, {"left": True}, {"left": 17, "extra": 1}],
)
def test_invalid_arguments_never_reach_delegate(arguments):
    delegate = RecordingExecutor()
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )
    result = wrapper.execute(call(arguments))
    assert result.status is ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert result.call_id == "call_1"
    assert result.tool_name == "calculator"
    assert result.output is None
    assert delegate.calls == []
    assert "private-value" not in repr(result)


def test_unknown_tool_never_reaches_delegate():
    delegate = RecordingExecutor()
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )
    result = wrapper.execute(call(name="unknown"))
    assert result.status is ToolExecutionStatus.TOOL_NOT_FOUND
    assert result.error_code is ToolErrorCode.TOOL_NOT_FOUND
    assert result.tool_name == "unknown"
    assert result.call_id == "call_1"
    assert result.output is None
    assert delegate.calls == []


@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_invalid_collection_type(value):
    with pytest.raises(TypeError):
        ValidatingExternalToolExecutor(descriptors=value, delegate=RecordingExecutor())


def test_empty_and_duplicate_collections():
    for descriptors in ((), (descriptor(), descriptor())):
        with pytest.raises(ValueError):
            ValidatingExternalToolExecutor(
                descriptors=descriptors, delegate=RecordingExecutor()
            )


@pytest.mark.parametrize("value", [None, {}, object()])
def test_invalid_descriptor_and_call_types(value):
    with pytest.raises(TypeError):
        ValidatingExternalToolExecutor(
            descriptors=(value,), delegate=RecordingExecutor()
        )
    delegate = RecordingExecutor()
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )
    with pytest.raises(TypeError):
        wrapper.execute(value)
    assert delegate.calls == []


@pytest.mark.parametrize(
    "delegate", [None, object(), type("Invalid", (), {"execute": 1})()]
)
def test_delegate_requires_callable_execute(delegate):
    with pytest.raises(TypeError):
        ValidatingExternalToolExecutor(descriptors=(descriptor(),), delegate=delegate)


@pytest.mark.parametrize(
    "schema,code",
    [
        ({"type": "object", "required": 42}, ExternalToolSchemaCode.INVALID_SCHEMA),
        (
            {"type": "object", "$ref": "https://example.invalid/schema"},
            ExternalToolSchemaCode.UNSUPPORTED_SCHEMA,
        ),
    ],
)
def test_bad_schema_rejected_at_construction(schema, code):
    selected = ToolDescriptor(definition=descriptor().definition, parameters=schema)
    delegate = RecordingExecutor()
    with pytest.raises(ExternalToolSchemaError) as caught:
        ValidatingExternalToolExecutor(descriptors=(selected,), delegate=delegate)
    assert caught.value.code is code
    assert delegate.calls == []


@pytest.mark.parametrize("value", [None, {}, object()])
def test_invalid_result_type(value):
    delegate = RecordingExecutor()
    delegate.result = value
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )
    with pytest.raises(TypeError, match="invalid result"):
        wrapper.execute(call())
    assert len(delegate.calls) == 1


@pytest.mark.parametrize(
    "call_id,name", [("other_id", "calculator"), ("call_1", "other_tool")]
)
def test_result_identity_must_match(call_id, name):
    delegate = RecordingExecutor()
    delegate.result = ToolResult(
        call_id=call_id, tool_name=name, status=ToolExecutionStatus.SUCCEEDED
    )
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )
    with pytest.raises(ValueError, match="does not match"):
        wrapper.execute(call())


@pytest.mark.parametrize(
    "status,code",
    [
        (ToolExecutionStatus.TOOL_NOT_FOUND, ToolErrorCode.TOOL_NOT_FOUND),
        (ToolExecutionStatus.INVALID_ARGUMENTS, ToolErrorCode.INVALID_ARGUMENTS),
        (ToolExecutionStatus.EXECUTION_FAILED, ToolErrorCode.TOOL_EXECUTION_FAILED),
        (ToolExecutionStatus.INVALID_OUTPUT, ToolErrorCode.INVALID_TOOL_OUTPUT),
        (ToolExecutionStatus.OUTPUT_TOO_LARGE, ToolErrorCode.TOOL_OUTPUT_TOO_LARGE),
    ],
)
def test_normalized_delegate_failures_preserved(status, code):
    delegate = RecordingExecutor()
    delegate.result = ToolResult(
        call_id="call_1", tool_name="calculator", status=status, error_code=code
    )
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )
    assert wrapper.execute(call()) is delegate.result


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["validation", "delegate"])
def test_unexpected_exceptions_propagate(monkeypatch, error_type, phase):
    error = error_type("Synthetic failure.")
    delegate = RecordingExecutor()
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )

    def reject(*args, **kwargs):
        raise error

    if phase == "validation":
        monkeypatch.setattr(executor_module, "validate_external_tool_arguments", reject)
    else:
        monkeypatch.setattr(delegate, "execute", reject)
    with pytest.raises(error_type) as caught:
        wrapper.execute(call())
    assert caught.value is error


@pytest.mark.parametrize(
    "code",
    [ExternalToolSchemaCode.INVALID_SCHEMA, ExternalToolSchemaCode.UNSUPPORTED_SCHEMA],
)
def test_setup_errors_not_normalized_as_argument_failures(monkeypatch, code):
    error = ExternalToolSchemaError(code)
    delegate = RecordingExecutor()
    wrapper = ValidatingExternalToolExecutor(
        descriptors=(descriptor(),), delegate=delegate
    )

    def reject(**kwargs):
        raise error

    monkeypatch.setattr(executor_module, "validate_external_tool_arguments", reject)
    with pytest.raises(ExternalToolSchemaError) as caught:
        wrapper.execute(call())
    assert caught.value is error
    assert delegate.calls == []

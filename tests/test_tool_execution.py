import pytest

from llm_benchmark.tool_calling import NormalizationToolTurn
from llm_benchmark.tool_conversation import ToolConversation, UserMessage
from llm_benchmark.tool_descriptors import descriptor_from_registration
from llm_benchmark.tool_execution import ToolExecutor
from llm_benchmark.tool_loop import (
    ToolLoopPolicy,
    ToolLoopStopReason,
    run_tool_loop,
)
from llm_benchmark.tool_provider_models import (
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_requests import ToolConversationRequest
from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolResult,
    ToolRuntime,
)
from llm_benchmark.tools import create_example_tool_registry


class ScriptedProvider:
    def __init__(self, results):
        self.results = tuple(results)
        self.requests = []

    def generate_tool_conversation(self, request):
        index = len(self.requests)
        self.requests.append(request)
        return self.results[index]


class RecordingExecutor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        return self.result


def make_request(*, descriptors=False):
    registry = create_example_tool_registry()
    registration = registry.get("calculator")
    assert registration is not None

    return ToolConversationRequest(
        conversation=ToolConversation(messages=(UserMessage("Synthetic task."),)),
        registrations=() if descriptors else (registration,),
        descriptors=(descriptor_from_registration(registration),)
        if descriptors
        else (),
    )


def make_call(
    call_id="call_1",
    tool_name="calculator",
):
    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments={
            "operation": "multiply",
            "left": 17,
            "right": 23,
        },
    )


def calls_response(*calls):
    return ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=NormalizationToolTurn(
            content=None,
            tool_calls=tuple(calls),
            finish_reason="tool_calls",
        ),
    )


def final_response():
    return ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=NormalizationToolTurn(
            content="391",
            tool_calls=(),
            finish_reason="stop",
        ),
    )


def successful_result(
    call_id="call_1",
    tool_name="calculator",
):
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        status=ToolExecutionStatus.SUCCEEDED,
        output={"result": 391},
    )


def test_existing_runtime_supports_executor_contract():
    executor: ToolExecutor = ToolRuntime(create_example_tool_registry())

    result = executor.execute(make_call())

    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.output == {"result": 391}


def test_default_and_injected_runtime_have_equal_results():
    request = make_request()
    policy = ToolLoopPolicy(
        max_provider_turns=2,
        max_tool_calls=1,
    )

    default_result = run_tool_loop(
        provider=ScriptedProvider((calls_response(make_call()), final_response())),
        request=request,
        policy=policy,
    )

    injected_result = run_tool_loop(
        provider=ScriptedProvider((calls_response(make_call()), final_response())),
        request=request,
        policy=policy,
        executor=ToolRuntime(create_example_tool_registry()),
    )

    assert injected_result == default_result


def test_injected_executor_avoids_local_runtime_initialization(
    monkeypatch,
):
    import llm_benchmark.tool_loop as loop_module

    request = make_request()
    call = make_call()
    executor = RecordingExecutor(successful_result())

    def reject_runtime(*args, **kwargs):
        raise AssertionError("Local runtime must not be initialized.")

    monkeypatch.setattr(loop_module, "ToolRuntime", reject_runtime)

    result = run_tool_loop(
        provider=ScriptedProvider((calls_response(call), final_response())),
        request=request,
        policy=ToolLoopPolicy(2, 1),
        executor=executor,
    )

    assert executor.calls == [call]
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert result.final_text == "391"


@pytest.mark.parametrize("executor", [object(), 123, "invalid"])
def test_invalid_executor_is_rejected_before_provider(executor):
    provider = ScriptedProvider(())

    with pytest.raises(
        TypeError,
        match="Executor must support tool execution.",
    ):
        run_tool_loop(
            provider=provider,
            request=make_request(),
            policy=ToolLoopPolicy(2, 1),
            executor=executor,
        )

    assert provider.requests == []


@pytest.mark.parametrize(
    ("returned", "error", "message"),
    [
        (
            None,
            TypeError,
            "Executor returned an invalid result",
        ),
        (
            successful_result(call_id="other_call"),
            ValueError,
            "Executor result does not match",
        ),
        (
            successful_result(tool_name="lookup_city_code"),
            ValueError,
            "Executor result does not match",
        ),
    ],
)
@pytest.mark.parametrize("descriptors", [False, True])
def test_invalid_executor_results_abort_before_next_provider_turn(
    returned,
    error,
    message,
    descriptors,
):
    provider = ScriptedProvider((calls_response(make_call()), final_response()))
    executor = RecordingExecutor(returned)
    request = make_request(descriptors=descriptors)

    with pytest.raises(error, match=message):
        run_tool_loop(
            provider=provider,
            request=request,
            policy=ToolLoopPolicy(2, 1),
            executor=executor,
        )

    assert len(provider.requests) == 1
    assert len(executor.calls) == 1
    assert len(request.conversation.messages) == 1


@pytest.mark.parametrize(
    ("calls", "policy", "expected_reason"),
    [
        (
            (
                make_call(),
                make_call("call_2", "lookup_city_code"),
            ),
            ToolLoopPolicy(2, 2),
            ToolLoopStopReason.TOOL_NOT_ALLOWED,
        ),
        (
            (make_call(), make_call("call_2")),
            ToolLoopPolicy(2, 1),
            ToolLoopStopReason.TOOL_CALL_LIMIT,
        ),
        (
            (make_call(),),
            ToolLoopPolicy(1, 1),
            ToolLoopStopReason.PROVIDER_TURN_LIMIT,
        ),
    ],
)
@pytest.mark.parametrize("descriptors", [False, True])
def test_guards_prevent_executor_calls(
    calls,
    policy,
    expected_reason,
    descriptors,
):
    executor = RecordingExecutor(successful_result())

    result = run_tool_loop(
        provider=ScriptedProvider((calls_response(*calls),)),
        request=make_request(descriptors=descriptors),
        policy=policy,
        executor=executor,
    )

    assert result.stop_reason is expected_reason
    assert executor.calls == []
    assert result.tool_results == ()


@pytest.mark.parametrize("descriptors", [False, True])
def test_normalized_executor_failure_is_added_to_conversation(descriptors):
    failure = ToolResult(
        call_id="call_1",
        tool_name="calculator",
        status=ToolExecutionStatus.EXECUTION_FAILED,
        error_code=ToolErrorCode.TOOL_EXECUTION_FAILED,
    )
    executor = RecordingExecutor(failure)
    provider = ScriptedProvider((calls_response(make_call()), final_response()))

    result = run_tool_loop(
        provider=provider,
        request=make_request(descriptors=descriptors),
        policy=ToolLoopPolicy(2, 1),
        executor=executor,
    )

    assert result.tool_results == (failure,)
    assert len(provider.requests) == 2
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
)
@pytest.mark.parametrize("descriptors", [False, True])
def test_executor_exceptions_propagate(error_type, descriptors):
    expected_error = error_type("Syntetic executor failure.")

    class RaisingExecutor:
        def execute(self, call):
            raise expected_error

    provider = ScriptedProvider((calls_response(make_call()), final_response()))

    with pytest.raises(error_type) as captured:
        run_tool_loop(
            provider=provider,
            request=make_request(descriptors=descriptors),
            policy=ToolLoopPolicy(2, 1),
            executor=RaisingExecutor(),
        )

    assert captured.value is expected_error
    assert len(provider.requests) == 1


def test_descriptor_request_requires_executor_before_runtime_or_provider(monkeypatch):
    import llm_benchmark.tool_loop as loop_module

    request = make_request(descriptors=True)
    provider = ScriptedProvider(())

    def reject(*args, **kwargs):
        raise AssertionError("Runtime must not initialize.")

    monkeypatch.setattr(loop_module, "ToolRegistry", reject)
    monkeypatch.setattr(loop_module, "ToolRuntime", reject)
    with pytest.raises(ValueError, match="Descriptor-only execution requires"):
        run_tool_loop(provider=provider, request=request, policy=ToolLoopPolicy(2, 1))
    assert provider.requests == []


def test_descriptor_loop_preserves_selection_and_uses_only_executor(monkeypatch):
    import llm_benchmark.tool_loop as loop_module

    request = make_request(descriptors=True)
    provider = ScriptedProvider((calls_response(make_call()), final_response()))
    executor = RecordingExecutor(successful_result())

    def reject(*args, **kwargs):
        raise AssertionError("Local runtime must not initialize.")

    monkeypatch.setattr(loop_module, "ToolRegistry", reject)
    monkeypatch.setattr(loop_module, "ToolRuntime", reject)
    result = run_tool_loop(
        provider=provider,
        request=request,
        policy=ToolLoopPolicy(2, 1),
        executor=executor,
    )
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert result.final_text == "391"
    assert executor.calls == [make_call()]
    assert len(provider.requests) == 2
    for sent in provider.requests:
        assert sent.registrations == ()
        assert sent.descriptors == request.descriptors
    assert len(request.conversation.messages) == 1

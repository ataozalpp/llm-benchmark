import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from llm_benchmark.tool_calling import (
    NormalizationToolTurn,
    ToolCallNormalizationErrorCode,
)
from llm_benchmark.tool_conversation import (
    AssistantMessage,
    ToolConversation,
    ToolResultMessage,
    UserMessage,
)
from llm_benchmark.tool_loop import (
    ToolLoopPolicy,
    ToolLoopResult,
    ToolLoopStopReason,
    run_tool_loop,
)
from llm_benchmark.tool_provider_models import (
    ToolProviderErrorCode,
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_requests import ToolConversationRequest
from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolRegistration,
)


def initial_conversation() -> ToolConversation:
    return ToolConversation(messages=(UserMessage("Synthetic task."),))


def completed_turn() -> NormalizationToolTurn:
    return NormalizationToolTurn(
        content="Synthetic final text.",
        tool_calls=(),
        finish_reason="stop",
    )


class ScriptedProvider:
    def __init__(
        self,
        results: tuple[ToolProviderResult, ...],
    ) -> None:
        self.results = results
        self.requests: list[ToolConversationRequest] = []

    def generate_tool_conversation(
        self,
        request: ToolConversationRequest,
    ) -> ToolProviderResult:
        index = len(self.requests)
        self.requests.append(request)
        return self.results[index]


def test_valid_policy_is_frozen() -> None:
    policy = ToolLoopPolicy(
        max_provider_turns=3,
        max_tool_calls=4,
    )

    assert policy.max_provider_turns == 3
    assert policy.max_tool_calls == 4

    with pytest.raises(FrozenInstanceError):
        policy.max_tool_calls = 10


@pytest.mark.parametrize(
    "field",
    ["max_provider_turns", "max_tool_calls"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_invalid_policy_values(field: str, value: object) -> None:
    values = {
        "max_provider_turns": 3,
        "max_tool_calls": 4,
    }
    values[field] = value

    with pytest.raises(ValueError):
        ToolLoopPolicy(**values)


def test_final_response_result() -> None:
    turn = completed_turn()
    provider_result = ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=turn,
    )
    conversation = initial_conversation().append(AssistantMessage(turn))

    result = ToolLoopResult(
        stop_reason=ToolLoopStopReason.FINAL_RESPONSE,
        conversation=conversation,
        provider_results=(provider_result,),
        tool_results=(),
        final_text=turn.content,
    )

    assert result.provider_turn_count == 1
    assert result.tool_call_count == 0
    assert result.final_text == "Synthetic final text."
    assert "Synthetic final text." not in repr(result)

    with pytest.raises(FrozenInstanceError):
        result.final_text = "Changed."


def test_non_final_result_has_no_final_text() -> None:
    with pytest.raises(ValueError, match="Only a final-response"):
        ToolLoopResult(
            stop_reason=ToolLoopStopReason.PROVIDER_TURN_LIMIT,
            conversation=initial_conversation(),
            provider_results=(),
            tool_results=(),
            final_text="Not confirmed.",
        )


@pytest.mark.parametrize(
    ("content", "finish_reason", "final_text"),
    [
        ("Text", "stop", None),
        ("Text", "stop", "Different text"),
        ("Text", "length", "Text"),
        ("Text", None, "Text"),
    ],
)
def test_invalid_final_response_claims(
    content: str,
    finish_reason: str | None,
    final_text: str | None,
) -> None:
    turn = NormalizationToolTurn(
        content=content,
        tool_calls=(),
        finish_reason=finish_reason,
    )
    conversation = initial_conversation().append(AssistantMessage(turn))

    with pytest.raises(ValueError):
        ToolLoopResult(
            stop_reason=ToolLoopStopReason.FINAL_RESPONSE,
            conversation=conversation,
            provider_results=(),
            tool_results=(),
            final_text=final_text,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"stop_reason": "provider_turn_limit"},
        {"conversation": None},
        {"provider_results": []},
        {"provider_results": (object(),)},
        {"tool_results": []},
        {"tool_results": (object(),)},
    ],
)
def test_result_rejects_invalid_types(
    updates: dict[str, object],
) -> None:
    values = {
        "stop_reason": ToolLoopStopReason.PROVIDER_TURN_LIMIT,
        "conversation": initial_conversation(),
        "provider_results": (),
        "tool_results": (),
    }
    values.update(updates)

    with pytest.raises(TypeError):
        ToolLoopResult(**values)


def test_loop_executes_tool_then_returns_final_response() -> None:
    class AddArguments(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        left: int
        right: int

    executions: list[tuple[int, int]] = []

    def add(arguments: AddArguments) -> dict[str, int]:
        executions.append((arguments.left, arguments.right))
        return {"result": arguments.left + arguments.right}

    registration = ToolRegistration(
        definition=ToolDefinition(
            name="add",
            description="Add two synthetic integers.",
        ),
        argument_model=AddArguments,
        handler=add,
    )

    call_turn = NormalizationToolTurn(
        content=None,
        tool_calls=(
            ToolCall(
                call_id="call_1",
                tool_name="add",
                arguments={"left": 2, "right": 3},
            ),
        ),
        finish_reason="tool_calls",
    )

    final_turn = NormalizationToolTurn(
        content="5",
        tool_calls=(),
        finish_reason="stop",
    )

    provider = ScriptedProvider(
        results=(
            ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=call_turn,
            ),
            ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=final_turn,
            ),
        ),
    )

    request = ToolConversationRequest(
        conversation=initial_conversation(),
        registrations=(registration,),
    )

    result = run_tool_loop(
        provider=provider,
        request=request,
        policy=ToolLoopPolicy(
            max_provider_turns=2,
            max_tool_calls=1,
        ),
    )

    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert result.final_text == "5"
    assert result.provider_turn_count == 2
    assert result.tool_call_count == 1
    assert executions == [(2, 3)]
    assert result.tool_results[0].output == {"result": 5}

    assert len(provider.requests) == 2
    assert len(provider.requests[0].conversation.messages) == 1
    assert len(provider.requests[1].conversation.messages) == 3
    assert len(result.conversation.messages) == 4

    assert request.conversation == initial_conversation()


@pytest.fixture
def loop_setup():
    class Arguments(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        value: int

    executions: list[int] = []

    def handler(arguments: Arguments) -> dict[str, int]:
        executions.append(arguments.value)
        return {"value": arguments.value}

    registration = ToolRegistration(
        definition=ToolDefinition(name="echo", description="Synthetic echo."),
        argument_model=Arguments,
        handler=handler,
    )
    return ToolConversationRequest(
        conversation=initial_conversation(), registrations=(registration,)
    ), executions


def call(call_id: str = "call_1", *, name: str = "echo", value=1) -> ToolCall:
    return ToolCall(call_id=call_id, tool_name=name, arguments={"value": value})


def calls_result(*calls: ToolCall) -> ToolProviderResult:
    return ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=NormalizationToolTurn(
            content=None, tool_calls=calls, finish_reason="tool_calls"
        ),
    )


def final_result(finish_reason="stop") -> ToolProviderResult:
    return ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=NormalizationToolTurn(
            content="Synthetic final text.", tool_calls=(), finish_reason=finish_reason
        ),
    )


def test_direct_final_never_executes_tools(loop_setup) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider((final_result(),))
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(1, 1)
    )
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert result.provider_turn_count == len(provider.requests) == 1
    assert result.tool_call_count == 0
    assert executions == []


@pytest.mark.parametrize("finish_reason", [None, "length", "content_filter", "unknown"])
def test_unconfirmed_completion_is_not_final(loop_setup, finish_reason) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider((final_result(finish_reason),))
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(3, 3)
    )
    assert result.stop_reason is ToolLoopStopReason.COMPLETION_UNCONFIRMED
    assert result.final_text is None
    assert len(provider.requests) == 1
    assert executions == []


@pytest.mark.parametrize("after_tool", [False, True])
@pytest.mark.parametrize("invalid_response", [False, True])
def test_provider_failure_stops_without_retry(
    loop_setup, after_tool, invalid_response
) -> None:
    request, executions = loop_setup
    failure = (
        ToolProviderResult(
            status=ToolProviderStatus.RESPONSE_INVALID,
            normalization_error_code=ToolCallNormalizationErrorCode.INVALID_RESPONSE,
        )
        if invalid_response
        else ToolProviderResult(
            status=ToolProviderStatus.REQUEST_FAILED,
            error_code=ToolProviderErrorCode.TIMEOUT,
        )
    )
    script = (calls_result(call()), failure) if after_tool else (failure,)
    provider = ScriptedProvider(script)
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(4, 4)
    )
    expected = (
        ToolLoopStopReason.INVALID_RESPONSE
        if invalid_response
        else ToolLoopStopReason.PROVIDER_FAILED
    )
    assert result.stop_reason is expected
    assert result.provider_results == script
    assert len(provider.requests) == len(script)
    assert executions == ([1] if after_tool else [])
    assert result.tool_call_count == int(after_tool)
    assert result.final_text is None


@pytest.mark.parametrize("unknown_first", [False, True])
def test_unknown_tool_rejects_entire_batch(loop_setup, unknown_first) -> None:
    request, executions = loop_setup
    batch = (call(), call("call_2", name="not_selected"))
    if unknown_first:
        batch = tuple(reversed(batch))
    provider = ScriptedProvider((calls_result(*batch),))
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(3, 3)
    )
    assert result.stop_reason is ToolLoopStopReason.TOOL_NOT_ALLOWED
    assert result.tool_results == ()
    assert executions == []
    assert len(provider.requests) == 1


@pytest.mark.parametrize("prior_call", [False, True])
def test_tool_budget_rejects_whole_batch(loop_setup, prior_call) -> None:
    request, executions = loop_setup
    batch = calls_result(call("call_2"), call("call_3"))
    script = (calls_result(call()), batch) if prior_call else (batch,)
    provider = ScriptedProvider(script)
    result = run_tool_loop(
        provider=provider,
        request=request,
        policy=ToolLoopPolicy(4, 2 if prior_call else 1),
    )
    assert result.stop_reason is ToolLoopStopReason.TOOL_CALL_LIMIT
    assert executions == ([1] if prior_call else [])
    assert result.tool_call_count == int(prior_call)
    assert len(provider.requests) == len(script)


@pytest.mark.parametrize("prior_call", [False, True])
def test_last_provider_turn_does_not_execute_new_tools(loop_setup, prior_call) -> None:
    request, executions = loop_setup
    script = (calls_result(call()), calls_result(call("call_2")))
    if not prior_call:
        script = script[:1]
    provider = ScriptedProvider(script)
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(len(script), 5)
    )
    assert result.stop_reason is ToolLoopStopReason.PROVIDER_TURN_LIMIT
    assert executions == ([1] if prior_call else [])
    assert result.provider_turn_count == len(script)
    assert result.tool_call_count == int(prior_call)


def test_exact_budget_batch_preserves_order_and_snapshots(loop_setup) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider(
        (
            calls_result(call("call_b", value=2), call("call_a", value=1)),
            final_result(),
        )
    )
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(2, 2)
    )
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert executions == [2, 1]
    assert [item.call_id for item in result.tool_results] == ["call_b", "call_a"]
    history = provider.requests[1].conversation.messages
    assert history[2:] == tuple(ToolResultMessage(r) for r in result.tool_results)
    assert len(request.conversation.messages) == 1
    assert len(provider.requests[0].conversation.messages) == 1
    assert all(r.registrations == request.registrations for r in provider.requests)
    output = result.tool_results[0].output
    output["value"] = 999
    assert result.tool_results[0].output == {"value": 2}
    assert "Synthetic" not in repr(result)


def test_repeated_call_id_stops_before_second_execution(loop_setup) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider((calls_result(call()), calls_result(call())))
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(3, 3)
    )
    assert result.stop_reason is ToolLoopStopReason.INVALID_CONVERSATION
    assert executions == [1]
    assert result.tool_call_count == 1
    assert result.provider_turn_count == 2
    assert result.conversation == provider.requests[1].conversation


def test_invalid_tool_arguments_are_sent_back_as_outcome(loop_setup) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider((calls_result(call(value="1")), final_result()))
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(2, 1)
    )
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert executions == []
    assert result.tool_call_count == 1
    tool_result = result.tool_results[0]
    assert tool_result.status is ToolExecutionStatus.INVALID_ARGUMENTS
    assert tool_result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert provider.requests[1].conversation.messages[-1] == ToolResultMessage(
        tool_result
    )


@pytest.mark.parametrize(
    "error", [RuntimeError("private sentinel"), KeyboardInterrupt(), SystemExit()]
)
def test_handler_exception_policy_is_owned_by_runtime(loop_setup, error) -> None:
    request, _ = loop_setup
    original = request.registrations[0]

    def handler(arguments):
        raise error

    registration = ToolRegistration(
        original.definition, original.argument_model, handler
    )
    request = ToolConversationRequest(request.conversation, (registration,))
    provider = ScriptedProvider((calls_result(call()), final_result()))
    if not isinstance(error, Exception):
        with pytest.raises(type(error)) as caught:
            run_tool_loop(
                provider=provider, request=request, policy=ToolLoopPolicy(2, 1)
            )
        assert caught.value is error
        assert len(provider.requests) == 1
        return
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(2, 1)
    )
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    tool_result = result.tool_results[0]
    assert tool_result.error_code is ToolErrorCode.TOOL_EXECUTION_FAILED
    assert tool_result.output is None
    assert "private sentinel" not in repr(result)
    assert provider.requests[1].conversation.messages[-1] == ToolResultMessage(
        tool_result
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("sentinel"),
        TypeError("sentinel"),
        AssertionError(),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
def test_unexpected_provider_errors_propagate(loop_setup, error) -> None:
    request, executions = loop_setup
    attempts = []

    class BrokenProvider:
        def generate_tool_conversation(self, request):
            attempts.append(request)
            raise error

    with pytest.raises(type(error)) as caught:
        run_tool_loop(
            provider=BrokenProvider(), request=request, policy=ToolLoopPolicy(3, 3)
        )
    assert caught.value is error
    assert len(attempts) == 1
    assert executions == []


def test_invalid_provider_return_type_is_not_normalized(loop_setup) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider((None,))
    with pytest.raises(TypeError, match="Provider returned an invalid result"):
        run_tool_loop(provider=provider, request=request, policy=ToolLoopPolicy(2, 2))
    assert len(provider.requests) == 1
    assert executions == []


@pytest.mark.parametrize("field", ["request", "policy"])
def test_invalid_setup_has_no_execution(loop_setup, field) -> None:
    request, executions = loop_setup
    provider = ScriptedProvider(())
    kwargs = dict(provider=provider, request=request, policy=ToolLoopPolicy(2, 2))
    kwargs[field] = None
    with pytest.raises(TypeError):
        run_tool_loop(**kwargs)
    assert provider.requests == []
    assert executions == []


def test_repeated_executions_have_independent_state(loop_setup) -> None:
    request, executions = loop_setup
    results = []
    for _ in range(2):
        provider = ScriptedProvider((calls_result(call()), final_result()))
        results.append(
            run_tool_loop(
                provider=provider, request=request, policy=ToolLoopPolicy(2, 1)
            )
        )
        assert len(provider.requests) == 2
    assert results[0] == results[1]
    assert results[0].conversation is not results[1].conversation
    assert results[0].provider_turn_count == 2
    assert results[0].tool_call_count == 1
    assert executions == [1, 1]
    assert len(request.conversation.messages) == 1


def test_import_has_no_runtime_initialization_or_io(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import os
import sys
sys.path.insert(0, sys.argv[1])
def reject_io(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.mkdir')):
        raise AssertionError('Unexpected runtime I/O')
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            raise AssertionError('Unexpected file write')
sys.addaudithook(reject_io)
from llm_benchmark.tool_runtime import ToolRegistry, ToolRuntime
def fail(*args, **kwargs):
    raise AssertionError('Unexpected runtime initialization')
ToolRegistry.__init__ = fail
ToolRuntime.__init__ = fail
from llm_benchmark.tool_loop import ToolLoopPolicy
assert ToolLoopPolicy(2, 1).max_provider_turns == 2
for name in ('providers', 'runner', 'trace', 'api', 'worker', 'db', 'tools'):
    assert 'llm_benchmark.' + name not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, source_root],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
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
from llm_benchmark.tool_evaluation import (
    ExpectedToolCall,
    ToolEvaluationCase,
    ToolEvaluationResult,
    evaluate_tool_loop,
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
    ToolRegistration,
    ToolRuntime,
)


def stopped_calls(calls: tuple[ToolCall, ...]) -> ToolLoopResult:
    turn = NormalizationToolTurn(
        content=None, tool_calls=calls, finish_reason="tool_calls"
    )
    return ToolLoopResult(
        stop_reason=ToolLoopStopReason.TOOL_CALL_LIMIT,
        conversation=ToolConversation(
            (UserMessage("Synthetic task."), AssistantMessage(turn))
        ),
        provider_results=(
            ToolProviderResult(status=ToolProviderStatus.SUCCEEDED, turn=turn),
        ),
        tool_results=(),
    )


@pytest.mark.parametrize("names", [("other",), (), ("echo", "echo")])
def test_mismatched_tool_sequence_is_not_compared_to_itself(names) -> None:
    case = ToolEvaluationCase(
        case_id="sequence",
        expected_calls=(ExpectedToolCall(tool_name="echo", arguments={}),),
        expected_final_text="done",
    )
    calls = tuple(
        ToolCall(call_id=f"call_{i}", tool_name=name, arguments={})
        for i, name in enumerate(names)
    )
    result = stopped_calls(calls) if calls else direct_final_result("done")
    evaluation = evaluate_tool_loop(case=case, result=result)
    assert evaluation.tool_sequence_match is False
    assert evaluation.arguments_match is None


def valid_evaluation() -> ToolEvaluationResult:
    return evaluate_tool_loop(
        case=ToolEvaluationCase(
            case_id="case", expected_calls=(), expected_final_text="done"
        ),
        result=direct_final_result("done"),
    )


@pytest.mark.parametrize("field", ["completed", "tool_sequence_match"])
@pytest.mark.parametrize("value", [1, "yes"])
def test_result_flags_reject_non_booleans(field, value) -> None:
    with pytest.raises(TypeError):
        replace(valid_evaluation(), **{field: value})


def direct_final_result(text: str) -> ToolLoopResult:
    turn = NormalizationToolTurn(
        content=text,
        tool_calls=(),
        finish_reason="stop",
    )
    conversation = ToolConversation(
        messages=(
            UserMessage("Synthetic task."),
            AssistantMessage(turn),
        )
    )
    return ToolLoopResult(
        stop_reason=ToolLoopStopReason.FINAL_RESPONSE,
        conversation=conversation,
        provider_results=(
            ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=turn,
            ),
        ),
        tool_results=(),
        final_text=text,
    )


def test_direct_final_without_expected_tools() -> None:
    case = ToolEvaluationCase(
        case_id="direct-final",
        expected_calls=(),
        expected_final_text="5",
    )

    evaluation = evaluate_tool_loop(
        case=case,
        result=direct_final_result("5"),
    )

    assert evaluation.completed is True
    assert evaluation.tool_sequence_match is True
    assert evaluation.arguments_match is True
    assert evaluation.final_answer_match is True
    assert evaluation.requested_tool_call_count == 0
    assert evaluation.executed_tool_call_count == 0
    assert evaluation.successful_tool_call_count == 0

    with pytest.raises(FrozenInstanceError):
        evaluation.completed = False


def test_final_matching_does_not_trim_text() -> None:
    case = ToolEvaluationCase(
        case_id="strict-final",
        expected_calls=(),
        expected_final_text="5",
    )

    evaluation = evaluate_tool_loop(
        case=case,
        result=direct_final_result(" 5 "),
    )

    assert evaluation.completed is True
    assert evaluation.final_answer_match is False


def test_expected_arguments_are_independent_snapshots() -> None:
    source = {"values": [1, 2]}
    expected = ExpectedToolCall(
        tool_name="example",
        arguments=source,
    )

    source["values"].append(3)
    assert expected.arguments == {"values": [1, 2]}

    returned = expected.arguments
    returned["values"].append(4)
    assert expected.arguments == {"values": [1, 2]}


def test_expected_calls_must_be_a_tuple() -> None:
    with pytest.raises(TypeError, match="tuple"):
        ToolEvaluationCase(
            case_id="invalid-case",
            expected_calls=[],
            expected_final_text="5",
        )


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ({"v": True}, {"v": 1}, False),
        ({"v": 1}, {"v": 1.0}, False),
        ({"v": 1}, {"v": "1"}, False),
        ({"v": [1, 2]}, {"v": [2, 1]}, False),
        ({"v": [1]}, {"v": [1, 2]}, False),
        ({"v": None}, {}, False),
        ({"v": None}, {"v": None}, True),
        ({"v": [{"x": True}]}, {"v": [{"x": 1}]}, False),
        (
            {"v": [{"x": "İstanbul", "y": None}]},
            {"v": [{"y": None, "x": "İstanbul"}]},
            True,
        ),
    ],
)
def test_strict_json_argument_matching(expected, actual, matches) -> None:
    case = ToolEvaluationCase(
        "json", (ExpectedToolCall(tool_name="echo", arguments=expected),), "done"
    )
    result = stopped_calls(
        (ToolCall(call_id="different_id", tool_name="echo", arguments=actual),)
    )
    evaluated = evaluate_tool_loop(case=case, result=result)
    assert evaluated.tool_sequence_match is True
    assert evaluated.arguments_match is matches
    assert evaluated.final_answer_match is None
    assert evaluated.requested_tool_call_count == 1
    assert evaluated.executed_tool_call_count == 0


def test_call_order_matters_but_call_ids_do_not() -> None:
    expected = tuple(
        ExpectedToolCall(tool_name=name, arguments={}) for name in ("first", "second")
    )
    case = ToolEvaluationCase("order", expected, "done")
    calls = tuple(
        ToolCall(call_id=f"id_{i}", tool_name=e.tool_name, arguments={})
        for i, e in enumerate(expected)
    )
    result = stopped_calls(calls)
    first = evaluate_tool_loop(case=case, result=result)
    renamed = tuple(
        ToolCall(call_id=f"new_{i}", tool_name=c.tool_name, arguments=c.arguments)
        for i, c in enumerate(calls)
    )
    assert evaluate_tool_loop(case=case, result=stopped_calls(renamed)) == first
    reversed_result = evaluate_tool_loop(
        case=case, result=stopped_calls(tuple(reversed(calls)))
    )
    assert reversed_result.tool_sequence_match is False
    assert reversed_result.arguments_match is None


@pytest.mark.parametrize("field", ["case_id", "expected_final_text"])
@pytest.mark.parametrize("value", [None, 1, "", "  ", "\ud800"])
def test_case_text_validation(field, value) -> None:
    values = dict(case_id="case", expected_calls=(), expected_final_text="done")
    values[field] = value
    with pytest.raises(ValueError):
        ToolEvaluationCase(**values)


@pytest.mark.parametrize(
    "arguments",
    [[], None, {"v": float("nan")}, {"v": float("inf")}, {"v": (1, 2)}, {1: "value"}],
)
def test_expectations_reuse_strict_json_validation(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExpectedToolCall(tool_name="echo", arguments=arguments)


def test_expectation_types_immutability_and_repr() -> None:
    expected = ExpectedToolCall(tool_name="echo", arguments={"v": "private_marker"})
    case = ToolEvaluationCase("case", (expected,), "private_final")
    with pytest.raises(FrozenInstanceError):
        expected._call = None
    with pytest.raises(FrozenInstanceError):
        case.expected_calls = ()
    assert "private_marker" not in repr(expected)
    assert "private_final" not in repr(case)
    with pytest.raises(TypeError):
        ToolEvaluationCase("case", (object(),), "done")
    with pytest.raises(ValueError):
        ExpectedToolCall(tool_name="invalid name", arguments={})


@pytest.mark.parametrize(
    "updates",
    [
        {"stop_reason": "final_response"},
        {"arguments_match": 1},
        {"final_answer_match": "yes"},
        {"completed": False},
        {"arguments_match": None},
        {"tool_sequence_match": False, "arguments_match": True},
        {"final_answer_match": None},
        {"stop_reason": ToolLoopStopReason.PROVIDER_FAILED, "completed": False},
        {"executed_tool_call_count": 1},
        {"successful_tool_call_count": 1},
        {"case_id": ""},
    ],
)
def test_inconsistent_result_contracts(updates) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(valid_evaluation(), **updates)


@pytest.mark.parametrize(
    "field",
    [
        "requested_tool_call_count",
        "executed_tool_call_count",
        "successful_tool_call_count",
    ],
)
@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_result_count_types(field, value) -> None:
    with pytest.raises(ValueError):
        replace(valid_evaluation(), **{field: value})


@pytest.mark.parametrize("field", ["case", "result"])
def test_evaluator_rejects_invalid_input_types(field) -> None:
    kwargs = dict(
        case=ToolEvaluationCase("case", (), "done"), result=direct_final_result("done")
    )
    kwargs[field] = None
    with pytest.raises(TypeError):
        evaluate_tool_loop(**kwargs)


class ScriptedProvider:
    def __init__(self, results):
        self.results = tuple(results)
        self.requests = []

    def generate_tool_conversation(self, request):
        index = len(self.requests)
        self.requests.append(request)
        return self.results[index]


def provider_calls(*calls):
    return ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=NormalizationToolTurn(
            content=None,
            tool_calls=calls,
            finish_reason="tool_calls",
        ),
    )


@pytest.fixture
def execution_setup():
    class Arguments(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        value: int

    executions = []

    def handler(arguments):
        executions.append(arguments.value)
        if arguments.value < 0:
            raise RuntimeError("private handler detail")
        return {"value": arguments.value}

    registration = ToolRegistration(
        ToolDefinition("echo", "Synthetic echo."), Arguments, handler
    )
    return ToolConversationRequest(
        ToolConversation((UserMessage("Synthetic task."),)), (registration,)
    ), executions


@pytest.mark.parametrize("value", [1, "1", -1])
def test_real_loop_evaluation_and_no_reexecution(
    execution_setup, monkeypatch, value
) -> None:
    request, executions = execution_setup
    call = ToolCall(call_id="actual_id", tool_name="echo", arguments={"value": value})
    provider = ScriptedProvider(
        (provider_calls(call), direct_final_result("done").provider_results[0])
    )
    result = run_tool_loop(
        provider=provider, request=request, policy=ToolLoopPolicy(2, 1)
    )
    case = ToolEvaluationCase(
        "roundtrip",
        (ExpectedToolCall(tool_name="echo", arguments={"value": value}),),
        "done",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation must not execute tools or providers")

    monkeypatch.setattr(ToolRuntime, "execute", forbidden)
    monkeypatch.setattr(provider, "generate_tool_conversation", forbidden)
    evaluated = evaluate_tool_loop(case=case, result=result)
    assert evaluated == evaluate_tool_loop(case=case, result=result)
    assert evaluated.completed is True
    assert evaluated.tool_sequence_match is True
    assert evaluated.arguments_match is True
    assert evaluated.final_answer_match is True
    assert (
        evaluated.requested_tool_call_count == evaluated.executed_tool_call_count == 1
    )
    assert evaluated.successful_tool_call_count == int(value == 1)
    assert executions == ([] if type(value) is str else [value])
    assert len(provider.requests) == 2
    assert len(request.conversation.messages) == 1
    assert "private handler detail" not in repr(evaluated)
    assert "done" not in repr(evaluated)


@pytest.mark.parametrize(
    "reason",
    [
        reason
        for reason in ToolLoopStopReason
        if reason is not ToolLoopStopReason.FINAL_RESPONSE
    ],
)
def test_all_nonfinal_outcomes_from_real_loop(execution_setup, reason) -> None:
    request, executions = execution_setup
    call = ToolCall(call_id="call_1", tool_name="echo", arguments={"value": 1})
    policy = ToolLoopPolicy(3, 3)
    requested = 0
    executed = 0
    if reason is ToolLoopStopReason.PROVIDER_FAILED:
        script = [
            ToolProviderResult(
                status=ToolProviderStatus.REQUEST_FAILED,
                error_code=ToolProviderErrorCode.TIMEOUT,
            )
        ]
    elif reason is ToolLoopStopReason.INVALID_RESPONSE:
        script = [
            ToolProviderResult(
                status=ToolProviderStatus.RESPONSE_INVALID,
                normalization_error_code=ToolCallNormalizationErrorCode.INVALID_RESPONSE,
            )
        ]
    elif reason is ToolLoopStopReason.COMPLETION_UNCONFIRMED:
        script = [
            ToolProviderResult(
                status=ToolProviderStatus.SUCCEEDED,
                turn=NormalizationToolTurn(
                    content="done", tool_calls=(), finish_reason="length"
                ),
            )
        ]
    elif reason is ToolLoopStopReason.INVALID_CONVERSATION:
        script = [provider_calls(call), provider_calls(call)]
        requested, executed = 2, 1
    elif reason is ToolLoopStopReason.TOOL_NOT_ALLOWED:
        script = [
            provider_calls(ToolCall(call_id="call_1", tool_name="other", arguments={}))
        ]
        requested = 1
    elif reason is ToolLoopStopReason.TOOL_CALL_LIMIT:
        script = [
            provider_calls(
                call,
                ToolCall(call_id="call_2", tool_name="echo", arguments={"value": 2}),
            )
        ]
        policy = ToolLoopPolicy(3, 1)
        requested = 2
    else:
        script = [provider_calls(call)]
        policy = ToolLoopPolicy(1, 3)
        requested = 1
    result = run_tool_loop(
        provider=ScriptedProvider(script), request=request, policy=policy
    )
    evaluated = evaluate_tool_loop(
        case=ToolEvaluationCase("nonfinal", (), "done"), result=result
    )
    assert evaluated.stop_reason is reason
    assert evaluated.completed is False
    assert evaluated.final_answer_match is None
    assert evaluated.requested_tool_call_count == requested
    assert evaluated.executed_tool_call_count == executed
    assert evaluated.successful_tool_call_count == executed
    assert executions == ([1] if executed else [])


def test_prior_history_is_not_counted_as_current_execution(execution_setup) -> None:
    request, _ = execution_setup
    call = ToolCall(call_id="prior", tool_name="echo", arguments={"value": 1})
    # Build valid prior history through the real loop, stopping after a failed next turn.
    failed = ToolProviderResult(
        status=ToolProviderStatus.REQUEST_FAILED,
        error_code=ToolProviderErrorCode.TIMEOUT,
    )
    prior = run_tool_loop(
        provider=ScriptedProvider((provider_calls(call), failed)),
        request=request,
        policy=ToolLoopPolicy(2, 1),
    )
    assert type(prior.conversation.messages[-1]) is ToolResultMessage
    current = run_tool_loop(
        provider=ScriptedProvider(direct_final_result("done").provider_results),
        request=ToolConversationRequest(prior.conversation, request.registrations),
        policy=ToolLoopPolicy(1, 1),
    )
    evaluated = evaluate_tool_loop(
        case=ToolEvaluationCase("current", (), "done"), result=current
    )
    assert (
        evaluated.requested_tool_call_count == evaluated.executed_tool_call_count == 0
    )
    assert evaluated.tool_sequence_match is True
    assert evaluated.final_answer_match is True


def test_import_and_evaluation_have_no_runtime_io(tmp_path: Path) -> None:
    script = """
import os
import sys
sys.path.insert(0, sys.argv[1])
def guard(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.mkdir')):
        raise AssertionError('Unexpected runtime I/O')
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            raise AssertionError('Unexpected write')
sys.addaudithook(guard)
from llm_benchmark.tool_runtime import ToolRegistry, ToolRuntime
def fail(*args, **kwargs):
    raise AssertionError('Unexpected runtime execution')
ToolRegistry.__init__ = fail
ToolRuntime.__init__ = fail
ToolRuntime.execute = fail
from llm_benchmark.tool_evaluation import ToolEvaluationCase, evaluate_tool_loop
from llm_benchmark.tool_loop import ToolLoopResult, ToolLoopStopReason
from llm_benchmark.tool_conversation import ToolConversation, UserMessage
result = ToolLoopResult(ToolLoopStopReason.PROVIDER_TURN_LIMIT, ToolConversation((UserMessage('Synthetic'),)), (), ())
assert evaluate_tool_loop(case=ToolEvaluationCase('case', (), 'done'), result=result).completed is False
for name in ('providers', 'runner', 'trace', 'api', 'worker', 'db', 'tools'):
    assert 'llm_benchmark.' + name not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []

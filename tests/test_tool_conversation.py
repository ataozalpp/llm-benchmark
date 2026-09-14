import json
import traceback
from dataclasses import FrozenInstanceError

import pytest

from llm_benchmark.tool_calling import NormalizationToolTurn
from llm_benchmark.tool_conversation import (
    AssistantMessage,
    SystemMessage,
    ToolConversation,
    ToolResultMessage,
    UserMessage,
    serialize_conversation,
)
from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolErrorCode,
    ToolExecutionStatus,
    ToolResult,
)


def assistant_call(call_id: str = "call_1") -> AssistantMessage:
    call = ToolCall(
        call_id=call_id,
        tool_name="calculator",
        arguments={"operation": "add", "left": 2, "right": 3},
    )
    return AssistantMessage(
        turn=NormalizationToolTurn(
            content=None,
            tool_calls=(call,),
            finish_reason="tool_calls",
        )
    )


def tool_result(
    call_id: str = "call_1",
    tool_name: str = "calculator",
) -> ToolResultMessage:
    return ToolResultMessage(
        result=ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCEEDED,
            output={"result": 5},
        )
    )


def initial() -> ToolConversation:
    return ToolConversation(messages=(UserMessage(content="Synthetic task."),))


def test_initial_messages() -> None:
    conversation = ToolConversation(
        messages=(
            SystemMessage(content="Synthetic instruction."),
            UserMessage(content="Synthetic task."),
        )
    )

    assert serialize_conversation(conversation) == [
        {"role": "system", "content": "Synthetic instruction."},
        {"role": "user", "content": "Synthetic task."},
    ]


def test_append_preserves_original_and_pending_state() -> None:
    original = initial()
    pending = original.append(assistant_call())

    assert len(original.messages) == 1
    assert len(pending.messages) == 2

    original.validate_ready_for_provider()

    with pytest.raises(ValueError, match="not ready"):
        pending.validate_ready_for_provider()


def test_completed_tool_exchange_serializes() -> None:
    conversation = initial().append(assistant_call()).append(tool_result())

    messages = serialize_conversation(conversation)

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
    ]
    serialized_call = messages[1]["tool_calls"][0]
    assert serialized_call == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": json.dumps(
                {"operation": "add", "left": 2, "right": 3},
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    }
    assert messages[2]["tool_call_id"] == "call_1"
    assert json.loads(messages[2]["content"]) == {
        "status": "succeeded",
        "output": {"result": 5},
        "error_code": None,
    }


@pytest.mark.parametrize(
    ("call_id", "tool_name"),
    [
        ("different_call", "calculator"),
        ("call_1", "different_tool"),
    ],
)
def test_result_must_match_pending_call(
    call_id: str,
    tool_name: str,
) -> None:
    pending = initial().append(assistant_call())

    with pytest.raises(ValueError, match="match"):
        pending.append(tool_result(call_id, tool_name))


def test_duplicate_result_is_rejected() -> None:
    conversation = initial().append(assistant_call()).append(tool_result())

    with pytest.raises(ValueError, match="no pending"):
        conversation.append(tool_result())


def test_call_id_cannot_be_reused_in_later_turn() -> None:
    conversation = initial().append(assistant_call()).append(tool_result())

    with pytest.raises(ValueError, match="unique"):
        conversation.append(assistant_call())


def test_failed_tool_result_can_be_sent_to_provider() -> None:
    failure = ToolResultMessage(
        result=ToolResult(
            call_id="call_1",
            tool_name="calculator",
            status=ToolExecutionStatus.INVALID_ARGUMENTS,
            error_code=ToolErrorCode.INVALID_ARGUMENTS,
        )
    )
    conversation = initial().append(assistant_call()).append(failure)

    content = json.loads(serialize_conversation(conversation)[-1]["content"])

    assert content == {
        "status": "invalid_arguments",
        "output": None,
        "error_code": "invalid_arguments",
    }


def test_terminal_assistant_cannot_be_continued() -> None:
    final = AssistantMessage(
        turn=NormalizationToolTurn(
            content="Synthetic final text.",
            tool_calls=(),
            finish_reason="stop",
        )
    )
    conversation = initial().append(final)

    with pytest.raises(ValueError, match="not ready"):
        serialize_conversation(conversation)

    with pytest.raises(ValueError, match="terminal"):
        conversation.append(assistant_call())


def test_serialized_payload_is_independent() -> None:
    conversation = initial().append(assistant_call()).append(tool_result())

    first = serialize_conversation(conversation)
    first[0]["content"] = "Changed"
    first[1]["tool_calls"][0]["function"]["arguments"] = "{}"

    second = serialize_conversation(conversation)

    assert second[0]["content"] == "Synthetic task."
    assert json.loads(second[1]["tool_calls"][0]["function"]["arguments"]) == {
        "operation": "add",
        "left": 2,
        "right": 3,
    }


def test_multiple_calls_require_ordered_complete_results() -> None:
    turn = AssistantMessage(
        turn=NormalizationToolTurn(
            content="Two synthetic calls.",
            tool_calls=(
                assistant_call("call_1").turn.tool_calls[0],
                assistant_call("call_2").turn.tool_calls[0],
            ),
            finish_reason="tool_calls",
        )
    )
    pending = initial().append(turn)
    with pytest.raises(ValueError, match="match"):
        pending.append(tool_result("call_2"))

    partial = pending.append(tool_result("call_1"))
    assert len(partial.messages) == 3
    with pytest.raises(ValueError, match="not ready"):
        serialize_conversation(partial)
    with pytest.raises(ValueError, match="required"):
        partial.append(assistant_call("call_3"))

    complete = partial.append(tool_result("call_2"))
    messages = serialize_conversation(complete)
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert messages[1]["content"] == "Two synthetic calls."
    assert [call["id"] for call in messages[1]["tool_calls"]] == ["call_1", "call_2"]
    assert [message["tool_call_id"] for message in messages[2:]] == ["call_1", "call_2"]
    assert len(pending.messages) == 2
    assert len(partial.messages) == 3
    next_pending = complete.append(assistant_call("call_3"))
    with pytest.raises(ValueError, match="not ready"):
        next_pending.validate_ready_for_provider()


@pytest.mark.parametrize(
    "messages",
    [
        (),
        (SystemMessage("Instruction"),),
        (assistant_call(),),
        (UserMessage("Task"), tool_result()),
        (UserMessage("Task"), UserMessage("Another task")),
        (UserMessage("Task"), SystemMessage("Later instruction")),
        (UserMessage("Task"), assistant_call(), assistant_call("call_2")),
    ],
    ids=[
        "empty",
        "system-only",
        "assistant-first",
        "orphan-result",
        "second-user",
        "late-system",
        "unanswered-call",
    ],
)
def test_invalid_start_and_sequence(messages: tuple) -> None:
    with pytest.raises(ValueError):
        ToolConversation(messages=messages)


def test_conversation_is_frozen_and_repr_hides_messages() -> None:
    conversation = ToolConversation(messages=(UserMessage("PRIVATE_SENTINEL"),))
    assert "PRIVATE_SENTINEL" not in repr(conversation)
    assert "messages=" not in repr(conversation)
    with pytest.raises(FrozenInstanceError):
        conversation.messages = ()


def test_invalid_assistant_utf8_fails_safely_at_serialization() -> None:
    turn = AssistantMessage(
        turn=NormalizationToolTurn(
            content="PRIVATE_SENTINEL\ud800",
            tool_calls=assistant_call().turn.tool_calls,
            finish_reason="tool_calls",
        )
    )
    conversation = initial().append(turn).append(tool_result())
    conversation.validate_ready_for_provider()
    with pytest.raises(ValueError, match="valid UTF-8 JSON") as captured:
        serialize_conversation(conversation)
    error = captured.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "PRIVATE_SENTINEL" not in str(error)
    assert "PRIVATE_SENTINEL" not in "".join(traceback.format_exception(error))

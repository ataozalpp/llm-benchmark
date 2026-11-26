"""Synthetic software fixture, never a measurement of model quality."""

from llm_benchmark.tool_use.calling import NormalizationToolTurn
from llm_benchmark.tool_use.conversation import ToolResultMessage, UserMessage
from llm_benchmark.tool_use.provider_models import (
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_use.requests import ToolConversationRequest
from llm_benchmark.tool_use.runtime import ToolCall, ToolExecutionStatus
from llm_benchmark.tool_use.scenarios import (
    CALCULATOR_ADD_TEXT,
    DIRECT_ANSWER_TEXT,
    NO_TOOL_ARITHMETIC_TEXT,
    TWO_STEP_TEXT,
)

_DIRECT_ANSWERS = {
    DIRECT_ANSWER_TEXT: "READY",
    NO_TOOL_ARITHMETIC_TEXT: "13",
}

# The add scenario deliberately uses the second accepted argument order.
_FIRST_CALLS = {
    CALCULATOR_ADD_TEXT: ("add", 23, 17),
    TWO_STEP_TEXT: ("multiply", 6, 7),
}
_DEFAULT_FIRST_CALL = ("multiply", 17, 23)


def _calculator_turn(call_id: str, operation: str, left, right):
    return NormalizationToolTurn(
        None,
        (
            ToolCall(
                call_id=call_id,
                tool_name="calculator",
                arguments={"operation": operation, "left": left, "right": right},
            ),
        ),
        "tool_calls",
    )


class DemoToolProvider:
    """Fixed demo behavior; reads requests/observations, never evaluation gold."""

    def generate_tool_conversation(
        self, request: ToolConversationRequest
    ) -> ToolProviderResult:
        messages = request.conversation.messages
        first = messages[0]
        user_text = first.content if isinstance(first, UserMessage) else None
        last = messages[-1]

        if isinstance(last, ToolResultMessage):
            result = last.result
            output = result.output
            succeeded = (
                result.status is ToolExecutionStatus.SUCCEEDED
                and isinstance(output, dict)
                and "result" in output
            )
            tool_result_count = sum(
                isinstance(message, ToolResultMessage) for message in messages
            )
            if succeeded and user_text == TWO_STEP_TEXT and tool_result_count == 1:
                turn = _calculator_turn("demo-2", "add", output["result"], 8)
            else:
                text = str(output["result"]) if succeeded else "TOOL_FAILED"
                turn = NormalizationToolTurn(text, (), "stop")
        elif isinstance(last, UserMessage) and last.content in _DIRECT_ANSWERS:
            turn = NormalizationToolTurn(_DIRECT_ANSWERS[last.content], (), "stop")
        else:
            operation, left, right = _FIRST_CALLS.get(user_text, _DEFAULT_FIRST_CALL)
            turn = _calculator_turn("demo-1", operation, left, right)
        return ToolProviderResult(status=ToolProviderStatus.SUCCEEDED, turn=turn)

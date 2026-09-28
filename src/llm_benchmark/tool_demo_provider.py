"""Synthetic software fixture, never a measurement of model quality."""

from .tool_calling import NormalizationToolTurn
from .tool_conversation import ToolResultMessage, UserMessage
from .tool_provider_models import ToolProviderResult, ToolProviderStatus
from .tool_requests import ToolConversationRequest
from .tool_runtime import ToolCall, ToolExecutionStatus


class DemoToolProvider:
    """Fixed demo behavior; reads requests/observations, never evaluation gold."""

    def generate_tool_conversation(
        self, request: ToolConversationRequest
    ) -> ToolProviderResult:
        last = request.conversation.messages[-1]
        if isinstance(last, ToolResultMessage):
            result = last.result
            output = result.output
            text = (
                str(output["result"])
                if result.status is ToolExecutionStatus.SUCCEEDED
                and isinstance(output, dict)
                and "result" in output
                else "TOOL_FAILED"
            )
            turn = NormalizationToolTurn(text, (), "stop")
        elif (
            isinstance(last, UserMessage)
            and last.content == "Return exactly READY. Do not use any tools."
        ):
            turn = NormalizationToolTurn("READY", (), "stop")
        else:
            turn = NormalizationToolTurn(
                None,
                (
                    ToolCall(
                        call_id="demo-1",
                        tool_name="calculator",
                        arguments={
                            "operation": "multiply",
                            "left": 17,
                            "right": 23,
                        },
                    ),
                ),
                "tool_calls",
            )
        return ToolProviderResult(status=ToolProviderStatus.SUCCEEDED, turn=turn)

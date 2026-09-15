"""Contracts for bounded, synchronous tool-loop execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .tool_conversation import (
    AssistantMessage,
    ToolConversation,
    ToolResultMessage,
)
from .tool_provider_models import (
    ToolProviderResult,
    ToolProviderStatus,
)
from .tool_requests import ToolConversationRequest
from .tool_runtime import (
    ToolRegistry,
    ToolResult,
    ToolRuntime,
)


@dataclass(frozen=True)
class ToolLoopPolicy:
    max_provider_turns: int
    max_tool_calls: int

    def __post_init__(self) -> None:
        for value in (
            self.max_provider_turns,
            self.max_tool_calls,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Loop budget must be positive integers.")


class ToolLoopStopReason(StrEnum):
    FINAL_RESPONSE = "final_response"
    PROVIDER_FAILED = "provider_failed"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_TURN_LIMIT = "provider_turn_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    INVALID_CONVERSATION = "invalid_conversation"
    COMPLETION_UNCONFIRMED = "completion_unconfirmed"


class ConversationProvider(Protocol):
    def generate_tool_conversation(
        self,
        request: ToolConversationRequest,
    ) -> ToolProviderResult:
        ...


@dataclass(frozen=True)
class ToolLoopResult:
    stop_reason: ToolLoopStopReason
    conversation: ToolConversation = field(repr=False)
    provider_results: tuple[ToolProviderResult, ...] = field(repr=False)
    tool_results: tuple[ToolResult, ...] = field(repr=False)
    final_text: str | None = field(default=None ,repr=False)

    def __post_init__(self) -> None:
        if type(self.stop_reason) is not ToolLoopStopReason:
            raise TypeError("Invalid loop stop reason.")

        if type(self.conversation) is not ToolConversation:
            raise TypeError("Expected a ToolConversation.")

        if type(self.provider_results) is not tuple:
            raise TypeError("Provider results must be a tuple.")

        if any(
            type(result) is not ToolProviderResult
            for result in self.provider_results
        ):
            raise TypeError("Invalid provider result.")

        if type(self.tool_results) is not tuple:
            raise TypeError("Tool results must be a tuple.")

        if any(
            type(result) is not ToolResult
            for result in self.tool_results
        ):
            raise TypeError("Invalid tool result.")

        if self.stop_reason is ToolLoopStopReason.FINAL_RESPONSE:
            self._validate_final_response()
        elif self.final_text is not None:
            raise ValueError("Only a final-response result may contain final_text.")

    def _validate_final_response(self) -> None:
        if type(self.final_text) is not str or not self.final_text.strip():
            raise ValueError("A final-response result requires non-blank text.")

        last_message = self.conversation.messages[-1]

        if type(last_message) is not AssistantMessage:
            raise ValueError("Final text requires a terminal assistant message.")

        turn = last_message.turn

        if(
            turn.tool_calls
            or turn.content != self.final_text
            or turn.finish_reason != "stop"
        ):
            raise ValueError("Final text must match a completed assistant turn.")

    @property
    def provider_turn_count(self) -> int:
        return len(self.provider_results)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_results)


def run_tool_loop(
    *,
    provider: ConversationProvider,
    request: ToolConversationRequest,
    policy: ToolLoopPolicy,
) -> ToolLoopResult:
    """Execute a bounded, sequential conversation with selected tools."""
    if type(request) is not ToolConversationRequest:
        raise TypeError("Expected a ToolConversationRequest.")

    if type(policy) is not ToolLoopPolicy:
        raise TypeError("Expected a ToolLoopPolicy.")

    request.conversation.validate_ready_for_provider()

    registry = ToolRegistry()
    for registration in request.registrations:
        registry.register(registration)

    runtime = ToolRuntime(registry)
    allowed_names = {
        registration.definition.name
        for registration in request.registrations
    }

    conversation = request.conversation
    provider_results: list[ToolProviderResult] = []
    tool_results: list[ToolResult] = []

    def finish(
        reason:ToolLoopStopReason,
        *,
        final_text: str | None = None,
    ) -> ToolLoopResult:
        return ToolLoopResult(
            stop_reason=reason,
            conversation=conversation,
            provider_results=tuple(provider_results),
            tool_results=tuple(tool_results),
            final_text=final_text,
        )

    while len(provider_results) < policy.max_provider_turns:
        current_request = ToolConversationRequest(
            conversation=conversation,
            registrations=request.registrations,
        )

        provider_result = provider.generate_tool_conversation(current_request)

        if type(provider_result) is not ToolProviderResult:
            raise TypeError("Provider returned an invalid result.")

        provider_results.append(provider_result)

        if provider_result.status is ToolProviderStatus.REQUEST_FAILED:
            return finish(ToolLoopStopReason.PROVIDER_FAILED)

        if provider_result.status is ToolProviderStatus.RESPONSE_INVALID:
            return finish(ToolLoopStopReason.INVALID_RESPONSE)

        turn = provider_result.turn
        if turn is None:
            raise AssertionError("Successful provider result requires a turn.")

        try:
            next_conversation = conversation.append(AssistantMessage(turn))
        except ValueError:
            return finish(ToolLoopStopReason.INVALID_CONVERSATION)

        conversation = next_conversation

        if not turn.tool_calls:
            if (
                turn.finish_reason == "stop"
                and turn.content is not None
                and turn.content.strip()
            ):
                return finish(
                    ToolLoopStopReason.FINAL_RESPONSE,
                    final_text=turn.content,
                )

            return finish(
                ToolLoopStopReason.COMPLETION_UNCONFIRMED
            )

        if any(
            call.tool_name not in allowed_names
            for call in turn.tool_calls
        ):
            return finish(ToolLoopStopReason.TOOL_NOT_ALLOWED)

        remaining_calls = (
            policy.max_tool_calls - len(tool_results)
        )

        if len(turn.tool_calls) > remaining_calls:
            return finish(ToolLoopStopReason.TOOL_CALL_LIMIT)

        if len(provider_results) >= policy.max_provider_turns:
            return finish(ToolLoopStopReason.PROVIDER_TURN_LIMIT)

        for call in turn.tool_calls:
            tool_result = runtime.execute(call)
            tool_results.append(tool_result)

            conversation = conversation.append(ToolResultMessage(tool_result))
    return finish(ToolLoopStopReason.PROVIDER_TURN_LIMIT)
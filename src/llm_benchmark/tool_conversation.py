from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TypeAlias

from .tool_calling import NormalizationToolTurn
from .tool_runtime import ToolResult


def _validate_message_text(content: object) -> None:
    if type(content) is not str or not content.strip():
        raise ValueError("Message content must be a non-blank string.")

    try:
        content.encode("utf-8")
    except UnicodeError:
        pass
    else:
        return

    raise ValueError("Message content must be valid UTF-8 text.")


@dataclass(frozen=True)
class SystemMessage:
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_message_text(self.content)


@dataclass(frozen=True)
class UserMessage:
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_message_text(self.content)


@dataclass(frozen=True)
class AssistantMessage:
    turn: NormalizationToolTurn = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.turn) is not NormalizationToolTurn:
            raise TypeError("Expected a normalized assistant turn.")


@dataclass(frozen=True)
class ToolResultMessage:
    result: ToolResult = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.result) is not ToolResult:
            raise TypeError("Expected a tool result.")


ConversationMessage: TypeAlias = (
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
)


@dataclass(frozen=True)
class ToolConversation:
    messages: tuple[ConversationMessage, ...] = field(repr= False)

    def __post_init__(self) -> None:
        if type(self.messages) is not tuple:
            raise TypeError("Conversation message must be a tuple.")

        if not self.messages:
            raise ValueError("Conversation must not be empty.")

        self._validate_sequence()

    def _validate_sequence(self) -> bool:
        allowed_types = (
            SystemMessage,
            UserMessage,
            AssistantMessage,
            ToolResultMessage,
        )

        for message in self.messages:
            if type(message) not in allowed_types:
                raise TypeError("Unsupported conversation message type.")

        index = 0

        if type(self.messages[0]) is SystemMessage:
            index = 1

        if index >= len(self.messages) or type(self.messages[index]) is not UserMessage:
            raise ValueError(
                "Conversation must begin with an optional system and a user message."
            )

        seen_call_ids: set[str] = set()
        pending_calls: list[tuple[str, str]] = []
        finished = False

        for message in self.messages[index + 1 :]:
            if finished:
                raise ValueError("Messages cannot follow a terminal assistant message.")

            if type(message) is AssistantMessage:
                if pending_calls:
                    raise ValueError(
                        "Tool results are required before another assistant message."
                    )

                calls = message.turn.tool_calls

                if not calls:
                    finished = True
                    continue

                for call in calls:
                    if call.call_id in seen_call_ids:
                        raise ValueError(
                            "Tool call IDs must be unique within a conversation."
                        )

                    seen_call_ids.add(call.call_id)
                    pending_calls.append((call.call_id, call.tool_name))

            elif type(message) is ToolResultMessage:
                if not pending_calls:
                    raise ValueError("Tool result has no pending call.")

                expected_id, expected_name = pending_calls[0]
                result = message.result

                if result.call_id != expected_id or result.tool_name != expected_name:
                    raise ValueError("Tool result must match the next pending call.")

                pending_calls.pop(0)

            else:
                raise ValueError(
                    "System and user messages are allowed only at the beginning."
                )

        return not finished and not pending_calls

    def validate_ready_for_provider(self) -> None:
        if not self._validate_sequence():
            raise ValueError("Conversation is not ready for a provider request.")

    def append(self, message: ConversationMessage) -> ToolConversation:
        return ToolConversation(
            messages=(*self.messages, message),
        )


def _json_text(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        pass
    else:
        return encoded

    raise ValueError("Conversation content must be valid UTF-8 JSON.")


def serialize_conversation(
    conversation: ToolConversation,
) -> list[dict[str, object]]:
    if type(conversation) is not ToolConversation:
        raise TypeError("Expected a ToolConversation.")

    conversation.validate_ready_for_provider()

    messages: list[dict[str, object]] = []

    for message in conversation.messages:
        if type(message) is SystemMessage:
            messages.append(
                {
                    "role": "system",
                    "content": message.content,
                }
            )

        elif type(message) is UserMessage:
            messages.append(
                {
                    "role": "user",
                    "content": message.content,
                }
            )

        elif type(message) is AssistantMessage:
            turn = message.turn

            serialized: dict[str, object] = {
                "role": "assistant",
                "content": turn.content,
            }

            if turn.tool_calls:
                serialized["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": _json_text(call.arguments),
                        },
                    }
                    for call in turn.tool_calls
                ]
            messages.append(serialized)

        elif type(message) is ToolResultMessage:
            result = message.result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": _json_text(
                        {
                            "status": result.status.value,
                            "output": result.output,
                            "error_code": (
                                result.error_code.value
                                if result.error_code is not None
                                else None
                            ),
                        }
                    ),
                }
            )
    _json_text(messages)

    return messages

"""Pure request mapping for initial and conversation-based tool turns."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .config import ModelConfig
from .tool_conversation import (
    ToolConversation,
    serialize_conversation,
)
from .tool_descriptors import (
    ToolDescriptor,
    serialize_tool_descriptor,
)
from .tool_runtime import ToolRegistration


def _validate_text(value: object, *, optional: bool = False) -> None:
    if value is None and optional:
        return

    if type(value) is not str or not value.strip():
        raise ValueError("Message content must be a non-blank string.")

    try:
        value.encode("utf-8")
    except UnicodeError:
        pass
    else:
        return

    raise ValueError("Message content must be valid UTF-8 text.")


def _validate_registrations(
    registrations: tuple[ToolRegistration, ...],
) -> None:
    if type(registrations) is not tuple:
        raise TypeError("Tool registrations must be a tuple.")

    if not registrations:
        raise ValueError("At least one tool registration is required.")

    names: set[str] = set()

    for registration in registrations:
        if type(registration) is not ToolRegistration:
            raise TypeError("Expected a ToolRegistration.")

        name = registration.definition.name

        if name in names:
            raise ValueError("Tool registration names must be unique.")

        names.add(name)


def _validate_descriptors(
    descriptors: tuple[ToolDescriptor, ...],
) -> None:
    if type(descriptors) is not tuple:
        raise TypeError("Tool descriptors must be a tuple.")

    if not descriptors:
        raise ValueError("At least one tool descriptor is required.")

    names: set[str] = set()

    for descriptor in descriptors:
        if type(descriptor) is not ToolDescriptor:
            raise TypeError("Expected a ToolDescriptor.")

        name = descriptor.definition.name

        if name in names:
            raise ValueError("Tool descriptor names must be unique.")

        names.add(name)


def _validate_tool_sources(
    registrations: tuple[ToolRegistration, ...],
    descriptors: tuple[ToolDescriptor, ...],
) -> None:
    if type(registrations) is not tuple:
        raise TypeError("Tool registrations must be a tuple.")

    if type(descriptors) is not tuple:
        raise TypeError("Tool descriptors must be a tuple.")

    if registrations and descriptors:
        raise ValueError("Select exactly one tool source.")

    if not registrations and not descriptors:
        raise ValueError("At least one tool selection is required.")

    if registrations:
        _validate_registrations(registrations)
    else:
        _validate_descriptors(descriptors)


def _schema_snapshot(registration: ToolRegistration) -> dict[str, object]:
    schema = registration.argument_model.model_json_schema(mode="validation")

    if type(schema) is not dict or schema.get("type") != "object":
        raise ValueError("Tool argument schema must describe an object.")

    # Schema hooks are trusted application code. The JSON round trip creates
    # an independent snapshot and rejects non-finite or non-serializable data.
    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded.encode("utf-8")
        snapshot = json.loads(encoded)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        pass
    else:
        return snapshot

    raise ValueError("Tool argument schema must be valid JSON.")


def serialize_tool_registration(
    registration: ToolRegistration,
) -> dict[str, object]:
    """Map one trusted registration without executing its handler."""

    if type(registration) is not ToolRegistration:
        raise TypeError("Expected a ToolRegistration.")

    return {
        "type": "function",
        "function": {
            "name": registration.definition.name,
            "description": registration.definition.description,
            "parameters": _schema_snapshot(registration),
        },
    }


def serialize_tool_registrations(
    registrations: tuple[ToolRegistration, ...],
) -> list[dict[str, object]]:
    """Map unique registrations in deterministic tool-name order."""

    _validate_registrations(registrations)

    ordered = sorted(
        registrations,
        key=lambda registration: registration.definition.name,
    )

    return [serialize_tool_registration(registration) for registration in ordered]


def serialize_tool_descriptors(
    descriptors: tuple[ToolDescriptor, ...],
) -> list[dict[str, object]]:
    """Map unique descriptors in deterministic tool-name order."""

    _validate_descriptors(descriptors)

    ordered = sorted(
        descriptors,
        key=lambda descriptor: descriptor.definition.name,
    )

    return [serialize_tool_descriptor(descriptor) for descriptor in ordered]


def _serialize_selected_tools(
    registrations: tuple[ToolRegistration, ...],
    descriptors: tuple[ToolDescriptor, ...],
) -> list[dict[str, object]]:
    _validate_tool_sources(registrations, descriptors)

    if registrations:
        return serialize_tool_registrations(registrations)

    return serialize_tool_descriptors(descriptors)


@dataclass(frozen=True)
class ToolTurnRequest:
    """Initial text messages and exactly one non empty tool source."""

    user_content: str = field(repr=False)
    registrations: tuple[ToolRegistration, ...] = field(
        default=(),
        repr=False,
    )
    system_content: str | None = field(
        default=None,
        repr=False,
    )
    descriptors: tuple[ToolDescriptor, ...] = field(
        default=(),
        repr=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_text(self.user_content)
        _validate_text(self.system_content, optional=True)
        _validate_tool_sources(
            self.registrations,
            self.descriptors,
        )


@dataclass(frozen=True)
class ToolConversationRequest:
    """A provider-ready conversation and one non empty tool source."""

    conversation: ToolConversation = field(repr=False)
    registrations: tuple[ToolRegistration, ...] = field(
        default=(),
        repr=False,
    )
    descriptors: tuple[ToolDescriptor, ...] = field(
        default=(),
        repr=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if type(self.conversation) is not ToolConversation:
            raise TypeError("Expected a ToolConversation.")

        _validate_tool_sources(
            self.registrations,
            self.descriptors,
        )

        self.conversation.validate_ready_for_provider()


def _build_tool_payload(
    config: ModelConfig,
    messages: list[dict[str, object]],
    registrations: tuple[ToolRegistration, ...],
    descriptors: tuple[ToolDescriptor, ...],
) -> dict[str, object]:
    if type(config) is not ModelConfig:
        raise TypeError("Expected a ModelConfig.")

    if config.provider != "openai_compatible":
        raise ValueError("Tool payload mapping requires openai_compatible.")

    unsupported_values = (
        config.reasoning,
        config.top_k,
        config.min_p,
        config.repeat_penalty,
    )
    if any(value is not None for value in unsupported_values):
        raise ValueError("Unsupported tool-request generation setting.")

    if not math.isfinite(config.temperature):
        raise ValueError("Temperature must be finite.")

    if config.top_p is not None and not math.isfinite(config.top_p):
        raise ValueError("Top-p must be finite.")

    payload: dict[str, object] = {
        "model": config.model_id,
        "messages": messages,
        "tools": _serialize_selected_tools(
            registrations,
            descriptors,
        ),
        "temperature": config.temperature,
        "stream": False,
    }

    if config.max_output_tokens is not None:
        payload["max_tokens"] = config.max_output_tokens

    if config.top_p is not None:
        payload["top_p"] = config.top_p

    return payload


def build_openai_tool_payload(
    config: ModelConfig,
    request: ToolTurnRequest,
) -> dict[str, object]:
    """Build one non-streaming initial-turn payload without runtime I/O.

    Local schema hooks are trusted code. Descriptor schemas are already
    snapshotted and bounded by their own contract. This builder does not
    enforce an aggregate payload or message-size budget.

    Returned payloads are independent, not secret-redacted. This is not an
    endpoint compatibility check.
    """

    if type(request) is not ToolTurnRequest:
        raise TypeError("Expected a ToolTurnRequest.")

    messages: list[dict[str, object]] = []

    if request.system_content is not None:
        messages.append(
            {
                "role": "system",
                "content": request.system_content,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": request.user_content,
        }
    )

    return _build_tool_payload(
        config,
        messages,
        request.registrations,
        request.descriptors,
    )


def build_openai_tool_conversation_payload(
    config: ModelConfig,
    request: ToolConversationRequest,
) -> dict[str, object]:
    """Map a ready conversation without provider calls or tool execution."""
    if type(request) is not ToolConversationRequest:
        raise TypeError("Expected a ToolConversationRequest.")

    messages = serialize_conversation(request.conversation)

    return _build_tool_payload(
        config,
        messages,
        request.registrations,
        request.descriptors,
    )

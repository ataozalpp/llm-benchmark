"""Pure request mapping for an initial OpenAI-compatible tool turn."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .config import ModelConfig
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


@dataclass(frozen=True)
class ToolTurnRequest:
    """Initial text messages and an explicit selection of trusted tools."""

    user_content: str = field(repr=False)
    registrations: tuple[ToolRegistration, ...] = field(repr=False)
    system_content: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_text(self.user_content)
        _validate_text(self.system_content, optional=True)
        _validate_registrations(self.registrations)


def build_openai_tool_payload(
    config: ModelConfig,
    request: ToolTurnRequest,
) -> dict[str, object]:
    """Build one non-streaming initial-turn payload without runtime I/O.

    Schema hooks are trusted code; this is not a schema sandbox or an endpoint
    compatibility check. Returned payloads are independent, not secret-redacted.
    No message/schema size budget is enforced here.
    """

    if type(config) is not ModelConfig:
        raise TypeError("Expected a ModelConfig.")

    if type(request) is not ToolTurnRequest:
        raise TypeError("Expected a ToolTurnRequest.")

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

    messages: list[dict[str, str]] = []

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

    payload: dict[str, object] = {
        "model": config.model_id,
        "messages": messages,
        "tools": serialize_tool_registrations(request.registrations),
        "temperature": config.temperature,
        "stream": False,
    }

    if config.max_output_tokens is not None:
        payload["max_tokens"] = config.max_output_tokens

    if config.top_p is not None:
        payload["top_p"] = config.top_p

    return payload

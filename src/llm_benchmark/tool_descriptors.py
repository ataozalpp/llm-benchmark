"""Immutable tool descriptions independent of executable handlers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .tool_runtime import ToolDefinition, ToolRegistration

MAX_TOOL_SCHEMA_BYTES = 65_536
MAX_TOOL_SCHEMA_DEPTH = 32


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > MAX_TOOL_SCHEMA_DEPTH:
        raise ValueError("Tool schema exceeds the nesting limit.")

    if value is None or type(value) in (bool, int):
        return

    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Tool schema must contain finite numbers.")
        return

    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ValueError("Tool schema must contain valid UTF-8 text.") from None
        return

    if type(value) is list:
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return

    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("Tool schema keys must be strings.")

            _validate_json_value(key, depth=depth + 1)
            _validate_json_value(item, depth=depth + 1)
        return

    raise ValueError("Tool schema must contain strict JSON values.")


def _snapshot_schema(schema: object) -> str:
    if type(schema) is not dict:
        raise TypeError("Tool schema must be an object.")

    if schema.get("type") != "object":
        raise ValueError("Tool schema must describe object arguments.")

    _validate_json_value(schema)

    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        size = len(encoded.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("Tool schema cannot be serialized.") from None

    if size > MAX_TOOL_SCHEMA_BYTES:
        raise ValueError("Tool schema exceeds the size limit.")

    return encoded


@dataclass(frozen=True, init=False)
class ToolDescriptor:
    """Bounded JSON snapshot, not a JSON Schema semantic validator.

    Depth starts at zero for the root and includes keys and scalar leaves.
    References are retained as data, never resolved. Size is measured after
    serialization; this is not a transport or peak-memory limit.
    """

    definition: ToolDefinition
    _parameters_json: str = field(repr=False)

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        parameters: object,
    ) -> None:
        if type(definition) is not ToolDefinition:
            raise TypeError("Expected a ToolDefinition.")

        snapshot = _snapshot_schema(parameters)

        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "_parameters_json", snapshot)

    @property
    def parameters(self) -> dict[str, object]:
        return json.loads(self._parameters_json)


def descriptor_from_registration(
    registration: ToolRegistration,
) -> ToolDescriptor:
    """Describe a trusted local registration without executing its handler.

    Pydantic schema hooks are trusted code; their exceptions propagate.
    """

    if type(registration) is not ToolRegistration:
        raise TypeError("Expected a ToolRegistration.")

    schema = registration.argument_model.model_json_schema(mode="validation")

    return ToolDescriptor(
        definition=registration.definition,
        parameters=schema,
    )


def serialize_tool_descriptor(
    descriptor: ToolDescriptor,
) -> dict[str, object]:
    """Build an independent OpenAI-compatible function description."""

    if type(descriptor) is not ToolDescriptor:
        raise TypeError("Expected a ToolDescriptor.")

    return {
        "type": "function",
        "function": {
            "name": descriptor.definition.name,
            "description": descriptor.definition.description,
            "parameters": descriptor.parameters,
        },
    }

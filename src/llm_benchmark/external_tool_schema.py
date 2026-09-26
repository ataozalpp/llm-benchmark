"""Bounded-profile validation for external tool parameter schemas.

Uses Draft 2020-12 semantics for the explicit keyword subset below. References,
regexes, formats, and schema composition are unsupported. Existing descriptor
bounds apply; this module does not provide a hard validation-time limit.
"""

from __future__ import annotations

from enum import StrEnum

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .tool_descriptors import ToolDescriptor
from .tool_runtime import ToolCall


class ExternalToolSchemaCode(StrEnum):
    INVALID_SCHEMA = "invalid_schema"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    INVALID_ARGUMENTS = "invalid_arguments"


_MESSAGES = {
    ExternalToolSchemaCode.INVALID_SCHEMA: "External tool schema is invalid.",
    ExternalToolSchemaCode.UNSUPPORTED_SCHEMA: "External tool schema uses unsupported features.",
    ExternalToolSchemaCode.INVALID_ARGUMENTS: "Tool arguments do not match the external schema.",
}


class ExternalToolSchemaError(ValueError):
    def __init__(self, code: ExternalToolSchemaCode) -> None:
        if type(code) is not ExternalToolSchemaCode:
            raise TypeError("Expected an ExternalToolSchemaCode.")

        self.code = code
        super().__init__(_MESSAGES[code])


_SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "title",
        "description",
        "default",
        "examples",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }
)


def _check_supported_keywords(schema: object) -> None:
    # Boolean schemas may appear in nested schema positions.
    if type(schema) is bool:
        return

    if type(schema) is not dict:
        raise ExternalToolSchemaError(ExternalToolSchemaCode.INVALID_SCHEMA)

    if set(schema) - _SUPPORTED_KEYWORDS:
        raise ExternalToolSchemaError(ExternalToolSchemaCode.UNSUPPORTED_SCHEMA)

    properties = schema.get("properties", {})
    if type(properties) is not dict:
        raise ExternalToolSchemaError(ExternalToolSchemaCode.INVALID_SCHEMA)

    for child_schema in properties.values():
        _check_supported_keywords(child_schema)

    for keyword in ("items", "additionalProperties"):
        if keyword in schema:
            _check_supported_keywords(schema[keyword])


def _validated_schema(
    descriptor: ToolDescriptor,
) -> dict[str, object]:
    if type(descriptor) is not ToolDescriptor:
        raise TypeError("Expected a ToolDescriptor.")

    schema = descriptor.parameters

    _check_supported_keywords(schema)

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise ExternalToolSchemaError(ExternalToolSchemaCode.INVALID_SCHEMA) from None

    return schema


def validate_external_tool_schema(
    descriptor: ToolDescriptor,
) -> None:
    """Validate the supported schema profile without execution or IO."""
    _validated_schema(descriptor)


def validate_external_tool_arguments(
    *,
    descriptor: ToolDescriptor,
    call: ToolCall,
) -> None:
    """Validate immutable call arguments without coercion or execution."""
    if type(call) is not ToolCall:
        raise TypeError("Expected a ToolCall.")

    schema = _validated_schema(descriptor)

    if call.tool_name != descriptor.definition.name:
        raise ValueError("Tool call does not match the descriptor.")

    try:
        Draft202012Validator(schema).validate(call.arguments)
    except ValidationError:
        raise ExternalToolSchemaError(
            ExternalToolSchemaCode.INVALID_ARGUMENTS
        ) from None

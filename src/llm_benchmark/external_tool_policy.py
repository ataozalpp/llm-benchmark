"""Pure admission policy for immutable external tool descriptors.

This module checks names, collection size, and aggregate schema bytes.
It does not validate JSON Schema semantics or execute tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .tool_descriptors import ToolDescriptor
from .tool_runtime import ToolDefinition


class ExternalToolAdmissionCode(StrEnum):
    EMPTY_COLLECTION = "empty_collection"
    TOO_MANY_TOOLS = "too_many_tools"
    DUPLICATE_TOOL_NAME = "duplicate_tool_name"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    SCHEMA_BUDGET_EXCEEDED = "schema_budget_exceeded"


_ADMISSION_MESSAGES = {
    ExternalToolAdmissionCode.EMPTY_COLLECTION: "At least one external tool is required.",
    ExternalToolAdmissionCode.TOO_MANY_TOOLS: "External tool count exceeds the policy limit.",
    ExternalToolAdmissionCode.DUPLICATE_TOOL_NAME: "External tool names must be unique.",
    ExternalToolAdmissionCode.TOOL_NOT_ALLOWED: "An external tool is not allowed by policy.",
    ExternalToolAdmissionCode.SCHEMA_BUDGET_EXCEEDED: "External tool schemas exceed the policy budget.",
}


class ExternalToolAdmissionError(ValueError):
    def __init__(self, code: ExternalToolAdmissionCode) -> None:
        if type(code) is not ExternalToolAdmissionCode:
            raise TypeError("Expected an ExternalToolAdmissionCode.")

        self.code = code
        super().__init__(_ADMISSION_MESSAGES[code])


@dataclass(frozen=True)
class ExternalToolPolicy:
    """Server-owned limits; an empty allowlist denies every tool.

    The byte budget covers canonical UTF-8 parameter schemas only, not the
    complete request, tool descriptions, transport input, or peak memory.
    """

    allowed_tool_names: tuple[str, ...] = field(repr=False)
    max_tools: int
    max_total_schema_bytes: int

    def __post_init__(self) -> None:
        if type(self.allowed_tool_names) is not tuple:
            raise TypeError("Allowed tool names must be a tuple.")

        for value in (
            self.max_tools,
            self.max_total_schema_bytes,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Policy limits must be positive integers.")

        seen: set[str] = set()

        for name in self.allowed_tool_names:
            ToolDefinition(
                name=name,
                description="External tool policy entry.",
            )

            if name in seen:
                raise ValueError("Allowed tool names must be unique.")

            seen.add(name)


def _schema_size_bytes(descriptor: ToolDescriptor) -> int:
    encoded = json.dumps(
        descriptor.parameters,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return len(encoded.encode("utf-8"))


def admit_tool_descriptors(
    *,
    descriptors: tuple[ToolDescriptor, ...],
    policy: ExternalToolPolicy,
) -> tuple[ToolDescriptor, ...]:
    """Admit an entire descriptor collection without execution or I/O."""

    if type(policy) is not ExternalToolPolicy:
        raise TypeError("Expected an ExternalToolPolicy.")

    if type(descriptors) is not tuple:
        raise TypeError("Tool descriptors must be a tuple.")

    if not descriptors:
        raise ExternalToolAdmissionError(ExternalToolAdmissionCode.EMPTY_COLLECTION)

    if len(descriptors) > policy.max_tools:
        raise ExternalToolAdmissionError(ExternalToolAdmissionCode.TOO_MANY_TOOLS)

    if any(type(descriptor) is not ToolDescriptor for descriptor in descriptors):
        raise TypeError("Expected a ToolDescriptor.")

    names = tuple(descriptor.definition.name for descriptor in descriptors)

    if len(set(names)) != len(names):
        raise ExternalToolAdmissionError(ExternalToolAdmissionCode.DUPLICATE_TOOL_NAME)

    allowed_names = set(policy.allowed_tool_names)

    if any(name not in allowed_names for name in names):
        raise ExternalToolAdmissionError(ExternalToolAdmissionCode.TOOL_NOT_ALLOWED)

    total_schema_bytes = 0

    for descriptor in descriptors:
        total_schema_bytes += _schema_size_bytes(descriptor)

        if total_schema_bytes > policy.max_total_schema_bytes:
            raise ExternalToolAdmissionError(
                ExternalToolAdmissionCode.SCHEMA_BUDGET_EXCEEDED
            )

    return tuple(
        sorted(
            descriptors,
            key=lambda descriptor: descriptor.definition.name,
        )
    )

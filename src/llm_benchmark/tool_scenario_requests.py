"""Prepare tool-conversation requests from scenario definitions."""

from __future__ import annotations

from .tool_conversation import ToolConversation, UserMessage
from .tool_descriptors import ToolDescriptor
from .tool_requests import ToolConversationRequest
from .tool_runtime import ToolRegistration, ToolRegistry
from .tool_scenarios import ToolScenario


def build_tool_scenario_request(
    *,
    scenario: ToolScenario,
    registry: ToolRegistry,
) -> ToolConversationRequest:
    """Resolve selected tools without provider calls or execution."""

    if type(scenario) is not ToolScenario:
        raise TypeError("Expected a ToolScenario.")

    if type(registry) is not ToolRegistry:
        raise TypeError("Expected a ToolRegistry.")

    registrations: list[ToolRegistration] = []

    for name in scenario.available_tools:
        registration = registry.get(name)

        if registration is None:
            raise ValueError("A selected scenario tool is not registered.")

        registrations.append(registration)

    conversation = ToolConversation(messages=(UserMessage(scenario.user_text),))

    return ToolConversationRequest(
        conversation=conversation,
        registrations=tuple(registrations),
    )


def build_descriptor_scenario_request(
    *,
    scenario: ToolScenario,
    descriptors: tuple[ToolDescriptor, ...],
) -> ToolConversationRequest:
    """Select scenario tools from an immutable descriptor collection."""

    if type(scenario) is not ToolScenario:
        raise TypeError("Expected a ToolScenario.")

    if type(descriptors) is not tuple:
        raise TypeError("Tool descriptors must be a tuple.")

    if not descriptors:
        raise ValueError("At least one tool descriptor is required.")

    by_name: dict[str, ToolDescriptor] = {}

    for descriptor in descriptors:
        if type(descriptor) is not ToolDescriptor:
            raise TypeError("Expected a ToolDescriptor.")

        name = descriptor.definition.name

        if name in by_name:
            raise ValueError("Tool descriptor names must be unique.")

        by_name[name] = descriptor

    selected: list[ToolDescriptor] = []

    for name in scenario.available_tools:
        descriptor = by_name.get(name)

        if descriptor is None:
            raise ValueError("A selected scenario tool is not available.")

        selected.append(descriptor)

    return ToolConversationRequest(
        conversation=ToolConversation(messages=(UserMessage(scenario.user_text),)),
        descriptors=tuple(selected),
    )

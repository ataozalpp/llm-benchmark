"""Prepare tool-conversation requests from scenario definitions."""

from __future__ import annotations

from .tool_conversation import ToolConversation, UserMessage
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

    conversation = ToolConversation(
        messages=(
            UserMessage(scenario.user_text),
        )
    )

    return ToolConversationRequest(
        conversation=conversation,
        registrations=tuple(registrations),
    )
from dataclasses import replace

import pytest

from llm_benchmark.config import ModelConfig
from llm_benchmark.tool_conversation import UserMessage
from llm_benchmark.tool_evaluation import ExpectedToolCall, ToolEvaluationCase
from llm_benchmark.tool_requests import build_openai_tool_conversation_payload
from llm_benchmark.tool_runtime import ToolRegistry, ToolRuntime
from llm_benchmark.tool_scenario_requests import (
    build_tool_scenario_request,
)
from llm_benchmark.tool_scenarios import (
    create_example_tool_scenarios,
)
from llm_benchmark.tools import create_example_tool_registry


def test_request_contains_user_text_and_only_selected_tools() -> None:
    scenario = create_example_tool_scenarios()[0]
    registry = create_example_tool_registry()

    request = build_tool_scenario_request(
        scenario=scenario,
        registry=registry,
    )

    assert request.conversation.messages == (UserMessage(scenario.user_text),)
    assert tuple(
        registration.definition.name for registration in request.registrations
    ) == ("calculator",)

    assert registry.get("lookup_city_code") is not None
    assert len(request.registrations) == 1


def test_expectation_changes_do_not_change_provider_messages() -> None:
    original = create_example_tool_scenarios()[1]

    changed = replace(
        original,
        evaluation=ToolEvaluationCase(
            case_id="different-case",
            expected_calls=(),
            expected_final_text="PRIVATE_EXPECTED_SENTINEL",
        ),
    )

    registry = create_example_tool_registry()

    first = build_tool_scenario_request(
        scenario=original,
        registry=registry,
    )
    second = build_tool_scenario_request(
        scenario=changed,
        registry=registry,
    )

    assert first == second
    assert second.conversation.messages == (UserMessage(original.user_text),)


def test_unknown_selected_tool_is_rejected() -> None:
    scenario = create_example_tool_scenarios()[0]

    with pytest.raises(ValueError, match="not registered"):
        build_tool_scenario_request(
            scenario=scenario,
            registry=ToolRegistry(),
        )


def test_request_building_does_not_execute_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = create_example_tool_scenarios()[0]
    registry = create_example_tool_registry()

    def forbidden(*args, **kwargs):
        pytest.fail("Request building must not execute tools.")

    monkeypatch.setattr(ToolRuntime, "__init__", forbidden)
    monkeypatch.setattr(ToolRuntime, "execute", forbidden)

    request = build_tool_scenario_request(
        scenario=scenario,
        registry=registry,
    )

    assert len(request.registrations) == 1


def test_each_request_has_a_fresh_conversation() -> None:
    scenario = create_example_tool_scenarios()[0]
    registry = create_example_tool_registry()

    first = build_tool_scenario_request(
        scenario=scenario,
        registry=registry,
    )
    second = build_tool_scenario_request(
        scenario=scenario,
        registry=registry,
    )

    assert first == second
    assert first.conversation is not second.conversation


@pytest.mark.parametrize("field", ["scenario", "registry"])
def test_invalid_input_types_are_rejected(field: str) -> None:
    values = {
        "scenario": create_example_tool_scenarios()[0],
        "registry": create_example_tool_registry(),
    }
    values[field] = None

    with pytest.raises(TypeError):
        build_tool_scenario_request(**values)


def test_gold_arguments_and_answer_do_not_enter_serialized_payload() -> None:
    original = create_example_tool_scenarios()[0]
    changed = replace(
        original,
        evaluation=ToolEvaluationCase(
            case_id="PRIVATE_CASE_SENTINEL",
            expected_calls=(
                ExpectedToolCall(
                    tool_name="calculator",
                    arguments={"gold": "PRIVATE_ARGUMENT_SENTINEL"},
                ),
            ),
            expected_final_text="PRIVATE_ANSWER_SENTINEL",
        ),
    )
    registry = create_example_tool_registry()
    config = ModelConfig(
        provider="openai_compatible",
        model_id="synthetic",
        base_url="http://127.0.0.1:1234/v1",
    )
    payloads = [
        build_openai_tool_conversation_payload(
            config,
            build_tool_scenario_request(scenario=scenario, registry=registry),
        )
        for scenario in (original, changed)
    ]
    assert payloads[0] == payloads[1]
    for sentinel in (
        "PRIVATE_CASE_SENTINEL",
        "PRIVATE_ARGUMENT_SENTINEL",
        "PRIVATE_ANSWER_SENTINEL",
    ):
        assert sentinel not in str(payloads[1])


def test_selection_preserves_order_and_does_not_modify_registry() -> None:
    registry = create_example_tool_registry()
    before = registry.list_definitions()
    scenario = replace(
        create_example_tool_scenarios()[0],
        available_tools=("lookup_city_code", "calculator"),
    )
    request = build_tool_scenario_request(scenario=scenario, registry=registry)
    assert (
        tuple(r.definition.name for r in request.registrations)
        == scenario.available_tools
    )
    assert registry.list_definitions() == before
    assert request.registrations[0] is registry.get("lookup_city_code")


def test_later_missing_tool_does_not_initialize_runtime(monkeypatch) -> None:
    registry = create_example_tool_registry()
    before = registry.list_definitions()
    scenario = replace(
        create_example_tool_scenarios()[0],
        available_tools=("calculator", "missing_tool"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Request preparation must not start execution")

    monkeypatch.setattr(ToolRuntime, "__init__", forbidden)
    with pytest.raises(ValueError) as caught:
        build_tool_scenario_request(scenario=scenario, registry=registry)
    assert str(caught.value) == "A selected scenario tool is not registered."
    assert registry.list_definitions() == before


def test_mutating_payload_does_not_change_request_or_scenario() -> None:
    scenario = create_example_tool_scenarios()[0]
    request = build_tool_scenario_request(
        scenario=scenario, registry=create_example_tool_registry()
    )
    config = ModelConfig(
        provider="openai_compatible",
        model_id="synthetic",
        base_url="http://127.0.0.1:1234/v1",
    )
    payload = build_openai_tool_conversation_payload(config, request)
    payload["messages"][0]["content"] = "changed"
    assert request.conversation.messages == (UserMessage(scenario.user_text),)
    assert (
        build_openai_tool_conversation_payload(config, request)["messages"][0][
            "content"
        ]
        == scenario.user_text
    )

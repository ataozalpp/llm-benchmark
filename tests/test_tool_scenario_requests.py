from dataclasses import replace

import pytest

from llm_benchmark.config import ModelConfig
from llm_benchmark.tool_conversation import UserMessage
from llm_benchmark.tool_descriptors import descriptor_from_registration
from llm_benchmark.tool_evaluation import ExpectedToolCall, ToolEvaluationCase
from llm_benchmark.tool_requests import build_openai_tool_conversation_payload
from llm_benchmark.tool_runtime import ToolRegistry, ToolRuntime
from llm_benchmark.tool_scenario_requests import (
    build_descriptor_scenario_request,
    build_tool_scenario_request,
)
from llm_benchmark.tool_scenarios import (
    create_example_tool_scenarios,
)
from llm_benchmark.tools import create_example_tool_registry


def descriptors():
    registry = create_example_tool_registry()
    return tuple(
        descriptor_from_registration(registry.get(d.name))
        for d in registry.list_definitions()
    )


def test_descriptor_selection_order_freshness_and_no_runtime(monkeypatch):
    available = descriptors()
    original = create_example_tool_scenarios()[0]
    scenario = replace(original, available_tools=("lookup_city_code", "calculator"))

    def reject(*args, **kwargs):
        pytest.fail("No runtime may initialize")

    monkeypatch.setattr(ToolRuntime, "__init__", reject)
    first = build_descriptor_scenario_request(scenario=scenario, descriptors=available)
    second = build_descriptor_scenario_request(scenario=scenario, descriptors=available)
    assert first == second
    assert first.conversation is not second.conversation
    assert first.registrations == ()
    assert (
        tuple(d.definition.name for d in first.descriptors) == scenario.available_tools
    )
    subset = build_descriptor_scenario_request(scenario=original, descriptors=available)
    assert tuple(d.definition.name for d in subset.descriptors) == ("calculator",)
    assert available == descriptors()


@pytest.mark.parametrize(
    "value,error",
    [
        (None, TypeError),
        ([], TypeError),
        ((object(),), TypeError),
        ((), ValueError),
    ],
)
def test_invalid_descriptor_collection(value, error):
    with pytest.raises(error):
        build_descriptor_scenario_request(
            scenario=create_example_tool_scenarios()[0], descriptors=value
        )


def test_invalid_descriptor_scenario():
    with pytest.raises(TypeError, match="ToolScenario"):
        build_descriptor_scenario_request(scenario=None, descriptors=descriptors())


def test_duplicate_unselected_descriptors_rejected():
    available = descriptors()
    city = next(d for d in available if d.definition.name == "lookup_city_code")
    with pytest.raises(ValueError, match="unique"):
        build_descriptor_scenario_request(
            scenario=create_example_tool_scenarios()[0], descriptors=available + (city,)
        )


def test_missing_descriptor_error_does_not_echo_name():
    scenario = replace(
        create_example_tool_scenarios()[1], available_tools=("private_missing",)
    )
    with pytest.raises(ValueError) as caught:
        build_descriptor_scenario_request(scenario=scenario, descriptors=descriptors())
    assert str(caught.value) == "A selected scenario tool is not available."


def test_descriptor_gold_separation_payload_parity_and_mutation():
    original = create_example_tool_scenarios()[0]
    changed = replace(
        original,
        evaluation=ToolEvaluationCase(
            case_id="PRIVATE_CASE",
            expected_calls=(
                ExpectedToolCall(
                    tool_name="calculator",
                    arguments={"private": "PRIVATE_ARGUMENT"},
                ),
            ),
            expected_final_text="PRIVATE_ANSWER",
        ),
    )
    config = ModelConfig(
        provider="openai_compatible",
        model_id="synthetic",
        base_url="http://127.0.0.1:1234/v1",
    )
    available = descriptors()
    requests = [
        build_descriptor_scenario_request(scenario=s, descriptors=available)
        for s in (original, changed)
    ]
    payloads = [build_openai_tool_conversation_payload(config, r) for r in requests]
    assert payloads[0] == payloads[1]
    assert "PRIVATE_" not in str(payloads[1])
    local = build_tool_scenario_request(
        scenario=original, registry=create_example_tool_registry()
    )
    assert payloads[0] == build_openai_tool_conversation_payload(config, local)
    payloads[0]["messages"][0]["content"] = "changed"
    payloads[0]["tools"][0]["function"]["parameters"]["properties"].clear()
    assert build_openai_tool_conversation_payload(config, requests[0]) == payloads[1]


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

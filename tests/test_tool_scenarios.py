from dataclasses import FrozenInstanceError, replace

import pytest

from llm_benchmark.tool_evaluation import (
    ExpectedToolCall,
    ToolEvaluationCase,
)
from llm_benchmark.tool_scenarios import (
    create_example_tool_scenarios,
)


def test_example_scenarios_have_unique_ids() -> None:
    scenarios = create_example_tool_scenarios()

    assert len(scenarios) == 2
    assert len({scenario.case_id for scenario in scenarios}) == 2


def test_calculator_scenario_matches_existing_tool_contract() -> None:
    scenario = create_example_tool_scenarios()[0]

    assert scenario.available_tools == ("calculator",)
    assert scenario.policy.max_provider_turns == 2

    expected = scenario.evaluation.expected_calls[0]

    assert expected.tool_name == "calculator"
    assert expected.arguments == {
        "operation": "multiply",
        "left": 17,
        "right": 23,
    }
    assert scenario.evaluation.expected_final_text == "391"


def test_direct_answer_can_expect_no_tool_calls() -> None:
    scenario = create_example_tool_scenarios()[1]

    assert scenario.available_tools == ("calculator",)
    assert scenario.evaluation.expected_calls == ()
    assert scenario.evaluation.expected_final_text == "READY"


def test_scenarios_are_frozen() -> None:
    scenario = create_example_tool_scenarios()[0]

    with pytest.raises(FrozenInstanceError):
        scenario.user_text = "Changed."


def test_factories_return_independent_snapshots() -> None:
    first = create_example_tool_scenarios()
    second = create_example_tool_scenarios()

    assert first == second
    assert first is not second
    assert first[0] is not second[0]

    arguments = first[0].evaluation.expected_calls[0].arguments
    arguments["left"] = 999

    assert second[0].evaluation.expected_calls[0].arguments["left"] == 17


@pytest.mark.parametrize("text", ["", " ", None, "\ud800"])
def test_invalid_user_text_is_rejected(text: object) -> None:
    scenario = create_example_tool_scenarios()[0]

    with pytest.raises(ValueError):
        replace(scenario, user_text=text)


@pytest.mark.parametrize(
    "names",
    [
        (),
        ("calculator", "calculator"),
        ("invalid name",),
    ],
)
def test_invalid_tool_selection_is_rejected(
    names: tuple[str, ...],
) -> None:
    scenario = create_example_tool_scenarios()[0]

    with pytest.raises(ValueError):
        replace(scenario, available_tools=names)


def test_tool_selection_must_be_a_tuple() -> None:
    scenario = create_example_tool_scenarios()[0]

    with pytest.raises(TypeError):
        replace(scenario, available_tools=["calculator"])


def test_expected_tool_must_be_available() -> None:
    scenario = create_example_tool_scenarios()[0]

    other_evaluation = ToolEvaluationCase(
        case_id="other",
        expected_calls=(
            ExpectedToolCall(
                tool_name="lookup_city_code",
                arguments={"city": "northport"},
            ),
        ),
        expected_final_text="SYN-001",
    )

    with pytest.raises(ValueError, match="available"):
        replace(scenario, evaluation=other_evaluation)


def test_example_prompt_preserves_sentence_boundary() -> None:
    assert create_example_tool_scenarios()[0].user_text == (
        "Use the calculator to multiply 17 by 23. Return only the resulting integer."
    )


@pytest.mark.parametrize("field", ["policy", "evaluation"])
@pytest.mark.parametrize("value", [None, {}, object()])
def test_configuration_requires_exact_contract_types(field, value) -> None:
    scenario = create_example_tool_scenarios()[0]
    with pytest.raises(TypeError):
        replace(scenario, **{field: value})


@pytest.mark.parametrize("name", [None, 1, True, [], "", "a" * 65])
def test_selected_names_reuse_tool_name_validation(name) -> None:
    with pytest.raises(ValueError):
        replace(create_example_tool_scenarios()[0], available_tools=(name,))


def test_case_id_is_derived_and_text_is_not_in_repr() -> None:
    scenario = create_example_tool_scenarios()[0]
    changed = replace(
        scenario, evaluation=replace(scenario.evaluation, case_id="new-case")
    )
    assert changed.case_id == "new-case"
    assert scenario.case_id == "calculator-multiply"
    assert scenario.user_text not in repr(scenario)
    assert "391" not in repr(scenario)


def test_unicode_user_text_is_preserved() -> None:
    text = "  Çarpımı hesapla.  "
    scenario = replace(create_example_tool_scenarios()[0], user_text=text)
    assert scenario.user_text == text


def test_examples_fit_budgets_and_registered_argument_schemas() -> None:
    from llm_benchmark.tools import create_example_tool_registry

    registry = create_example_tool_registry()
    for scenario in create_example_tool_scenarios():
        assert len(scenario.evaluation.expected_calls) <= scenario.policy.max_tool_calls
        if scenario.evaluation.expected_calls:
            assert scenario.policy.max_provider_turns >= 2
        for name in scenario.available_tools:
            assert registry.get(name) is not None
        for expected in scenario.evaluation.expected_calls:
            registration = registry.get(expected.tool_name)
            registration.argument_model.model_validate(expected.arguments)


def test_factory_does_not_initialize_runtime_or_execute_handlers(monkeypatch) -> None:
    from llm_benchmark import tools
    from llm_benchmark.tool_runtime import ToolRegistry, ToolRuntime

    def forbidden(*args, **kwargs):
        pytest.fail("Scenario construction must not execute tools")

    monkeypatch.setattr(ToolRegistry, "__init__", forbidden)
    monkeypatch.setattr(ToolRuntime, "__init__", forbidden)
    monkeypatch.setattr(ToolRuntime, "execute", forbidden)
    monkeypatch.setattr(tools, "calculator_handler", forbidden)
    monkeypatch.setattr(tools, "lookup_city_code_handler", forbidden)
    assert len(create_example_tool_scenarios()) == 2

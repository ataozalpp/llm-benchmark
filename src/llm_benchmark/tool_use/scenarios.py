"""Immutable definitions for synthetic tool-evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from llm_benchmark.tool_use.evaluation import (
    ExpectedToolCall,
    ExpectedTrajectory,
    ToolEvaluationCase,
    ToolNecessity,
)
from llm_benchmark.tool_use.loop import ToolLoopPolicy
from llm_benchmark.tool_use.runtime import ToolDefinition

type ExampleToolSuite = Literal["example-tools-v1", "example-tools-v2"]

DIRECT_ANSWER_TEXT = "Return exactly READY. Do not use any tools."
CALCULATOR_MULTIPLY_TEXT = (
    "Use the calculator to multiply 17 by 23. Return only the resulting integer."
)
CALCULATOR_ADD_TEXT = (
    "Use the calculator to add 17 and 23. Return only the resulting integer."
)
NO_TOOL_ARITHMETIC_TEXT = (
    "Six plus seven is thirteen. Without using any tools, "
    "return only that result as an integer."
)
TWO_STEP_TEXT = (
    "Use the calculator to multiply 6 by 7, then use it again to add 8 to "
    "that product. Return only the final integer."
)


@dataclass(frozen=True)
class ToolScenario:
    user_text: str = field(repr=False)
    available_tools: tuple[str, ...]
    policy: ToolLoopPolicy
    evaluation: ToolEvaluationCase = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.user_text) is not str or not self.user_text.strip():
            raise ValueError("Scenario text must be non-blank.")

        try:
            self.user_text.encode("utf-8")
        except UnicodeError:
            pass
        else:
            self._validate_configuration()
            return

        raise ValueError("Scenario text must be valid UTF-8.")

    def _validate_configuration(self) -> None:
        if type(self.available_tools) is not tuple:
            raise TypeError("Available tools must be a tuple.")

        if not self.available_tools:
            raise ValueError("At least one available tool is required.")

        for name in self.available_tools:
            ToolDefinition(
                name=name,
                description="Scenario tool selection.",
            )

        if len(set(self.available_tools)) != len(self.available_tools):
            raise ValueError("Available tool names must be unique.")

        if type(self.policy) is not ToolLoopPolicy:
            raise TypeError("Expected a ToolLoopPolicy.")

        if type(self.evaluation) is not ToolEvaluationCase:
            raise TypeError("Expected a ToolEvaluationCase.")

        allowed_names = set(self.available_tools)

        for trajectory in self.evaluation.accepted_trajectories:
            if any(call.tool_name not in allowed_names for call in trajectory.calls):
                raise ValueError("Expected calls must use available tools.")

            if len(trajectory.calls) > self.policy.max_tool_calls:
                raise ValueError("Accepted trajectories must fit the tool budget.")

    @property
    def case_id(self) -> str:
        return self.evaluation.case_id


def _calculator(operation: str, left: int, right: int) -> ExpectedToolCall:
    return ExpectedToolCall(
        tool_name="calculator",
        arguments={"operation": operation, "left": left, "right": right},
    )


def _example_tools_v1() -> tuple[ToolScenario, ...]:
    return (
        ToolScenario(
            user_text=CALCULATOR_MULTIPLY_TEXT,
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=2,
                max_tool_calls=1,
            ),
            evaluation=ToolEvaluationCase(
                case_id="calculator-multiply",
                tool_necessity=ToolNecessity.REQUIRED,
                accepted_trajectories=(
                    ExpectedTrajectory((_calculator("multiply", 17, 23),)),
                ),
                expected_final_text="391",
            ),
        ),
        ToolScenario(
            user_text=DIRECT_ANSWER_TEXT,
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=1,
                max_tool_calls=1,
            ),
            evaluation=ToolEvaluationCase(
                case_id="direct-answer",
                tool_necessity=ToolNecessity.NOT_NEEDED,
                accepted_trajectories=(ExpectedTrajectory(()),),
                expected_final_text="READY",
            ),
        ),
    )


def _example_tools_v2_additions() -> tuple[ToolScenario, ...]:
    # Calculator-only so the MCP calculator profile can run the same suite.
    return (
        ToolScenario(
            user_text=CALCULATOR_ADD_TEXT,
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=2,
                max_tool_calls=1,
            ),
            evaluation=ToolEvaluationCase(
                case_id="calculator-add-commutative",
                tool_necessity=ToolNecessity.REQUIRED,
                accepted_trajectories=(
                    ExpectedTrajectory((_calculator("add", 17, 23),)),
                    ExpectedTrajectory((_calculator("add", 23, 17),)),
                ),
                expected_final_text="40",
            ),
        ),
        ToolScenario(
            user_text=NO_TOOL_ARITHMETIC_TEXT,
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=1,
                max_tool_calls=1,
            ),
            evaluation=ToolEvaluationCase(
                case_id="no-tool-arithmetic",
                tool_necessity=ToolNecessity.NOT_NEEDED,
                accepted_trajectories=(ExpectedTrajectory(()),),
                expected_final_text="13",
            ),
        ),
        ToolScenario(
            user_text=TWO_STEP_TEXT,
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=3,
                max_tool_calls=2,
            ),
            evaluation=ToolEvaluationCase(
                case_id="two-step",
                tool_necessity=ToolNecessity.REQUIRED,
                accepted_trajectories=tuple(
                    ExpectedTrajectory((multiply, add))
                    for multiply in (
                        _calculator("multiply", 6, 7),
                        _calculator("multiply", 7, 6),
                    )
                    for add in (
                        _calculator("add", 42, 8),
                        _calculator("add", 8, 42),
                    )
                ),
                expected_final_text="50",
            ),
        ),
    )


def create_example_tool_scenarios(
    suite: ExampleToolSuite = "example-tools-v1",
) -> tuple[ToolScenario, ...]:
    """Return fresh synthetic scenario definitions without execution."""

    if suite == "example-tools-v1":
        return _example_tools_v1()

    if suite == "example-tools-v2":
        return (*_example_tools_v1(), *_example_tools_v2_additions())

    raise ValueError("Unknown example tool suite.")

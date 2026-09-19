"""Immutable definitions for synthetic tool-evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field

from .tool_evaluation import ExpectedToolCall, ToolEvaluationCase
from .tool_loop import ToolLoopPolicy
from .tool_runtime import ToolDefinition


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

        if any(
            call.tool_name not in allowed_names
            for call in self.evaluation.expected_calls
        ):
            raise ValueError("Expected calls must use available tools.")

    @property
    def case_id(self) -> str:
        return self.evaluation.case_id


def create_example_tool_scenarios() -> tuple[ToolScenario, ...]:
    """Return fresh synthetic scenario definitions without execution."""

    return (
        ToolScenario(
            user_text=(
                "Use the calculator to multiply 17 by 23. "
                "Return only the resulting integer."
            ),
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=2,
                max_tool_calls=1,
            ),
            evaluation=ToolEvaluationCase(
                case_id="calculator-multiply",
                expected_calls=(
                    ExpectedToolCall(
                        tool_name="calculator",
                        arguments={
                            "operation": "multiply",
                            "left": 17,
                            "right": 23,
                        },
                    ),
                ),
                expected_final_text="391",
            ),
        ),
        ToolScenario(
            user_text=("Return exactly READY. Do not use any tools."),
            available_tools=("calculator",),
            policy=ToolLoopPolicy(
                max_provider_turns=1,
                max_tool_calls=1,
            ),
            evaluation=ToolEvaluationCase(
                case_id="direct-answer",
                expected_calls=(),
                expected_final_text="READY",
            ),
        ),
    )

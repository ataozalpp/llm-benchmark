"""Pure deterministic evaluation of bounded tool-loop outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field

from .tool_loop import ToolLoopResult, ToolLoopStopReason
from .tool_provider_models import ToolProviderStatus
from .tool_runtime import (
    JsonObject,
    ToolCall,
    ToolExecutionStatus,
)


@dataclass(frozen=True)
class ExpectedToolCall:
    """Expected tool name and immutable strict-JSON arguments."""

    _call: ToolCall = field(repr=False)

    def __init__(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> None:
        validated = ToolCall(
            call_id="expectation",
            tool_name=tool_name,
            arguments=arguments,
        )
        object.__setattr__(self, "_call", validated)

    @property
    def tool_name(self) -> str:
        return self._call.tool_name

    @property
    def arguments(self) -> JsonObject:
        return self._call.arguments


@dataclass(frozen=True)
class ToolEvaluationCase:
    case_id: str
    expected_calls: tuple[ExpectedToolCall, ...] = field(repr=False)
    expected_final_text: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id.strip():
            raise ValueError("Case ID must be a non-blank string.")

        if type(self.expected_calls) is not tuple:
            raise TypeError("Expected calls must be a tuple.")

        if any(type(call) is not ExpectedToolCall for call in self.expected_calls):
            raise TypeError("Invalid expected tool call.")

        if (
            type(self.expected_final_text) is not str
            or not self.expected_final_text.strip()
        ):
            raise ValueError("Expected final text must be non-blank.")

        try:
            self.case_id.encode("utf-8")
            self.expected_final_text.encode("utf-8")
        except UnicodeError:
            pass
        else:
            return

        raise ValueError("Evaluation text must be valid UTF-8.")


@dataclass(frozen=True)
class ToolEvaluationResult:
    case_id: str
    stop_reason: ToolLoopStopReason
    completed: bool

    tool_sequence_match: bool
    arguments_match: bool | None
    final_answer_match: bool | None

    requested_tool_call_count: int
    executed_tool_call_count: int
    successful_tool_call_count: int

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id.strip():
            raise ValueError("Case ID must be a non-blank string.")

        if type(self.stop_reason) is not ToolLoopStopReason:
            raise TypeError("Invalid stop reason.")

        for value in (self.completed, self.tool_sequence_match):
            if type(value) is not bool:
                raise TypeError("Evaluation flags must be booleans.")

        for value in (
            self.arguments_match,
            self.final_answer_match,
        ):
            if value is not None and type(value) is not bool:
                raise TypeError("Match values must be booleans or null.")

        for value in (
            self.requested_tool_call_count,
            self.executed_tool_call_count,
            self.successful_tool_call_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Counts must be non-negative integers.")

        if not (
            self.successful_tool_call_count
            <= self.executed_tool_call_count
            <= self.requested_tool_call_count
        ):
            raise ValueError("Tool counts are inconsistent.")

        if self.completed != (self.stop_reason is ToolLoopStopReason.FINAL_RESPONSE):
            raise ValueError("Completion must match the stop reason.")

        if self.tool_sequence_match:
            if self.arguments_match is None:
                raise ValueError("Matching sequences require argument evaluation.")
        elif self.arguments_match is not None:
            raise ValueError("Different sequences cannot have argument evaluation.")

        if self.completed:
            if self.final_answer_match is None:
                raise ValueError("Completed results require final evaluation.")
        elif self.final_answer_match is not None:
            raise ValueError("Incomplete results cannot have final evaluation.")


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False

    if type(left) is dict:
        if left.keys() != right.keys():
            return False

        return all(_strict_json_equal(left[key], right[key]) for key in left)

    if type(left) is list:
        if len(left) != len(right):
            return False

        return all(
            _strict_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )

    return left == right


def evaluate_tool_loop(
    *,
    case: ToolEvaluationCase,
    result: ToolLoopResult,
) -> ToolEvaluationResult:
    """Evaluate one invocation without executing providers or tools."""

    if type(case) is not ToolEvaluationCase:
        raise TypeError("Expected a ToolEvaluationCase.")

    if type(result) is not ToolLoopResult:
        raise TypeError("Expected a ToolLoopResult.")

    requested_calls: list[ToolCall] = []

    for provider_result in result.provider_results:
        if provider_result.status is not ToolProviderStatus.SUCCEEDED:
            continue

        turn = provider_result.turn
        if turn is None:
            raise AssertionError("Successful provider result requires a turn.")

        requested_calls.extend(turn.tool_calls)

    expected_names = tuple(call.tool_name for call in case.expected_calls)
    requested_names = tuple(call.tool_name for call in requested_calls)

    tool_sequence_match = requested_names == expected_names

    arguments_match: bool | None = None
    if tool_sequence_match:
        arguments_match = all(
            _strict_json_equal(expected.arguments, actual.arguments)
            for expected, actual in zip(
                case.expected_calls,
                requested_calls,
                strict=True,
            )
        )

    completed = result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE

    final_answer_match: bool | None = None
    if completed:
        final_answer_match = result.final_text == case.expected_final_text

    successful_tool_call_count = sum(
        tool_result.status is ToolExecutionStatus.SUCCEEDED
        for tool_result in result.tool_results
    )

    return ToolEvaluationResult(
        case_id=case.case_id,
        stop_reason=result.stop_reason,
        completed=completed,
        tool_sequence_match=tool_sequence_match,
        arguments_match=arguments_match,
        final_answer_match=final_answer_match,
        requested_tool_call_count=len(requested_calls),
        executed_tool_call_count=len(result.tool_results),
        successful_tool_call_count=successful_tool_call_count,
    )

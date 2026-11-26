"""Pure deterministic evaluation of bounded tool-loop outcomes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from llm_benchmark.tool_use.loop import ToolLoopResult, ToolLoopStopReason
from llm_benchmark.tool_use.provider_models import ToolProviderStatus
from llm_benchmark.tool_use.runtime import (
    JsonObject,
    ToolCall,
    ToolExecutionStatus,
)


class ToolNecessity(StrEnum):
    """Whether a correct answer requires at least one tool call."""

    REQUIRED = "required"
    NOT_NEEDED = "not_needed"


class TrajectoryErrorCategory(StrEnum):
    """Single trajectory label per case, assigned in declaration order."""

    UNNECESSARY_TOOL_CALL = "unnecessary_tool_call"
    INVALID_ARGUMENTS = "invalid_arguments"
    MISSING_TOOL_CALL = "missing_tool_call"
    INCOMPLETE_TRAJECTORY = "incomplete_trajectory"
    EXTRA_TOOL_CALL = "extra_tool_call"
    WRONG_ORDER = "wrong_order"
    WRONG_TOOL = "wrong_tool"


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
class ExpectedTrajectory:
    """One accepted, ordered sequence of tool calls (possibly empty)."""

    calls: tuple[ExpectedToolCall, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.calls) is not tuple:
            raise TypeError("Trajectory calls must be a tuple.")

        if any(type(call) is not ExpectedToolCall for call in self.calls):
            raise TypeError("Invalid expected tool call.")

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(call.tool_name for call in self.calls)


def _trajectories_equal(left: ExpectedTrajectory, right: ExpectedTrajectory) -> bool:
    return left.tool_names == right.tool_names and all(
        _strict_json_equal(a.arguments, b.arguments)
        for a, b in zip(left.calls, right.calls, strict=True)
    )


@dataclass(frozen=True)
class ToolEvaluationCase:
    case_id: str
    tool_necessity: ToolNecessity
    accepted_trajectories: tuple[ExpectedTrajectory, ...] = field(repr=False)
    expected_final_text: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id.strip():
            raise ValueError("Case ID must be a non-blank string.")

        if type(self.tool_necessity) is not ToolNecessity:
            raise TypeError("Invalid tool necessity.")

        if type(self.accepted_trajectories) is not tuple:
            raise TypeError("Accepted trajectories must be a tuple.")

        if any(
            type(trajectory) is not ExpectedTrajectory
            for trajectory in self.accepted_trajectories
        ):
            raise TypeError("Invalid accepted trajectory.")

        if not self.accepted_trajectories:
            raise ValueError("At least one accepted trajectory is required.")

        for index, trajectory in enumerate(self.accepted_trajectories):
            if any(
                _trajectories_equal(trajectory, other)
                for other in self.accepted_trajectories[:index]
            ):
                raise ValueError("Accepted trajectories must be unique.")

        if self.tool_necessity is ToolNecessity.NOT_NEEDED:
            if len(self.accepted_trajectories) != 1 or (
                self.accepted_trajectories[0].calls
            ):
                raise ValueError(
                    "Cases without needed tools accept only the empty trajectory."
                )
        elif any(not trajectory.calls for trajectory in self.accepted_trajectories):
            raise ValueError("Cases requiring tools cannot accept no calls.")

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

    tool_necessity: ToolNecessity
    accepted_trajectory_count: int
    matched_trajectory_index: int | None
    trajectory_error: TrajectoryErrorCategory | None

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

        if type(self.tool_necessity) is not ToolNecessity:
            raise TypeError("Invalid tool necessity.")

        if self.trajectory_error is not None and (
            type(self.trajectory_error) is not TrajectoryErrorCategory
        ):
            raise TypeError("Invalid trajectory error category.")

        for value in (
            self.requested_tool_call_count,
            self.executed_tool_call_count,
            self.successful_tool_call_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Counts must be non-negative integers.")

        if (
            type(self.accepted_trajectory_count) is not int
            or self.accepted_trajectory_count < 1
        ):
            raise ValueError("Accepted trajectory count must be positive.")

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
            index = self.matched_trajectory_index
            if (
                type(index) is not int
                or not 0 <= index < self.accepted_trajectory_count
            ):
                raise ValueError("Matching sequences require a trajectory index.")
        else:
            if self.arguments_match is not None:
                raise ValueError("Different sequences cannot have argument evaluation.")
            if self.matched_trajectory_index is not None:
                raise ValueError("Different sequences cannot have a trajectory index.")

        if self.completed:
            if self.final_answer_match is None:
                raise ValueError("Completed results require final evaluation.")
        elif self.final_answer_match is not None:
            raise ValueError("Incomplete results cannot have final evaluation.")

        self._validate_trajectory_error()

    def _validate_trajectory_error(self) -> None:
        error = self.trajectory_error
        called = self.requested_tool_call_count > 0

        if self.tool_necessity is ToolNecessity.NOT_NEEDED:
            if self.accepted_trajectory_count != 1:
                raise ValueError("No-tool cases accept exactly one trajectory.")
            if self.tool_sequence_match == called:
                raise ValueError("No-tool sequence match must equal abstention.")
            if called != (error is TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL):
                raise ValueError("Unnecessary calls must be labeled exactly.")
        else:
            if not called and self.tool_sequence_match:
                raise ValueError("Tool-requiring cases cannot match without calls.")
            if error is TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL:
                raise ValueError("Tool-requiring cases cannot be unnecessary.")

        correct = self.tool_sequence_match and self.arguments_match is True
        if correct != (error is None):
            raise ValueError("Only fully matching trajectories have no error.")

        if (error is TrajectoryErrorCategory.INVALID_ARGUMENTS) != (
            self.tool_sequence_match and self.arguments_match is False
        ):
            raise ValueError("Argument errors require a matching tool sequence.")

        if error is TrajectoryErrorCategory.MISSING_TOOL_CALL and not self.completed:
            raise ValueError("Missing calls require a completed loop.")

        if error is TrajectoryErrorCategory.INCOMPLETE_TRAJECTORY and self.completed:
            raise ValueError("Incomplete trajectories require a stopped loop.")


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


def _is_strict_prefix(prefix: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    return len(prefix) < len(sequence) and sequence[: len(prefix)] == prefix


def _classify_mismatch(
    *,
    case: ToolEvaluationCase,
    requested_names: tuple[str, ...],
    completed: bool,
) -> TrajectoryErrorCategory:
    """Label a trajectory that matches no accepted name sequence."""

    if case.tool_necessity is ToolNecessity.NOT_NEEDED:
        return TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL

    accepted = tuple(t.tool_names for t in case.accepted_trajectories)

    if any(_is_strict_prefix(requested_names, names) for names in accepted):
        if completed:
            return TrajectoryErrorCategory.MISSING_TOOL_CALL
        return TrajectoryErrorCategory.INCOMPLETE_TRAJECTORY

    if any(_is_strict_prefix(names, requested_names) for names in accepted):
        return TrajectoryErrorCategory.EXTRA_TOOL_CALL

    requested_counter = Counter(requested_names)
    if any(Counter(names) == requested_counter for names in accepted):
        return TrajectoryErrorCategory.WRONG_ORDER

    return TrajectoryErrorCategory.WRONG_TOOL


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

    requested_names = tuple(call.tool_name for call in requested_calls)

    name_matches = tuple(
        index
        for index, trajectory in enumerate(case.accepted_trajectories)
        if trajectory.tool_names == requested_names
    )

    matched_trajectory_index: int | None = None
    arguments_match: bool | None = None

    for index in name_matches:
        expected_calls = case.accepted_trajectories[index].calls
        if all(
            _strict_json_equal(expected.arguments, actual.arguments)
            for expected, actual in zip(expected_calls, requested_calls, strict=True)
        ):
            matched_trajectory_index = index
            arguments_match = True
            break

    if name_matches and matched_trajectory_index is None:
        matched_trajectory_index = name_matches[0]
        arguments_match = False

    tool_sequence_match = bool(name_matches)
    completed = result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE

    trajectory_error: TrajectoryErrorCategory | None
    if not tool_sequence_match:
        trajectory_error = _classify_mismatch(
            case=case,
            requested_names=requested_names,
            completed=completed,
        )
    elif arguments_match is False:
        trajectory_error = TrajectoryErrorCategory.INVALID_ARGUMENTS
    else:
        trajectory_error = None

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
        tool_necessity=case.tool_necessity,
        accepted_trajectory_count=len(case.accepted_trajectories),
        matched_trajectory_index=matched_trajectory_index,
        trajectory_error=trajectory_error,
    )

"""Pure aggregation of deterministic tool-evaluation results."""

from __future__ import annotations

from dataclasses import dataclass

from .tool_evaluation import ToolEvaluationResult
from .tool_loop import ToolLoopStopReason


@dataclass(frozen=True)
class MetricCount:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        for value in (self.numerator, self.denominator):
            if type(value) is not int or value < 0:
                raise ValueError("Metric counts must be non negative integers.")

        if self.numerator > self.denominator:
            raise ValueError("Numerator cannot exceed denominator.")

    @property
    def rate(self) -> float | None:
        if self.denominator == 0:
            return None

        return self.numerator / self.denominator


@dataclass(frozen=True)
class StopReasonCount:
    reason: ToolLoopStopReason
    count: int

    def __post_init__(self) -> None:
        if type(self.reason) is not ToolLoopStopReason:
            raise TypeError("Invalid stop reason.")

        if type(self.count) is not int or self.count < 0:
            raise ValueError("Stop reason count must be a non negative integer.")


@dataclass(frozen=True)
class ToolEvaluationSummary:
    case_count: int

    completion: MetricCount
    tool_sequence_match: MetricCount
    argument_match: MetricCount
    final_answer_match: MetricCount

    argument_unavailable_count: int
    final_unavailable_count: int

    requested_call_count: int
    executed_call_count: int
    successful_call_count: int

    stop_reason_counts: tuple[StopReasonCount, ...]

    def __post_init__(self) -> None:
        if type(self.case_count) is not int or self.case_count <= 0:
            raise ValueError("Case count must be a positive integer.")

        for metric in (
            self.completion,
            self.tool_sequence_match,
            self.argument_match,
            self.final_answer_match,
        ):
            if type(metric) is not MetricCount:
                raise TypeError("Expected a MetricCount.")

        for value in (
            self.argument_unavailable_count,
            self.final_unavailable_count,
            self.requested_call_count,
            self.executed_call_count,
            self.successful_call_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Summary counts must be non negative integers.")

        if self.completion.denominator != self.case_count:
            raise ValueError("Invalid completion denominator.")

        if self.tool_sequence_match.denominator != self.case_count:
            raise ValueError("Invalid tool sequence denominator.")

        if self.argument_match.denominator != self.tool_sequence_match.numerator:
            raise ValueError("Invalid argument denominator.")

        if self.final_answer_match.denominator != self.completion.numerator:
            raise ValueError("Invalid final answer denominator.")

        if (
            self.argument_match.denominator + self.argument_unavailable_count
            != self.case_count
        ):
            raise ValueError("Argument coverage is inconsistent.")

        if (
            self.final_answer_match.denominator + self.final_unavailable_count
            != self.case_count
        ):
            raise ValueError("Final answer coverage is inconsistent.")

        if not (
            self.successful_call_count
            <= self.executed_call_count
            <= self.requested_call_count
        ):
            raise ValueError("Tool call counts are inconsistent.")

        if type(self.stop_reason_counts) is not tuple:
            raise TypeError("Stop reason counts must be a tuple.")

        if any(type(item) is not StopReasonCount for item in self.stop_reason_counts):
            raise TypeError("Invalid stop reason entry.")

        expected_reasons = tuple(
            sorted(ToolLoopStopReason, key=lambda reason: reason.value)
        )
        actual_reasons = tuple(item.reason for item in self.stop_reason_counts)

        if actual_reasons != expected_reasons:
            raise ValueError("Stop reasons must occur once in canonical order.")

        if sum(item.count for item in self.stop_reason_counts) != self.case_count:
            raise ValueError("Stop reason total is inconsistent.")

        final_count = next(
            item.count
            for item in self.stop_reason_counts
            if item.reason is ToolLoopStopReason.FINAL_RESPONSE
        )

        if final_count != self.completion.numerator:
            raise ValueError("Final response count must match completion count.")


def _count_matches(
    values: tuple[bool | None, ...],
) -> MetricCount:
    applicable = tuple(value for value in values if value is not None)

    return MetricCount(
        numerator=sum(value is True for value in applicable),
        denominator=len(applicable),
    )


def aggregate_tool_evaluations(
    results: tuple[ToolEvaluationResult, ...],
) -> ToolEvaluationSummary:
    """Aggregate one coherent batch of evaluation results without I/O."""

    if type(results) is not tuple:
        raise TypeError("Evaluation results must be a tuple.")

    if not results:
        raise ValueError("At least one evaluation result is required.")

    if any(type(result) is not ToolEvaluationResult for result in results):
        raise TypeError("Invalid evaluation result.")

    case_ids = tuple(result.case_id for result in results)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique within a report.")

    case_count = len(results)

    completion = MetricCount(
        numerator=sum(result.completed for result in results),
        denominator=case_count,
    )

    tool_sequence_match = MetricCount(
        numerator=sum(result.tool_sequence_match for result in results),
        denominator=case_count,
    )

    argument_match = _count_matches(tuple(result.arguments_match for result in results))

    final_answer_match = _count_matches(
        tuple(result.final_answer_match for result in results)
    )

    stop_reason_counts = tuple(
        StopReasonCount(
            reason=reason,
            count=sum(result.stop_reason is reason for result in results),
        )
        for reason in sorted(
            ToolLoopStopReason,
            key=lambda item: item.value,
        )
    )

    return ToolEvaluationSummary(
        case_count=case_count,
        completion=completion,
        tool_sequence_match=tool_sequence_match,
        argument_match=argument_match,
        final_answer_match=final_answer_match,
        argument_unavailable_count=(case_count - argument_match.denominator),
        final_unavailable_count=(case_count - final_answer_match.denominator),
        requested_call_count=sum(
            result.requested_tool_call_count for result in results
        ),
        executed_call_count=sum(result.executed_tool_call_count for result in results),
        successful_call_count=sum(
            result.successful_tool_call_count for result in results
        ),
        stop_reason_counts=stop_reason_counts,
    )

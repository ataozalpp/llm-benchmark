from dataclasses import FrozenInstanceError, replace

import pytest

from llm_benchmark.tool_use.evaluation import (
    ToolEvaluationResult,
    ToolNecessity,
    TrajectoryErrorCategory,
)
from llm_benchmark.tool_use.loop import ToolLoopStopReason
from llm_benchmark.tool_use.reporting import (
    MetricCount,
    StopReasonCount,
    TrajectoryErrorCount,
    aggregate_tool_evaluations,
)

_UNSET = object()


def _derived_error(
    *, necessity, completed, sequence_match, arguments_match, requested
) -> TrajectoryErrorCategory | None:
    if necessity is ToolNecessity.NOT_NEEDED and requested:
        return TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL
    if sequence_match:
        return None if arguments_match else TrajectoryErrorCategory.INVALID_ARGUMENTS
    if requested == 0:
        return (
            TrajectoryErrorCategory.MISSING_TOOL_CALL
            if completed
            else TrajectoryErrorCategory.INCOMPLETE_TRAJECTORY
        )
    return TrajectoryErrorCategory.WRONG_TOOL


def evaluation(
    case_id: str,
    *,
    completed: bool = True,
    sequence_match: bool = True,
    arguments_match: bool | None = True,
    final_match: bool | None = True,
    requested: int = 1,
    executed: int = 1,
    successful: int = 1,
    necessity: ToolNecessity = ToolNecessity.REQUIRED,
    trajectory_error: object = _UNSET,
) -> ToolEvaluationResult:
    if trajectory_error is _UNSET:
        trajectory_error = _derived_error(
            necessity=necessity,
            completed=completed,
            sequence_match=sequence_match,
            arguments_match=arguments_match,
            requested=requested,
        )
    return ToolEvaluationResult(
        case_id=case_id,
        stop_reason=(
            ToolLoopStopReason.FINAL_RESPONSE
            if completed
            else ToolLoopStopReason.PROVIDER_FAILED
        ),
        completed=completed,
        tool_sequence_match=sequence_match,
        arguments_match=arguments_match,
        final_answer_match=final_match,
        requested_tool_call_count=requested,
        executed_tool_call_count=executed,
        successful_tool_call_count=successful,
        tool_necessity=necessity,
        accepted_trajectory_count=1,
        matched_trajectory_index=0 if sequence_match else None,
        trajectory_error=trajectory_error,
    )


def no_tool(case_id: str, *, called: bool = False) -> ToolEvaluationResult:
    count = int(called)
    return evaluation(
        case_id,
        necessity=ToolNecessity.NOT_NEEDED,
        sequence_match=not called,
        arguments_match=None if called else True,
        requested=count,
        executed=count,
        successful=count,
    )


def test_metric_rate() -> None:
    metric = MetricCount(numerator=2, denominator=3)

    assert metric.rate == pytest.approx(2 / 3)


def test_zero_denominator_is_unavailable() -> None:
    metric = MetricCount(numerator=0, denominator=0)

    assert metric.rate is None


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (-1, 2),
        (3, 2),
        (True, 2),
        (1, False),
        (1.0, 2),
        (1, "2"),
    ],
)
def test_invalid_metric_counts(
    numerator: object,
    denominator: object,
) -> None:
    with pytest.raises(ValueError):
        MetricCount(
            numerator=numerator,
            denominator=denominator,
        )


def test_all_successful_results() -> None:
    summary = aggregate_tool_evaluations(
        (
            evaluation("case-a"),
            evaluation("case-b"),
        )
    )

    assert summary.case_count == 2
    assert summary.completion == MetricCount(2, 2)
    assert summary.tool_sequence_match == MetricCount(2, 2)
    assert summary.argument_match == MetricCount(2, 2)
    assert summary.final_answer_match == MetricCount(2, 2)

    assert summary.argument_unavailable_count == 0
    assert summary.final_unavailable_count == 0

    assert summary.requested_call_count == 2
    assert summary.executed_call_count == 2
    assert summary.successful_call_count == 2

    distribution = {item.reason: item.count for item in summary.stop_reason_counts}

    assert distribution[ToolLoopStopReason.FINAL_RESPONSE] == 2
    assert distribution[ToolLoopStopReason.PROVIDER_FAILED] == 0
    assert sum(distribution.values()) == 2


def test_mixed_results_keep_denominators_and_coverage() -> None:
    results = (
        evaluation("case-a"),
        evaluation(
            "case-b",
            sequence_match=False,
            arguments_match=None,
            final_match=False,
        ),
        evaluation(
            "case-c",
            completed=False,
            arguments_match=False,
            final_match=None,
            successful=0,
        ),
    )

    summary = aggregate_tool_evaluations(results)

    assert summary.case_count == 3

    assert summary.completion == MetricCount(2, 3)
    assert summary.tool_sequence_match == MetricCount(2, 3)

    assert summary.argument_match == MetricCount(1, 2)
    assert summary.argument_unavailable_count == 1

    assert summary.final_answer_match == MetricCount(1, 2)
    assert summary.final_unavailable_count == 1

    assert summary.requested_call_count == 3
    assert summary.executed_call_count == 3
    assert summary.successful_call_count == 2


def test_unavailable_matches_are_not_zero_accuracy() -> None:
    result = evaluation(
        "case-a",
        completed=False,
        sequence_match=False,
        arguments_match=None,
        final_match=None,
        requested=0,
        executed=0,
        successful=0,
    )

    summary = aggregate_tool_evaluations((result,))

    assert summary.argument_match == MetricCount(0, 0)
    assert summary.argument_match.rate is None
    assert summary.argument_unavailable_count == 1

    assert summary.final_answer_match == MetricCount(0, 0)
    assert summary.final_answer_match.rate is None
    assert summary.final_unavailable_count == 1

    assert summary.completion.rate == 0.0


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        aggregate_tool_evaluations(
            (
                evaluation("same-case"),
                evaluation("same-case"),
            )
        )


@pytest.mark.parametrize(
    "results",
    [
        None,
        [],
        (object(),),
    ],
)
def test_invalid_input_types_are_rejected(results: object) -> None:
    with pytest.raises(TypeError):
        aggregate_tool_evaluations(results)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        aggregate_tool_evaluations(())


def test_input_order_does_not_change_summary() -> None:
    results = (
        evaluation("case-a"),
        evaluation("case-b", final_match=False),
    )

    original = aggregate_tool_evaluations(results)
    reversed_summary = aggregate_tool_evaluations(tuple(reversed(results)))

    assert original == reversed_summary
    assert results[0].case_id == "case-a"
    assert results[1].case_id == "case-b"


def test_summary_and_nested_values_are_frozen() -> None:
    summary = aggregate_tool_evaluations((evaluation("case-a"),))

    with pytest.raises(FrozenInstanceError):
        summary.case_count = 10

    with pytest.raises(FrozenInstanceError):
        summary.completion.numerator = 10

    with pytest.raises(FrozenInstanceError):
        summary.stop_reason_counts[0].count = 10


def test_inconsistent_summary_is_rejected() -> None:
    summary = aggregate_tool_evaluations((evaluation("case-a"),))

    with pytest.raises(ValueError):
        replace(summary, final_unavailable_count=1)

    with pytest.raises(ValueError):
        replace(summary, successful_call_count=2)

    with pytest.raises(ValueError):
        replace(
            summary,
            stop_reason_counts=tuple(reversed(summary.stop_reason_counts)),
        )


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_stop_reason_count_rejects_invalid_counts(value) -> None:
    with pytest.raises(ValueError):
        StopReasonCount(ToolLoopStopReason.FINAL_RESPONSE, value)


def test_stop_reason_requires_enum() -> None:
    with pytest.raises(TypeError):
        StopReasonCount("final_response", 1)


@pytest.mark.parametrize(
    "field",
    [
        "case_count",
        "argument_unavailable_count",
        "final_unavailable_count",
        "requested_call_count",
        "executed_call_count",
        "successful_call_count",
    ],
)
@pytest.mark.parametrize("value", [-1, True, "1"])
def test_summary_rejects_invalid_count_types(field, value) -> None:
    summary = aggregate_tool_evaluations((evaluation("case"),))
    with pytest.raises(ValueError):
        replace(summary, **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "completion",
        "tool_sequence_match",
        "argument_match",
        "final_answer_match",
        "trajectory_match",
        "tool_use_when_required",
        "tool_abstention_when_not_needed",
    ],
)
def test_summary_requires_metric_objects(field) -> None:
    summary = aggregate_tool_evaluations((evaluation("case"),))
    with pytest.raises(TypeError):
        replace(summary, **{field: (1, 1)})


@pytest.mark.parametrize(
    "updates",
    [
        {"case_count": 0},
        {"completion": MetricCount(1, 2)},
        {"tool_sequence_match": MetricCount(1, 2)},
        {"argument_match": MetricCount(0, 0)},
        {"final_answer_match": MetricCount(0, 0)},
        {"argument_unavailable_count": 1},
        {"requested_call_count": 0},
    ],
)
def test_summary_relationships(updates) -> None:
    summary = aggregate_tool_evaluations((evaluation("case"),))
    with pytest.raises(ValueError):
        replace(summary, **updates)


@pytest.mark.parametrize("counts", [[], (object(),)])
def test_stop_distribution_types(counts) -> None:
    summary = aggregate_tool_evaluations((evaluation("case"),))
    with pytest.raises(TypeError):
        replace(summary, stop_reason_counts=counts)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "total", "completion"])
def test_stop_distribution_consistency(mode) -> None:
    summary = aggregate_tool_evaluations((evaluation("case"),))
    counts = summary.stop_reason_counts
    if mode == "missing":
        counts = counts[:-1]
    elif mode == "duplicate":
        counts = (*counts, counts[0])
    elif mode == "total":
        counts = tuple(replace(item, count=0) for item in counts)
    else:
        counts = tuple(
            replace(item, count=int(item.reason is ToolLoopStopReason.PROVIDER_FAILED))
            for item in counts
        )
    with pytest.raises(ValueError):
        replace(summary, stop_reason_counts=counts)


def test_every_stop_reason_and_failed_runtime_counts() -> None:
    results = tuple(
        replace(
            evaluation(
                reason.value,
                completed=reason is ToolLoopStopReason.FINAL_RESPONSE,
                final_match=True
                if reason is ToolLoopStopReason.FINAL_RESPONSE
                else None,
                requested=3,
                executed=2,
                successful=1,
            ),
            stop_reason=reason,
        )
        for reason in ToolLoopStopReason
    )
    summary = aggregate_tool_evaluations(results)
    count = len(ToolLoopStopReason)
    assert summary.completion == MetricCount(1, count)
    assert summary.final_answer_match == MetricCount(1, 1)
    assert summary.final_unavailable_count == count - 1
    assert summary.requested_call_count == 3 * count
    assert summary.executed_call_count == 2 * count
    assert summary.successful_call_count == count
    assert all(item.count == 1 for item in summary.stop_reason_counts)
    assert tuple(item.reason.value for item in summary.stop_reason_counts) == tuple(
        sorted(reason.value for reason in ToolLoopStopReason)
    )


def test_real_evaluator_output_flows_into_reporting() -> None:
    from llm_benchmark.tool_use.calling import NormalizationToolTurn
    from llm_benchmark.tool_use.conversation import (
        AssistantMessage,
        ToolConversation,
        UserMessage,
    )
    from llm_benchmark.tool_use.evaluation import (
        ExpectedTrajectory,
        ToolEvaluationCase,
        evaluate_tool_loop,
    )
    from llm_benchmark.tool_use.loop import ToolLoopResult
    from llm_benchmark.tool_use.provider_models import (
        ToolProviderResult,
        ToolProviderStatus,
    )

    turn = NormalizationToolTurn(content="done", tool_calls=(), finish_reason="stop")
    result = ToolLoopResult(
        stop_reason=ToolLoopStopReason.FINAL_RESPONSE,
        conversation=ToolConversation(
            (UserMessage("Synthetic"), AssistantMessage(turn))
        ),
        provider_results=(
            ToolProviderResult(status=ToolProviderStatus.SUCCEEDED, turn=turn),
        ),
        tool_results=(),
        final_text="done",
    )
    evaluated = evaluate_tool_loop(
        case=ToolEvaluationCase(
            "case", ToolNecessity.NOT_NEEDED, (ExpectedTrajectory(()),), "done"
        ),
        result=result,
    )
    summary = aggregate_tool_evaluations((evaluated,))
    assert summary.tool_abstention_when_not_needed == MetricCount(1, 1)
    assert summary.trajectory_match == MetricCount(1, 1)
    assert summary.final_answer_match == MetricCount(1, 1)
    assert summary.argument_match == MetricCount(1, 1)
    assert (
        summary.requested_call_count
        == summary.executed_call_count
        == summary.successful_call_count
        == 0
    )


def test_necessity_metrics_use_their_own_denominators() -> None:
    results = (
        evaluation("required-used"),
        evaluation(
            "required-skipped",
            sequence_match=False,
            arguments_match=None,
            requested=0,
            executed=0,
            successful=0,
        ),
        no_tool("abstained"),
        no_tool("called", called=True),
        no_tool("abstained-2"),
    )

    summary = aggregate_tool_evaluations(results)

    assert summary.tool_use_when_required == MetricCount(1, 2)
    assert summary.tool_abstention_when_not_needed == MetricCount(2, 3)
    assert summary.trajectory_match == MetricCount(3, 5)
    errors = {item.category: item.count for item in summary.trajectory_error_counts}
    assert errors[TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL] == 1
    assert errors[TrajectoryErrorCategory.MISSING_TOOL_CALL] == 1
    assert sum(errors.values()) == 2


def test_absent_necessity_class_has_null_rate() -> None:
    required_only = aggregate_tool_evaluations((evaluation("case"),))
    no_tool_only = aggregate_tool_evaluations((no_tool("case"),))

    assert required_only.tool_abstention_when_not_needed == MetricCount(0, 0)
    assert required_only.tool_abstention_when_not_needed.rate is None
    assert no_tool_only.tool_use_when_required == MetricCount(0, 0)
    assert no_tool_only.tool_use_when_required.rate is None


def test_every_trajectory_category_is_counted_in_canonical_order() -> None:
    labels = {
        TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL: no_tool("u", called=True),
        TrajectoryErrorCategory.INVALID_ARGUMENTS: evaluation(
            "i", arguments_match=False
        ),
    }
    for category in (
        TrajectoryErrorCategory.MISSING_TOOL_CALL,
        TrajectoryErrorCategory.INCOMPLETE_TRAJECTORY,
        TrajectoryErrorCategory.EXTRA_TOOL_CALL,
        TrajectoryErrorCategory.WRONG_ORDER,
        TrajectoryErrorCategory.WRONG_TOOL,
    ):
        completed = category is not TrajectoryErrorCategory.INCOMPLETE_TRAJECTORY
        labels[category] = evaluation(
            category.value,
            completed=completed,
            sequence_match=False,
            arguments_match=None,
            final_match=True if completed else None,
            trajectory_error=category,
        )

    summary = aggregate_tool_evaluations((*labels.values(), evaluation("ok")))

    assert tuple(item.category for item in summary.trajectory_error_counts) == tuple(
        TrajectoryErrorCategory
    )
    assert all(item.count == 1 for item in summary.trajectory_error_counts)
    assert summary.trajectory_match == MetricCount(1, len(labels) + 1)


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_trajectory_error_count_rejects_invalid_counts(value) -> None:
    with pytest.raises(ValueError):
        TrajectoryErrorCount(TrajectoryErrorCategory.WRONG_TOOL, value)


def test_trajectory_error_count_requires_enum() -> None:
    with pytest.raises(TypeError):
        TrajectoryErrorCount("wrong_tool", 1)


@pytest.mark.parametrize(
    "updates",
    [
        {"trajectory_match": MetricCount(1, 2)},
        {"trajectory_match": MetricCount(0, 1)},
        {"tool_use_when_required": MetricCount(1, 2)},
        {"tool_abstention_when_not_needed": MetricCount(0, 1)},
    ],
)
def test_trajectory_summary_relationships(updates) -> None:
    summary = aggregate_tool_evaluations((evaluation("case"),))
    with pytest.raises(ValueError):
        replace(summary, **updates)


@pytest.mark.parametrize(
    "mode", ["type", "entry", "missing", "order", "total", "unnecessary", "invalid"]
)
def test_trajectory_error_distribution_consistency(mode) -> None:
    wrong_tool = evaluation("wrong", sequence_match=False, arguments_match=None)
    summary = aggregate_tool_evaluations((wrong_tool, no_tool("n")))
    counts = summary.trajectory_error_counts
    error: type[Exception] = ValueError
    if mode == "type":
        counts, error = list(counts), TypeError
    elif mode == "entry":
        counts, error = (*counts[:-1], object()), TypeError
    elif mode == "missing":
        counts = counts[:-1]
    elif mode == "order":
        counts = tuple(reversed(counts))
    elif mode == "total":
        counts = tuple(replace(item, count=0) for item in counts)
    else:
        # Keep the total intact and move the one error to a constrained label.
        target = (
            TrajectoryErrorCategory.UNNECESSARY_TOOL_CALL
            if mode == "unnecessary"
            else TrajectoryErrorCategory.INVALID_ARGUMENTS
        )
        counts = tuple(
            replace(item, count=int(item.category is target)) for item in counts
        )
    with pytest.raises(error):
        replace(summary, trajectory_error_counts=counts)

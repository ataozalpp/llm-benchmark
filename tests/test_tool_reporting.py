from dataclasses import FrozenInstanceError, replace

import pytest

from llm_benchmark.tool_evaluation import ToolEvaluationResult
from llm_benchmark.tool_loop import ToolLoopStopReason
from llm_benchmark.tool_reporting import (
    MetricCount,
    StopReasonCount,
    aggregate_tool_evaluations,
)


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
) -> ToolEvaluationResult:
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
    ["completion", "tool_sequence_match", "argument_match", "final_answer_match"],
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
    from llm_benchmark.tool_calling import NormalizationToolTurn
    from llm_benchmark.tool_conversation import (
        AssistantMessage,
        ToolConversation,
        UserMessage,
    )
    from llm_benchmark.tool_evaluation import ToolEvaluationCase, evaluate_tool_loop
    from llm_benchmark.tool_loop import ToolLoopResult
    from llm_benchmark.tool_provider_models import (
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
        case=ToolEvaluationCase("case", (), "done"), result=result
    )
    summary = aggregate_tool_evaluations((evaluated,))
    assert summary.final_answer_match == MetricCount(1, 1)
    assert summary.argument_match == MetricCount(1, 1)
    assert (
        summary.requested_call_count
        == summary.executed_call_count
        == summary.successful_call_count
        == 0
    )

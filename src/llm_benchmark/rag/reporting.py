"""Aggregate completed RAG suite outputs with explicit denominators."""

from collections import Counter

from .models import RagScenario
from .suite import RagSuiteCase

STATUSES = (
    "completed",
    "invalid_response",
    "incomplete_response",
    "provider_failed",
)


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _mean(values: list[float], eligible_count: int) -> dict[str, object]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "measured_case_count": len(values),
        "eligible_case_count": eligible_count,
    }


def summarize_rag(
    *,
    scenarios: tuple[RagScenario, ...],
    cases: tuple[RagSuiteCase, ...],
) -> dict[str, object]:
    if not isinstance(scenarios, tuple):
        raise TypeError("Scenarios must be a tuple.")

    if not isinstance(cases, tuple):
        raise TypeError("Cases must be a tuple.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    if any(not isinstance(item, RagScenario) for item in scenarios):
        raise TypeError("Expected RagScenario instances.")

    if any(not isinstance(item, RagSuiteCase) for item in cases):
        raise TypeError("Expected RagSuiteCase instances.")

    scenario_ids = tuple(item.query.query_id for item in scenarios)
    case_ids = tuple(item.result.query_id for item in cases)

    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Scenario query IDs must be unique.")

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case query IDs must be unique.")

    if set(scenario_ids) != set(case_ids):
        raise ValueError("Cases must cover exactly the supplied scenarios.")

    by_id = {item.result.query_id: item for item in cases}

    counts = Counter({status: 0 for status in STATUSES})
    answerable_count = 0
    unanswerable_count = 0
    correct_answers = 0
    correct_abstentions = 0
    full_context_count = 0
    full_context_correct = 0
    no_citations = 0
    failure_codes: Counter[str] = Counter()

    context_recalls: list[float] = []
    citation_validities: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []

    for scenario in scenarios:
        query_id = scenario.query.query_id
        case = by_id[query_id]
        result = case.result

        if case.ranking.query_id != query_id or case.context.query_id != query_id:
            raise ValueError("Ranking, context and result IDs must match.")

        if result.status not in STATUSES:
            raise ValueError("Unknown case status.")

        if result.status == "completed":
            if (
                result.answer is None
                or result.evaluation is None
                or result.response is None
            ):
                raise ValueError("Completed cases require answer and evaluation.")

            if result.evaluation.query_id != query_id:
                raise ValueError("Evaluation query id must match.")

            if result.response.finish_reason != "stop":
                raise ValueError("Completed cases must finish with stop.")
        else:
            if result.answer is not None or result.evaluation is not None:
                raise ValueError("Unsuccessful cases must not have evaluation.")

            if result.status == "provider_failed":
                if result.response is not None:
                    raise ValueError("Provider failure must not have a response.")
            elif result.response is None:
                raise ValueError("Invalid/Incomplete cases require a response.")
            elif result.status == "invalid_response":
                if result.response.finish_reason != "stop":
                    raise ValueError("Invalid responses must finish with stop.")
            elif result.response.finish_reason == "stop":
                raise ValueError("Incomplete responses must not finish with stop.")

        if result.status == "invalid_response":
            if result.failure_code is None:
                raise ValueError("Invalid responses require a failure code.")
            failure_codes[result.failure_code] += 1
        elif result.failure_code is not None:
            raise ValueError("Only invalid responses may have a failure code.")

        counts[result.status] += 1

        if scenario.answerable:
            answerable_count += 1
            gold_ids = scenario.relevant_chunk_ids
            context_ids = set(case.context.chunk_ids)
            recall = len(gold_ids & context_ids) / len(gold_ids)

            context_recalls.append(recall)

            exact = (
                result.evaluation is not None
                and result.evaluation.answer_exact_match is True
            )
            correct_answers += exact

            if recall == 1.0:
                full_context_count += 1
                full_context_correct += exact
        else:
            unanswerable_count += 1
            if result.answer is not None:
                correct_abstentions += result.answer.abstained

        if result.evaluation is not None:
            if result.evaluation.citation_count == 0:
                no_citations += 1

            validity = result.evaluation.citation_validity
            if validity is not None:
                citation_validities.append(validity)

        if result.response is not None:
            if result.response.input_tokens is not None:
                input_tokens.append(result.response.input_tokens)
            if result.response.output_tokens is not None:
                output_tokens.append(result.response.output_tokens)

    total = len(scenarios)

    return {
        "case_count": total,
        "answerable_case_count": answerable_count,
        "unanswerable_case_count": unanswerable_count,
        "status_counts": dict(counts),
        "invalid_response_reasons": dict(sorted(failure_codes.items())),
        "completion": _rate(counts["completed"], total),
        "answer_exact_match": _rate(correct_answers, answerable_count),
        "answer_exact_match_given_full_context": _rate(
            full_context_correct,
            full_context_count,
        ),
        "unanswerable_abstention": _rate(
            correct_abstentions,
            unanswerable_count,
        ),
        "context_evidence_recall": _mean(
            context_recalls,
            answerable_count,
        ),
        "citation_validity": _mean(
            citation_validities,
            counts["completed"],
        ),
        "no_citation_case_count": no_citations,
        "token_usage": {
            "input_tokens": {
                "observed_sum": sum(input_tokens),
                "measured_case_count": len(input_tokens),
                "total_case_count": total,
            },
            "output_tokens": {
                "observed_sum": sum(output_tokens),
                "measured_case_count": len(output_tokens),
                "total_case_count": total,
            },
        },
    }

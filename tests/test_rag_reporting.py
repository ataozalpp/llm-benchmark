import json
from dataclasses import replace

import pytest

from llm_benchmark.rag.case import run_rag_case
from llm_benchmark.rag.models import RagContext, RagScenario
from llm_benchmark.rag.providers import (
    RagGenerationResponse,
    RagProviderError,
)
from llm_benchmark.rag.reporting import summarize_rag
from llm_benchmark.rag.suite import RagSuiteCase, run_rag_suite
from llm_benchmark.retrieval.models import (
    Chunk,
    RankedRetrieval,
    RetrievalQuery,
)


class StubRetriever:
    def retrieve(self, query, *, top_k):
        return RankedRetrieval(
            query_id=query.query_id,
            chunk_ids=("refund",),
        )


class StubProvider:
    def generate(self, request):
        if request.query.query_id == "failed":
            raise RagProviderError("Unavailable")

        answer = "30 days" if request.query.query_id == "correct" else "60 days"

        return RagGenerationResponse(
            text=(
                '{"answer":"' + answer + '","citations":["refund"],"abstained":false}'
            ),
            finish_reason="stop",
            input_tokens=20,
            output_tokens=10,
        )


def test_provider_failure_remains_in_accuracy_denominator():
    scenarios = tuple(
        RagScenario(
            query=RetrievalQuery(query_id=query_id, text="Refund period?"),
            reference_answers=("30 days",),
            relevant_chunk_ids=frozenset({"refund"}),
        )
        for query_id in ("correct", "wrong", "failed")
    )

    suite = run_rag_suite(
        chunks=(
            Chunk(
                chunk_id="refund",
                document_id="policy",
                text="Refunds are available within 30 days.",
            ),
        ),
        scenarios=scenarios,
        retriever=StubRetriever(),
        provider=StubProvider(),
        k=1,
        max_context_chars=1000,
    )

    summary = summarize_rag(
        scenarios=scenarios,
        cases=suite.cases,
    )

    assert summary["case_count"] == 3
    assert summary["answer_exact_match"] == {
        "numerator": 1,
        "denominator": 3,
        "rate": 1 / 3,
    }
    assert summary["completion"]["numerator"] == 2
    assert summary["status_counts"]["provider_failed"] == 1

    assert summary["context_evidence_recall"]["mean"] == 1.0

    assert summary["token_usage"]["input_tokens"] == {
        "observed_sum": 40,
        "measured_case_count": 2,
        "total_case_count": 3,
    }


def make_case(
    query_id="q1",
    *,
    answerable=True,
    answer="30 days",
    citations=("refund",),
    abstained=False,
    status="completed",
    input_tokens=None,
    output_tokens=None,
):
    scenario = RagScenario(
        query=RetrievalQuery(query_id=query_id, text="Refund period?"),
        reference_answers=("30 days",) if answerable else (),
        relevant_chunk_ids=frozenset({"refund"}) if answerable else frozenset(),
        answerable=answerable,
    )
    context = RagContext(
        query_id=query_id,
        chunks=(Chunk(chunk_id="refund", document_id="policy", text="30 days."),),
    )

    class Provider:
        def generate(self, request):
            if status == "provider_failed":
                raise RagProviderError("Unavailable")
            return RagGenerationResponse(
                text="bad json"
                if status == "invalid_response"
                else json.dumps(
                    {
                        "answer": answer,
                        "citations": citations,
                        "abstained": abstained,
                    }
                ),
                finish_reason="length" if status == "incomplete_response" else "stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    result = run_rag_case(scenario=scenario, context=context, provider=Provider())
    return scenario, RagSuiteCase(
        ranking=RankedRetrieval(query_id=query_id, chunk_ids=("refund",)),
        context=context,
        result=result,
    )


def summarize_pairs(*pairs):
    return summarize_rag(
        scenarios=tuple(p[0] for p in pairs), cases=tuple(p[1] for p in pairs)
    )


def test_unanswerable_denominators_include_failures_and_wrong_answers():
    report = summarize_pairs(
        make_case("a", answerable=False, answer="", citations=(), abstained=True),
        make_case("b", answerable=False),
        make_case("c", answerable=False, status="provider_failed"),
    )
    assert report["answer_exact_match"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    assert report["unanswerable_abstention"] == {
        "numerator": 1,
        "denominator": 3,
        "rate": 1 / 3,
    }
    assert report["context_evidence_recall"] == {
        "mean": None,
        "measured_case_count": 0,
        "eligible_case_count": 0,
    }


def test_all_failure_statuses_remain_in_denominator_and_retain_observed_usage():
    report = summarize_pairs(
        make_case("a", status="invalid_response", input_tokens=5, output_tokens=2),
        make_case("b", status="incomplete_response", input_tokens=7, output_tokens=3),
        make_case("c", status="provider_failed"),
    )
    assert report["completion"] == {"numerator": 0, "denominator": 3, "rate": 0.0}
    assert report["answer_exact_match"]["denominator"] == 3
    assert report["context_evidence_recall"]["mean"] == 1.0
    assert report["citation_validity"]["mean"] is None
    assert report["token_usage"]["output_tokens"] == {
        "observed_sum": 5,
        "measured_case_count": 2,
        "total_case_count": 3,
    }
    assert report["token_usage"]["input_tokens"]["observed_sum"] == 12
    assert set(report["token_usage"]) == {"input_tokens", "output_tokens"}


def test_missing_usage_is_distinct_from_measured_zero():
    report = summarize_pairs(
        make_case("a"), make_case("b", input_tokens=0, output_tokens=0)
    )
    for name in ("input_tokens", "output_tokens"):
        assert report["token_usage"][name] == {
            "observed_sum": 0,
            "measured_case_count": 1,
            "total_case_count": 2,
        }


def test_citation_mean_is_per_case_and_excludes_undefined_values():
    report = summarize_pairs(
        make_case("a", citations=()),
        make_case("b", citations=("refund",)),
        make_case("c", citations=("refund", "fake")),
    )
    assert report["citation_validity"] == {
        "mean": 0.75,
        "measured_case_count": 2,
        "eligible_case_count": 3,
    }
    assert report["no_citation_case_count"] == 1
    assert report["unanswerable_abstention"]["rate"] is None
    assert report["status_counts"]["provider_failed"] == 0


def test_result_order_does_not_change_summary():
    a, b = make_case("a"), make_case("b", answer="60 days")
    expected = summarize_pairs(a, b)
    assert summarize_rag(scenarios=(a[0], b[0]), cases=(b[1], a[1])) == expected
    json.dumps(expected, allow_nan=False)


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "duplicate_case", "duplicate_scenario"]
)
def test_invalid_case_coverage_is_rejected(kind):
    a, b = make_case("a"), make_case("b")
    scenarios, cases = (a[0],), (a[1],)
    if kind == "missing":
        cases = ()
    elif kind == "extra":
        cases = (a[1], b[1])
    elif kind == "duplicate_case":
        cases = (a[1], a[1])
    else:
        scenarios = (a[0], a[0])
    with pytest.raises(ValueError):
        summarize_rag(scenarios=scenarios, cases=cases)


@pytest.mark.parametrize(
    ("scenarios", "cases"),
    [([], ()), ((), []), ((), ()), ((None,), ()), ((make_case()[0],), (None,))],
)
def test_invalid_collections_are_rejected(scenarios, cases):
    with pytest.raises((TypeError, ValueError)):
        summarize_rag(scenarios=scenarios, cases=cases)


@pytest.mark.parametrize("field", ["ranking", "context", "evaluation"])
def test_mismatched_component_ids_are_rejected(field):
    scenario, case = make_case()
    if field == "evaluation":
        case = replace(
            case,
            result=replace(
                case.result,
                evaluation=replace(case.result.evaluation, query_id="other"),
            ),
        )
    else:
        case = replace(case, **{field: replace(getattr(case, field), query_id="other")})
    with pytest.raises(ValueError):
        summarize_pairs((scenario, case))


@pytest.mark.parametrize("field", ["answer", "evaluation", "response"])
def test_completed_requires_all_components(field):
    scenario, case = make_case()
    case = replace(case, result=replace(case.result, **{field: None}))
    with pytest.raises(ValueError):
        summarize_pairs((scenario, case))


@pytest.mark.parametrize(
    "status", ["unknown", "provider_failed", "invalid_response", "incomplete_response"]
)
def test_unsuccessful_status_cannot_keep_completed_fields(status):
    scenario, case = make_case()
    case = replace(case, result=replace(case.result, status=status))
    with pytest.raises(ValueError):
        summarize_pairs((scenario, case))


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("completed", "length"),
        ("invalid_response", "length"),
        ("incomplete_response", "stop"),
    ],
)
def test_finish_reason_must_match_status(status, reason):
    scenario, case = make_case(status=status)
    case = replace(
        case,
        result=replace(
            case.result, response=replace(case.result.response, finish_reason=reason)
        ),
    )
    with pytest.raises(ValueError):
        summarize_pairs((scenario, case))


@pytest.mark.parametrize("status", ["invalid_response", "incomplete_response"])
def test_unsuccessful_generated_case_requires_response(status):
    scenario, case = make_case(status=status)
    case = replace(case, result=replace(case.result, response=None))
    with pytest.raises(ValueError):
        summarize_pairs((scenario, case))


def test_provider_failure_cannot_have_response():
    scenario, case = make_case(status="provider_failed")
    case = replace(
        case, result=replace(case.result, response=make_case()[1].result.response)
    )
    with pytest.raises(ValueError):
        summarize_pairs((scenario, case))

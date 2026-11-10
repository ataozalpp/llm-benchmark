"""Pure query-level comparison of RAG result rows."""

import pytest

from llm_benchmark.rag.comparison import compare_rag_outcomes
from llm_benchmark.rag.models import RagScenario
from llm_benchmark.retrieval.models import RetrievalQuery


def answerable(query_id):
    return RagScenario(
        query=RetrievalQuery(query_id, f"question {query_id}"),
        reference_answers=("yes",),
        relevant_chunk_ids=frozenset({"c1"}),
    )


def unanswerable(query_id):
    return RagScenario(
        query=RetrievalQuery(query_id, f"question {query_id}"),
        reference_answers=(),
        relevant_chunk_ids=frozenset(),
        answerable=False,
    )


SCENARIOS = (answerable("a1"), answerable("a2"), answerable("a3"), unanswerable("u1"))


def completed(query_id, *, exact=None, abstained=False, context=("c1",), answer="x"):
    return {
        "query_id": query_id,
        "status": "completed",
        "failure_code": None,
        "context_chunk_ids": list(context),
        "answer": {
            "answer": "" if abstained else answer,
            "citations": [],
            "abstained": abstained,
        },
        "evaluation": {
            "answer_exact_match": exact,
            "abstention_correct": abstained if exact is None else not abstained,
        },
    }


def invalid(query_id, code="abstention_with_content", context=("c1",)):
    return {
        "query_id": query_id,
        "status": "invalid_response",
        "failure_code": code,
        "context_chunk_ids": list(context),
        "answer": None,
        "evaluation": None,
    }


def baseline_rows():
    return [
        completed("a1", exact=True, answer="yes"),
        completed("a2", exact=True, answer="yes"),
        completed("a3", exact=False),
        invalid("u1"),
    ]


def candidate_rows():
    return [
        completed("a1", exact=True, answer="yes"),
        invalid("a2"),
        completed("a3", exact=True, answer="yes", context=("c1", "c2")),
        completed("u1", abstained=True),
    ]


def test_groups_rates_and_changes():
    report = compare_rag_outcomes(baseline_rows(), candidate_rows(), SCENARIOS)

    em = report["answer_exact_match"]
    assert em["query_ids"] == {
        "both_correct": ["a1"],
        "baseline_only": ["a2"],
        "candidate_only": ["a3"],
        "both_incorrect": [],
    }
    assert em["denominator"] == 3
    assert em["baseline_rate"] == em["candidate_rate"] == 2 / 3
    assert em["delta"] == 0

    abstention = report["unanswerable_abstention"]
    assert abstention["query_ids"]["candidate_only"] == ["u1"]
    assert abstention["baseline_rate"] == 0
    assert abstention["candidate_rate"] == 1
    assert abstention["delta"] == 1

    assert report["status_transitions"] == {
        "completed->invalid_response": ["a2"],
        "invalid_response->completed": ["u1"],
    }
    assert report["failure_code_changes"] == [
        {"query_id": "a2", "baseline": None, "candidate": "abstention_with_content"},
        {"query_id": "u1", "baseline": "abstention_with_content", "candidate": None},
    ]
    assert report["context_changed"] == ["a3"]
    assert [item["query_id"] for item in report["answer_changed"]] == ["a2", "a3", "u1"]
    assert report["answer_changed"][0]["candidate"] is None


def test_identical_rows_have_no_changes():
    report = compare_rag_outcomes(baseline_rows(), baseline_rows(), SCENARIOS)
    assert report["answer_exact_match"]["delta"] == 0
    assert report["status_transitions"] == {}
    assert report["failure_code_changes"] == []
    assert report["context_changed"] == []
    assert report["answer_changed"] == []


def test_answerable_abstention_is_not_counted_as_abstention_success():
    rows = baseline_rows()
    rows[2] = completed("a3", abstained=True, exact=False)
    report = compare_rag_outcomes(rows, rows, SCENARIOS)
    assert report["unanswerable_abstention"]["denominator"] == 1
    assert "a3" in report["answer_exact_match"]["query_ids"]["both_incorrect"]


def test_empty_denominator_is_null():
    scenarios = (answerable("a1"),)
    rows = [completed("a1", exact=True)]
    report = compare_rag_outcomes(rows, rows, scenarios)
    assert report["unanswerable_abstention"]["denominator"] == 0
    assert report["unanswerable_abstention"]["baseline_rate"] is None
    assert report["unanswerable_abstention"]["delta"] is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows.pop(), "exactly the suite query IDs"),
        (lambda rows: rows.append(dict(rows[0])), "duplicate query ID"),
        (lambda rows: rows[0].update(query_id=" "), "nonblank"),
        (lambda rows: rows[0].update(status="done"), "unknown status"),
        (lambda rows: rows[0].update(context_chunk_ids="c1"), "context IDs"),
        (lambda rows: rows[0].update(evaluation=None), "inconsistent"),
        (lambda rows: rows[3].update(evaluation={}), "inconsistent"),
        (lambda rows: rows[0].update(answer=[]), "invalid answer"),
        (lambda rows: rows.clear(), "no result rows"),
    ],
)
def test_invalid_rows_are_rejected(mutate, message):
    rows = baseline_rows()
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        compare_rag_outcomes(baseline_rows(), rows, SCENARIOS)


def test_scenarios_are_required():
    with pytest.raises(ValueError, match="at least one scenario"):
        compare_rag_outcomes(baseline_rows(), baseline_rows(), ())

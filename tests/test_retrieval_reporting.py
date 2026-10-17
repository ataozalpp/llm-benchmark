import pytest

from llm_benchmark.retrieval.evaluation import evaluate_retrieval
from llm_benchmark.retrieval.models import (
    RankedRetrieval,
    RetrievalEvaluationCase,
)
from llm_benchmark.retrieval.reporting import summarize_retrieval


def make_metrics(query_id, relevant, retrieved, k=2):
    return evaluate_retrieval(
        case=RetrievalEvaluationCase(
            query_id=query_id,
            relevant_chunk_ids=frozenset(relevant),
        ),
        result=RankedRetrieval(
            query_id=query_id,
            chunk_ids=tuple(retrieved),
        ),
        k=k,
    )


def test_macro_metrics_include_unsuccessful_queries():
    first = make_metrics("q1", {"a", "b"}, ("noise", "a"))
    second = make_metrics("q2", {"c"}, ())

    summary = summarize_retrieval((first,second))

    assert summary.query_count == 2
    assert summary.k == 2
    assert summary.hit_rate_at_k == pytest.approx(0.5)
    assert summary.mean_precision_at_k == pytest.approx(0.25)
    assert summary.mean_recall_at_k == pytest.approx(0.25)
    assert summary.mrr_at_k == pytest.approx(0.25)


def test_empty_metrics_are_rejected():
    with pytest.raises(ValueError):
        summarize_retrieval(())


def test_different_k_values_are_rejected():
    first = make_metrics("q1", {"a"}, ("a",), k=1)
    second = make_metrics("q2", {"b"}, ("b",), k=2)

    with pytest.raises(ValueError):
        summarize_retrieval((first, second))


def test_duplicate_queries_are_rejected():
    metrics = make_metrics("q1", {"a"}, ("a",))

    with pytest.raises(ValueError):
        summarize_retrieval((metrics, metrics))

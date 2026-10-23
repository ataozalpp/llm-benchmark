import pytest

from llm_benchmark.retrieval.evaluation import evaluate_retrieval
from llm_benchmark.retrieval.models import (
    RankedRetrieval,
    RetrievalEvaluationCase,
)


def test_metrics_use_only_top_k_results():
    case = RetrievalEvaluationCase(
        query_id="q1",
        relevant_chunk_ids=frozenset({"chunk-a", "chunk-b"}),
    )
    result = RankedRetrieval(
        query_id="q1",
        chunk_ids=("noise", "chunk-a", "chunk-b"),
    )

    metrics = evaluate_retrieval(case=case, result=result, k=2)

    assert metrics.retrieved_count == 2
    assert metrics.relevant_count == 2
    assert metrics.relevant_retrieved_count == 1
    assert metrics.hit_at_k == 1.0
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 0.5
    assert metrics.reciprocal_rank_at_k == 0.5


@pytest.mark.parametrize("chunk_ids", [(), ("noise",)])
def test_no_relevant_results_have_zero_scores(chunk_ids):
    case = RetrievalEvaluationCase(
        query_id="q1",
        relevant_chunk_ids=frozenset({"chunk-a"}),
    )
    result = RankedRetrieval(query_id="q1", chunk_ids=chunk_ids)

    metrics = evaluate_retrieval(case=case, result=result, k=3)

    assert metrics.hit_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.reciprocal_rank_at_k == 0.0


def test_short_result_keeps_k_as_precision_denominator():
    case = RetrievalEvaluationCase(
        query_id="q1",
        relevant_chunk_ids=frozenset({"chunk-a"}),
    )
    result = RankedRetrieval(
        query_id="q1",
        chunk_ids=("chunk-a",),
    )

    metrics = evaluate_retrieval(case=case, result=result, k=4)

    assert metrics.retrieved_count == 1
    assert metrics.precision_at_k == 0.25
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank_at_k == 1.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k):
    case = RetrievalEvaluationCase(
        query_id="q1",
        relevant_chunk_ids=frozenset({"chunk-a"}),
    )
    result = RankedRetrieval(query_id="q1", chunk_ids=())

    with pytest.raises(ValueError):
        evaluate_retrieval(case=case, result=result, k=k)


def test_query_mismatch_is_rejected():
    case = RetrievalEvaluationCase(
        query_id="q1",
        relevant_chunk_ids=frozenset({"chunk-a"}),
    )
    result = RankedRetrieval(
        query_id="q2",
        chunk_ids=("chunk-a",),
    )

    with pytest.raises(ValueError, match="Query IDs"):
        evaluate_retrieval(case=case, result=result, k=1)

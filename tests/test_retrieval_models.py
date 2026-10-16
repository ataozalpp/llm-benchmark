import pytest

from llm_benchmark.retrieval.models import (
    RankedRetrieval,
    RetrievalEvaluationCase,
)


def test_empty_retrieval_is_valid():
    result = RankedRetrieval(query_id="q1", chunk_ids=())

    assert result.chunk_ids == ()


def test_duplicate_chunk_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        RankedRetrieval(
            query_id="q1",
            chunk_ids=("chunk-a", "chunk-a"),
        )


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError, match="relevant"):
        RetrievalEvaluationCase(
            query_id="q1",
            relevant_chunk_ids=frozenset(),
        )


@pytest.mark.parametrize("query_id", [""," ", None])
def test_invalid_query_id_is_rejected(query_id):
    with pytest.raises(ValueError):
        RankedRetrieval(query_id=query_id, chunk_ids=())


@pytest.mark.parametrize("chunk_id", ["", " ", None])
def test_invalid_chunk_id_is_rejected(chunk_id):
    with pytest.raises(ValueError):
        RankedRetrieval(query_id="q1", chunk_ids=(chunk_id,))


def test_mutable_result_container_is_rejected():
    with pytest.raises(TypeError, match="tuple"):
        RankedRetrieval(query_id="q1", chunk_ids=["chunk-a"])


def test_mutable_gold_container_is_rejected():
    with pytest.raises(TypeError, match="frozenset"):
        RetrievalEvaluationCase(
            query_id="q1",
            relevant_chunk_ids={"chunk-a"},
        )


def test_invalid_gold_chunk_id_is_rejected():
    with pytest.raises(ValueError):
        RetrievalEvaluationCase(query_id="q1", relevant_chunk_ids=frozenset({""}))
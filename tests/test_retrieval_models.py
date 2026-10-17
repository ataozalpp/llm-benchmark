import pytest

from llm_benchmark.retrieval.models import (
    Chunk,
    RankedRetrieval,
    RetrievalEvaluationCase,
    RetrievalQuery,
    RetrievalScenario,
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


@pytest.mark.parametrize("query_id", ["", " ", None])
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


@pytest.mark.parametrize("text", ["", " ", None])
def test_invalid_chunk_text_is_rejected(text):
    with pytest.raises(ValueError):
        Chunk("c1", "d1", text)


@pytest.mark.parametrize("text", ["", " ", None])
def test_invalid_query_text_is_rejected(text):
    with pytest.raises(ValueError):
        RetrievalQuery("q1", text)


@pytest.mark.parametrize("chunk_id", ["", " ", None])
def test_invalid_chunk_identity_is_rejected(chunk_id):
    with pytest.raises(ValueError):
        Chunk(chunk_id, "d1", "valid text")


@pytest.mark.parametrize("document_id", ["", " ", None])
def test_invalid_document_id_is_rejected(document_id):
    with pytest.raises(ValueError):
        Chunk("c1", document_id, "valid text")


def test_scenario_rejects_query_id_mismatch():
    with pytest.raises(ValueError, match="Scenario query IDs"):
        RetrievalScenario(
            query=RetrievalQuery("q1", "refund policy"),
            evaluation=RetrievalEvaluationCase(
                query_id="q2",
                relevant_chunk_ids=frozenset({"refund"}),
            ),
        )

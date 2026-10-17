import pytest

from llm_benchmark.retrieval.evaluation import evaluate_retrieval
from llm_benchmark.retrieval.keyword import KeywordRetriever
from llm_benchmark.retrieval.models import (
    Chunk,
    RetrievalEvaluationCase,
    RetrievalQuery,
)


def test_retrieval_ranks_by_unique_token_overlap():
    retriver = KeywordRetriever(
        (
            Chunk("c1", "d1", "refund policy"),
            Chunk("c2", "d2", "refund"),
            Chunk("c3", "d3", "shipping"),
        )
    )

    result = retriver.retrieve(
        RetrievalQuery("q1", "REFUND policy!"),
        top_k=2,
    )

    assert result.query_id == "q1"
    assert result.chunk_ids == ("c1", "c2")


def test_equal_scores_use_chunk_id_not_corpus_order():
    chunks = (
        Chunk("b", "d1", "refund"),
        Chunk("a", "d2", "refund"),
    )
    query = RetrievalQuery("q1", "refund")

    first = KeywordRetriever(chunks).retrieve(query, top_k=2)
    second = KeywordRetriever(tuple(reversed(chunks))).retrieve(query, top_k=2)

    assert first.chunk_ids == ("a", "b")
    assert first == second


def test_repeated_words_do_not_increase_score():
    retriever = KeywordRetriever(
        (
            Chunk("b", "d1", "refund refund refund"),
            Chunk("a", "d2", "refund"),
        )
    )

    result = retriever.retrieve(
        RetrievalQuery("q1", "refund refund"),
        top_k=2,
    )

    assert result.chunk_ids == ("a", "b")


@pytest.mark.parametrize("query_text", ["unknown", "!!!"])
def test_no_token_overlap_returns_empty_results(query_text):
    retriever = KeywordRetriever(
        (Chunk("c1", "d1", "refund policy"),)
    )

    result = retriever.retrieve(RetrievalQuery("q1", query_text), top_k=3)

    assert result.chunk_ids == ()


def test_empty_corpus_returns_empty_results():
    retriever = KeywordRetriever(())

    result = retriever.retrieve(
        RetrievalQuery("q1", "refund"),
        top_k=2,
    )

    assert result.chunk_ids == ()


def test_duplicate_corpus_ids_are_rejected():
    with pytest.raises(ValueError):
        KeywordRetriever(
            (
                Chunk("c1", "d1", "refund"),
                Chunk("c1", "d2", "shipping"),
            )
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k):
    retriever = KeywordRetriever(())

    with pytest.raises(ValueError):
        retriever.retrieve(
            RetrievalQuery("q1", "refund"),
            top_k=top_k,
        )


def test_top_k_limits_results():
    retriever = KeywordRetriever(
        (
            Chunk("c1", "d1", "refund policy"),
            Chunk("c2", "d2", "refund"),
        )
    )

    result = retriever.retrieve(
        RetrievalQuery("q1", "refund policy"),
        top_k=1,
    )

    assert result.chunk_ids == ("c1",)


def test_retrieval_connects_to_existing_evaluator():
    retriever = KeywordRetriever(
        (
            Chunk("refund-policy", "policy", "refund period thirty days"),
            Chunk("shipping-policy", "policy", "shipping five days"),
        )
    )
    query = RetrievalQuery("q1", "refund period")

    result = retriever.retrieve(query, top_k=2)

    metrics = evaluate_retrieval(
        case=RetrievalEvaluationCase(
            query_id="q1",
            relevant_chunk_ids=frozenset({"refund-policy"}),
        ),
        result=result,
        k=2,
    )

    assert metrics.hit_at_k == 1.0
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank_at_k == 1.0
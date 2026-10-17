import pytest

from llm_benchmark.retrieval.keyword import KeywordRetriever
from llm_benchmark.retrieval.models import (
    Chunk,
    RankedRetrieval,
    RetrievalEvaluationCase,
    RetrievalQuery,
    RetrievalScenario,
)
from llm_benchmark.retrieval.suite import run_retrieval_suite


def scenario(query_id, text, relevant_ids):
    return RetrievalScenario(
        query=RetrievalQuery(query_id, text),
        evaluation=RetrievalEvaluationCase(
            query_id=query_id,
            relevant_chunk_ids=frozenset(relevant_ids),
        ),
    )


def test_keyword_suite_produces_rankings_and_summary():
    result = run_retrieval_suite(
        chunks=(
            Chunk("refund", "policy", "refund period thirty days"),
            Chunk("shipping", "policy", "shipping five days"),
        ),
        scenarios=(
            scenario("q1", "refund period", {"refund"}),
            scenario("q2", "delivery", {"shipping"}),
        ),
        retriever_factory=KeywordRetriever,
        k=2,
    )

    assert result.rankings[0].chunk_ids == ("refund",)
    assert result.rankings[1].chunk_ids == ()
    assert len(result.evaluations) == 2

    assert result.summary.hit_rate_at_k == pytest.approx(0.5)
    assert result.summary.mean_precision_at_k == pytest.approx(0.25)
    assert result.summary.mean_recall_at_k == pytest.approx(0.5)
    assert result.summary.mrr_at_k == pytest.approx(0.5)


def test_invalid_gold_is_rejected_before_retriever_creation():
    def must_not_create(chunks):
        pytest.fail("Retriever must not be created for invalid gold.")

    with pytest.raises(ValueError, match="Gold chunk IDs"):
        run_retrieval_suite(
            chunks=(Chunk("known", "doc", "known text"),),
            scenarios=(
                scenario("q1", "known", {"known"}),
                scenario("q2", "missing", {"missing"}),
            ),
            retriever_factory=must_not_create,
            k=2,
        )


@pytest.mark.parametrize(
    ("query_id", "chunk_ids", "message"),
    [
        ("wrong", ("known",), "wrong query ID"),
        ("q1", ("missing",), "exist in the corpus"),
        ("q1", ("known", "other"), "more than top_k"),
    ],
)
def test_invalid_retriever_output_is_rejected(
    query_id, chunk_ids, message
):
    class StubRetriever:
        def retrieve(self, query, *, top_k):
            return RankedRetrieval(query_id, chunk_ids)

    with pytest.raises(ValueError, match=message):
        run_retrieval_suite(
            chunks=(
                Chunk("known", "doc", "known text"),
                Chunk("other", "doc", "other text"),
            ),
            scenarios=(scenario("q1", "known", {"known"}),),
            retriever_factory=lambda chunks: StubRetriever(),
            k=1,
        )


@pytest.mark.parametrize(
    ("chunks", "scenarios", "k", "message"),
    [
        ((Chunk("known", "doc", "text"),), (), 1, "At least one scenario"),
        (
            (Chunk("known", "doc", "text"),),
            (scenario("q1", "text", {"known"}), scenario("q1", "text", {"known"})),
            1,
            "Scenario query IDs",
        ),
        (
            (Chunk("known", "doc", "one"), Chunk("known", "doc", "two")),
            (scenario("q1", "text", {"known"}),),
            1,
            "Corpus chunk IDs",
        ),
        *[
            (
                (Chunk("known", "doc", "text"),),
                (scenario("q1", "text", {"known"}),),
                k,
                "positive integer",
            )
            for k in (0, -1, True, 1.5)
        ],
    ],
)
def test_invalid_suite_input_is_rejected_before_factory(chunks, scenarios, k, message):
    def must_not_create(chunks):
        pytest.fail("Invalid input must fail before retriever creation.")

    with pytest.raises(ValueError, match=message):
        run_retrieval_suite(
            chunks=chunks,
            scenarios=scenarios,
            retriever_factory=must_not_create,
            k=k,
        )


def test_invalid_result_type_is_rejected():
    class BrokenRetriever:
        def retrieve(self, query, *, top_k):
            return ["known"]

    with pytest.raises(TypeError, match="RankedRetrieval"):
        run_retrieval_suite(
            chunks=(Chunk("known", "doc", "text"),),
            scenarios=(scenario("q1", "text", {"known"}),),
            retriever_factory=lambda chunks: BrokenRetriever(),
            k=1,
        )


def test_retriever_failure_propagates():
    failure = RuntimeError("retriever unavailable")

    class FailingRetriever:
        def retrieve(self, query, *, top_k):
            raise failure

    with pytest.raises(RuntimeError) as caught:
        run_retrieval_suite(
            chunks=(Chunk("known", "doc", "text"),),
            scenarios=(scenario("q1", "text", {"known"}),),
            retriever_factory=lambda chunks: FailingRetriever(),
            k=1,
        )

    assert caught.value is failure

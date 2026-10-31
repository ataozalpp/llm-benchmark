import pytest

from llm_benchmark.retrieval.hybrid import HybridRetriever
from llm_benchmark.retrieval.models import RankedRetrieval, RetrievalQuery


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def retrieve(self, query, *, top_k):
        self.calls.append((query.query_id, top_k))
        return RankedRetrieval(
            query_id=query.query_id,
            chunk_ids=self.ids[:top_k],
        )


def test_shared_document_gains_contributions_from_both_rankings():
    lexical = FixedRetriever(("a", "b"))
    dense = FixedRetriever(("c", "b"))
    retriever = HybridRetriever(
        lexical=lexical,
        dense=dense,
        candidate_k=5,
        rank_constant=60,
    )

    result = retriever.retrieve(
        RetrievalQuery(query_id="q", text="question"),
        top_k=3,
    )

    assert result.chunk_ids == ("b", "a", "c")
    assert lexical.calls == [("q", 5)]
    assert dense.calls == [("q", 5)]


def test_empty_child_does_not_remove_other_results():
    retriever = HybridRetriever(
        lexical=FixedRetriever(()),
        dense=FixedRetriever(("b", "a")),
    )

    result = retriever.retrieve(
        RetrievalQuery(query_id="q", text="question"),
        top_k=2,
    )

    assert result.chunk_ids == ("b", "a")


def test_both_empty_returns_empty_result():
    retriever = HybridRetriever(
        lexical=FixedRetriever(()),
        dense=FixedRetriever(()),
    )

    assert (
        retriever.retrieve(
            RetrievalQuery(query_id="q", text="question"),
            top_k=2,
        ).chunk_ids
        == ()
    )


def test_duplicate_across_children_is_returned_once():
    retriever = HybridRetriever(
        lexical=FixedRetriever(("a",)),
        dense=FixedRetriever(("a",)),
    )

    assert retriever.retrieve(
        RetrievalQuery(query_id="q", text="question"),
        top_k=2,
    ).chunk_ids == ("a",)


@pytest.mark.parametrize("value", [0, -1, True, "50"])
def test_invalid_candidate_depth(value):
    with pytest.raises(ValueError):
        HybridRetriever(
            lexical=FixedRetriever(()),
            dense=FixedRetriever(()),
            candidate_k=value,
        )


@pytest.mark.parametrize("value", [0, -1, True, "60"])
def test_invalid_rank_constant(value):
    with pytest.raises(ValueError):
        HybridRetriever(
            lexical=FixedRetriever(()),
            dense=FixedRetriever(()),
            rank_constant=value,
        )


def test_top_k_cannot_exceed_candidate_depth():
    lexical = FixedRetriever(("a",))
    retriever = HybridRetriever(
        lexical=lexical,
        dense=FixedRetriever(()),
        candidate_k=2,
    )

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.retrieve(
            RetrievalQuery(query_id="q", text="question"),
            top_k=3,
        )

    assert lexical.calls == []


def test_wrong_query_id_from_child_is_rejected():
    class WrongQueryRetriever:
        def retrieve(self, query, *, top_k):
            return RankedRetrieval(query_id="other", chunk_ids=("a",))

    retriever = HybridRetriever(
        lexical=WrongQueryRetriever(),
        dense=FixedRetriever(()),
    )

    with pytest.raises(ValueError, match="another query"):
        retriever.retrieve(
            RetrievalQuery(query_id="q", text="question"),
            top_k=1,
        )

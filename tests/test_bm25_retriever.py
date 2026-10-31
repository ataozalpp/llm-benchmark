import math
import re

import pytest
from rank_bm25 import BM25Okapi

from llm_benchmark.retrieval.bm25 import BM25Retriever
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery
from llm_benchmark.retrieval.profile import BM25_PROFILE


def chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
    )


def retrieve(chunks, text, top_k=10):
    return (
        BM25Retriever(chunks)
        .retrieve(
            RetrievalQuery(
                query_id="q1",
                text=text,
            ),
            top_k=top_k,
        )
        .chunk_ids
    )


def test_ranking_matches_backend_scores_for_matching_documents():
    chunks = (
        chunk("c", "bread oven"),
        chunk("a", "cat sofa"),
        chunk("b", "cat cat garden"),
    )
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    tokens = [re.findall(r"\w+", item.text.casefold()) for item in ordered]
    backend = BM25Okapi(
        tokens,
        k1=BM25_PROFILE.k1,
        b=BM25_PROFILE.b,
        epsilon=BM25_PROFILE.epsilon,
    )
    scores = backend.get_scores(["cat"])

    matching = [
        index
        for index, document_tokens in enumerate(tokens)
        if "cat" in document_tokens
    ]
    expected = tuple(
        ordered[index].chunk_id
        for index in sorted(
            matching,
            key=lambda index: (
                -float(scores[index]),
                ordered[index].chunk_id,
            ),
        )
    )

    assert retrieve(chunks, "CAT cat!") == expected


def test_zero_score_matching_document_is_preserved():
    # N=2 and df=1 gives zero IDF in this backend.
    chunks = (
        chunk("cat", "cat"),
        chunk("dog", "dog"),
    )

    assert retrieve(chunks, "cat") == ("cat",)


def test_negative_score_matching_document_is_preserved():
    backend = BM25Okapi([["cat"]], k1=1.2, b=0.75, epsilon=0.25)
    assert float(backend.get_scores(["cat"])[0]) == pytest.approx(
        0.25 * math.log(1 / 3)
    )
    assert retrieve((chunk("a", "cat"),), "cat") == ("a",)


def test_equal_scores_use_chunk_id_and_ignore_corpus_order():
    chunks = (chunk("b", "cat"), chunk("a", "cat"))
    assert retrieve(chunks, "cat") == ("a", "b")
    assert retrieve(tuple(reversed(chunks)), "cat") == ("a", "b")


@pytest.mark.parametrize("text", ["spaceship", "..."])
def test_no_matching_tokens_returns_no_results(text):
    assert retrieve((chunk("a", "cat"),), text) == ()


@pytest.mark.parametrize("chunks", [(), (chunk("punctuation", "..."),)])
def test_tokenless_corpus_does_not_initialize_backend(chunks, monkeypatch):
    import rank_bm25

    def forbidden(*args, **kwargs):
        pytest.fail("Tokenless corpus must not initialize the backend.")

    monkeypatch.setattr(rank_bm25, "BM25Okapi", forbidden)
    assert retrieve(chunks, "cat") == ()


def test_mixed_tokenless_and_nonempty_documents():
    assert retrieve((chunk("empty", "..."), chunk("cat", "cat")), "cat") == ("cat",)


@pytest.mark.parametrize("top_k", [0, -1, True, "2", 1.5, None])
def test_invalid_top_k_is_rejected_even_for_empty_corpus(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        retrieve((), "cat", top_k=top_k)


def test_top_k_limits_results():
    assert retrieve((chunk("b", "cat"), chunk("a", "cat")), "cat", 1) == ("a",)


def test_duplicate_chunk_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever((chunk("a", "cat"), chunk("a", "dog")))


@pytest.mark.parametrize("chunks", [[], None, "cat"])
def test_invalid_corpus_container_is_rejected(chunks):
    with pytest.raises(TypeError, match="tuple"):
        BM25Retriever(chunks)


def test_invalid_corpus_item_is_rejected():
    with pytest.raises(TypeError, match="Chunk"):
        BM25Retriever(("cat",))


def test_invalid_query_type_is_rejected():
    with pytest.raises(TypeError, match="RetrievalQuery"):
        BM25Retriever(()).retrieve("cat", top_k=1)


def test_backend_version_mismatch_is_rejected(monkeypatch):
    from llm_benchmark.retrieval import bm25

    monkeypatch.setattr(bm25, "version", lambda name: "unexpected")
    with pytest.raises(ValueError, match="version"):
        BM25Retriever((chunk("a", "cat"),))


@pytest.fixture
def captured_backend(monkeypatch):
    import rank_bm25

    captured = {}

    class Backend:
        def __init__(self, corpus, **kwargs):
            captured["corpus"] = corpus
            captured["options"] = kwargs

        def get_scores(self, terms):
            captured["query"] = terms
            return captured["scores"]

    monkeypatch.setattr(rank_bm25, "BM25Okapi", Backend)
    return captured


def test_backend_receives_profile_options_and_canonical_tokens(captured_backend):
    retriever = BM25Retriever((chunk("b", "DOG!"), chunk("a", "CAT cat!")))
    captured_backend["scores"] = [1.0, 2.0]
    result = retriever.retrieve(
        RetrievalQuery(query_id="original-query", text="DOG cat CAT!"), top_k=10
    )
    assert captured_backend["corpus"] == [["cat", "cat"], ["dog"]]
    assert captured_backend["options"] == {"k1": 1.2, "b": 0.75, "epsilon": 0.25}
    assert captured_backend["query"] == ["cat", "dog"]
    assert result.query_id == "original-query"
    assert result.chunk_ids == ("b", "a")


@pytest.mark.parametrize("scores", [[], [1.0, 2.0]])
def test_wrong_score_count_is_rejected(captured_backend, scores):
    retriever = BM25Retriever((chunk("a", "cat"),))
    captured_backend["scores"] = scores
    with pytest.raises(ValueError, match="score count"):
        retriever.retrieve(RetrievalQuery(query_id="q", text="cat"), top_k=1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_are_rejected(captured_backend, score):
    retriever = BM25Retriever((chunk("a", "cat"),))
    captured_backend["scores"] = [score]
    with pytest.raises(ValueError, match="non-finite"):
        retriever.retrieve(RetrievalQuery(query_id="q", text="cat"), top_k=1)


def test_nonmatching_document_cannot_displace_negative_match(captured_backend):
    retriever = BM25Retriever((chunk("a", "cat"), chunk("b", "dog")))
    captured_backend["scores"] = [-2.0, 0.0]
    result = retriever.retrieve(RetrievalQuery(query_id="q", text="cat"), top_k=1)
    assert result.chunk_ids == ("a",)


def test_no_match_skips_backend_scoring(captured_backend):
    retriever = BM25Retriever((chunk("a", "cat"),))
    # No scores are supplied: a backend call would fail this test.
    result = retriever.retrieve(RetrievalQuery(query_id="q", text="dog"), top_k=1)
    assert result.chunk_ids == ()
    assert "query" not in captured_backend


def test_profile_describes_candidate_and_tie_break_policies():
    assert BM25_PROFILE.tie_break == "chunk-id-ascending"
    assert BM25_PROFILE.corpus_statistics == "all-chunks-including-zero-token"
    assert BM25_PROFILE.score_filter == "none"

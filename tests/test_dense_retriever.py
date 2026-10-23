import pytest

from llm_benchmark.retrieval.dense import DenseRetriever
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery


class StubEmbedder:
    def __init__(self, documents, query):
        self.documents = documents
        self.query = query
        self.document_calls = 0
        self.query_calls = 0

    def encode_documents(self, texts):
        self.document_calls += 1
        return self.documents

    def encode_query(self, text):
        self.query_calls += 1
        return self.query


def chunks():
    return (
        Chunk("b", "doc-b", "First document"),
        Chunk("a", "doc-a", "Second document"),
        Chunk("c", "doc-c", "Third document"),
    )


def test_cosine_ranking_and_tie_break():
    embedder = StubEmbedder(
        documents=((2.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        query=(3.0, 0.0),
    )
    retriever = DenseRetriever(chunks(), embedder=embedder)

    result = retriever.retrieve(
        RetrievalQuery("q", "query"),
        top_k=3,
    )

    assert result.chunk_ids == ("a", "b", "c")


def test_corpus_is_encoded_once():
    embedder = StubEmbedder(
        documents=((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        query=(1.0, 0.0),
    )
    retriever = DenseRetriever(chunks(), embedder=embedder)

    retriever.retrieve(RetrievalQuery("q1", "first"), top_k=1)
    retriever.retrieve(RetrievalQuery("q2", "second"), top_k=2)

    assert embedder.document_calls == 1
    assert embedder.query_calls == 2


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (0.0, 0.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (True, 1.0),
    ],
)
def test_invalid_document_vector(vector):
    embedder = StubEmbedder(
        documents=(vector,),
        query=(1.0, 0.0),
    )

    with pytest.raises(ValueError):
        DenseRetriever(
            (Chunk("a", "doc", "text"),),
            embedder=embedder,
        )


def test_document_dimension_mismatch():
    embedder = StubEmbedder(
        documents=((1.0, 0.0), (1.0,), (0.0, 1.0)),
        query=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="dimensions"):
        DenseRetriever(chunks(), embedder=embedder)


def test_document_count_mismatch():
    embedder = StubEmbedder(
        documents=((1.0, 0.0),),
        query=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="count"):
        DenseRetriever(chunks(), embedder=embedder)


def test_query_dimension_mismatch():
    embedder = StubEmbedder(
        documents=((1.0, 0.0),),
        query=(1.0,),
    )
    retriever = DenseRetriever(
        (Chunk("a", "doc", "text"),),
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="dimension"):
        retriever.retrieve(RetrievalQuery("q", "query"), top_k=1)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    retriever = DenseRetriever(
        (),
        embedder=StubEmbedder((), (1.0,)),
    )

    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve(RetrievalQuery("q", "query"), top_k=top_k)


def test_empty_corpus_does_not_call_embedder():
    embedder = StubEmbedder((), (1.0,))
    retriever = DenseRetriever((), embedder=embedder)

    result = retriever.retrieve(
        RetrievalQuery("q", "query"),
        top_k=10,
    )

    assert result.chunk_ids == ()
    assert embedder.document_calls == 0
    assert embedder.query_calls == 0


def test_negative_similarity_is_not_filtered():
    embedder = StubEmbedder(
        documents=((-1.0, 0.0),),
        query=(1.0, 0.0),
    )
    retriever = DenseRetriever(
        (Chunk("a", "doc", "text"),),
        embedder=embedder,
    )

    result = retriever.retrieve(
        RetrievalQuery("q", "query"),
        top_k=10,
    )

    assert result.chunk_ids == ("a",)


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (0.0, 0.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (True, 1.0),
    ],
)
def test_invalid_query_vector(vector):
    embedder = StubEmbedder(
        documents=((1.0, 0.0),),
        query=vector,
    )
    retriever = DenseRetriever(
        (Chunk("a", "doc", "text"),),
        embedder=embedder,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            RetrievalQuery("q", "query"),
            top_k=1,
        )


def test_duplicate_chunk_ids_are_rejected():
    embedder = StubEmbedder(
        documents=((1.0, 0.0), (0.0, 1.0)),
        query=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="unique"):
        DenseRetriever(
            (
                Chunk("same", "doc-a", "first"),
                Chunk("same", "doc-b", "second"),
            ),
            embedder=embedder,
        )

    assert embedder.document_calls == 0


def test_large_finite_vectors_are_normalized():
    embedder = StubEmbedder(
        documents=((1e308, 1e308), (-1e308, -1e308)),
        query=(1e308, 1e308),
    )
    retriever = DenseRetriever(
        (
            Chunk("positive", "doc-a", "first"),
            Chunk("negative", "doc-b", "second"),
        ),
        embedder=embedder,
    )

    result = retriever.retrieve(
        RetrievalQuery("q", "query"),
        top_k=1,
    )

    assert result.chunk_ids == ("positive",)


def test_document_embeddings_must_follow_tuple_contract():
    embedder = StubEmbedder(
        documents=[(1.0, 0.0)],
        query=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="tuple"):
        DenseRetriever(
            (Chunk("a", "doc", "text"),),
            embedder=embedder,
        )

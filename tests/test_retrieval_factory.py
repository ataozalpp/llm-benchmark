"""Shared retriever construction for retrieval and RAG runs."""

import pytest

from llm_benchmark.retrieval import factory
from llm_benchmark.retrieval.bm25 import BM25Retriever
from llm_benchmark.retrieval.dense import DenseRetriever
from llm_benchmark.retrieval.hybrid import HybridRetriever
from llm_benchmark.retrieval.keyword import KeywordRetriever
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery
from llm_benchmark.retrieval.profile import (
    BM25_PROFILE,
    KEYWORD_PROFILE,
    DenseRetrievalProfile,
    HybridRetrievalProfile,
)
from llm_benchmark.retrieval.run_config import (
    DenseEmbeddingConfig,
    HybridFusionConfig,
    build_retrieval_profile,
    validate_retriever_settings,
)

CHUNKS = (
    Chunk("refund", "policy", "refund period thirty days"),
    Chunk("shipping", "policy", "shipping five days"),
)
EMBEDDING = DenseEmbeddingConfig(
    model_id="test/encoder",
    model_revision="a" * 40,
    embedding_dimension=2,
    max_sequence_length=256,
)


class FixedEmbedder:
    def __init__(self):
        self.model = object()

    def encode_documents(self, texts):
        return tuple((1.0, 0.0) if "refund" in text else (0.0, 1.0) for text in texts)

    def encode_query(self, text):
        return (1.0, 0.0) if "refund" in text else (0.0, 1.0)


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def load(profile, *, local_files_only):
        calls.append((profile, local_files_only))
        return FixedEmbedder()

    monkeypatch.setattr(factory, "load_cpu_embedder", load)
    return calls


@pytest.mark.parametrize(
    "profile,expected",
    [(KEYWORD_PROFILE, KeywordRetriever), (BM25_PROFILE, BM25Retriever)],
)
def test_lexical_profiles_do_not_load_models(profile, expected, loads):
    built = factory.build_retriever(profile, CHUNKS)
    assert isinstance(built.retriever, expected)
    assert built.model_load_ms == 0.0
    assert built.index_build_ms >= 0.0
    assert built.embedder is None
    assert loads == []


def test_dense_profile_loads_once_and_reports_embedder(loads):
    profile = EMBEDDING.to_profile()
    seen = []
    built = factory.build_retriever(
        profile, CHUNKS, local_files_only=False, on_embedder_loaded=seen.append
    )
    assert isinstance(built.retriever, DenseRetriever)
    assert loads == [(profile, False)]
    assert seen == [built.embedder]
    ranking = built.retriever.retrieve(RetrievalQuery("q", "refund"), top_k=1)
    assert ranking.chunk_ids == ("refund",)


def test_hybrid_profile_uses_dense_child_profile(loads):
    profile = HybridRetrievalProfile(
        dense=EMBEDDING.to_profile(), candidate_k=5, rank_constant=60
    )
    built = factory.build_retriever(profile, CHUNKS)
    assert isinstance(built.retriever, HybridRetriever)
    assert loads == [(profile.dense, True)]
    ranking = built.retriever.retrieve(RetrievalQuery("q", "refund period"), top_k=2)
    assert ranking.chunk_ids[0] == "refund"


def test_callback_runs_before_indexing(loads, monkeypatch):
    events = []
    original = factory.DenseRetriever

    def indexed(*args, **kwargs):
        events.append("index")
        return original(*args, **kwargs)

    monkeypatch.setattr(factory, "DenseRetriever", indexed)
    factory.build_retriever(
        EMBEDDING.to_profile(),
        CHUNKS,
        on_embedder_loaded=lambda embedder: events.append("loaded"),
    )
    assert events == ["loaded", "index"]


def test_callback_failure_prevents_indexing(loads, monkeypatch):
    def fail(embedder):
        raise OSError("manifest write failed")

    monkeypatch.setattr(factory, "DenseRetriever", pytest.fail)
    with pytest.raises(OSError):
        factory.build_retriever(
            EMBEDDING.to_profile(), CHUNKS, on_embedder_loaded=fail
        )


def test_invalid_inputs(loads):
    with pytest.raises(TypeError):
        factory.build_retriever(object(), CHUNKS)
    with pytest.raises(TypeError):
        factory.build_retriever(BM25_PROFILE, CHUNKS, on_embedder_loaded=1)


def test_profile_builder_matches_configured_retriever():
    fusion = HybridFusionConfig(candidate_k=5, rank_constant=10)
    assert build_retrieval_profile(
        retriever="keyword", embedding=None, hybrid=None
    ) is KEYWORD_PROFILE
    assert build_retrieval_profile(
        retriever="bm25", embedding=None, hybrid=None
    ) is BM25_PROFILE
    dense = build_retrieval_profile(retriever="dense", embedding=EMBEDDING, hybrid=None)
    assert isinstance(dense, DenseRetrievalProfile)
    hybrid = build_retrieval_profile(
        retriever="hybrid", embedding=EMBEDDING, hybrid=fusion
    )
    assert isinstance(hybrid, HybridRetrievalProfile)
    assert (hybrid.candidate_k, hybrid.rank_constant) == (5, 10)
    assert hybrid.dense == dense


@pytest.mark.parametrize(
    "retriever,embedding,hybrid",
    [
        ("dense", None, None),
        ("hybrid", None, None),
        ("hybrid", EMBEDDING, None),
        ("unknown", None, None),
    ],
)
def test_profile_builder_rejects_missing_settings(retriever, embedding, hybrid):
    with pytest.raises(ValueError):
        build_retrieval_profile(retriever=retriever, embedding=embedding, hybrid=hybrid)


@pytest.mark.parametrize(
    "retriever,embedding,hybrid,k",
    [
        ("dense", None, None, 2),
        ("keyword", EMBEDDING, None, 2),
        ("bm25", EMBEDDING, None, 2),
        ("hybrid", None, HybridFusionConfig(), 2),
        ("hybrid", EMBEDDING, None, 2),
        ("hybrid", EMBEDDING, HybridFusionConfig(candidate_k=2), 3),
        ("bm25", None, HybridFusionConfig(), 2),
    ],
)
def test_shared_setting_rules(retriever, embedding, hybrid, k):
    with pytest.raises(ValueError):
        validate_retriever_settings(
            retriever=retriever, embedding=embedding, hybrid=hybrid, k=k
        )


def test_hybrid_profile_default_candidate_k_is_valid():
    profile = HybridRetrievalProfile(dense=EMBEDDING.to_profile())
    assert profile.candidate_k == 50

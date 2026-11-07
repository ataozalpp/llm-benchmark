"""Construct retrievers from validated profiles with load and index timings."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .embedding_loader import load_cpu_embedder
from .hybrid import HybridRetriever
from .keyword import KeywordRetriever
from .models import Chunk
from .profile import (
    BM25RetrievalProfile,
    DenseRetrievalProfile,
    HybridRetrievalProfile,
    KeywordRetrieverProfile,
)
from .retriever import Retriever
from .run_config import RetrievalProfile


@dataclass(frozen=True)
class BuiltRetriever:
    retriever: Retriever
    model_load_ms: float
    index_build_ms: float
    embedder: Any | None = None


def _elapsed_ms(before: float) -> float:
    return (time.perf_counter() - before) * 1000


def build_retriever(
    profile: RetrievalProfile,
    chunks: tuple[Chunk, ...],
    *,
    local_files_only: bool = True,
    on_embedder_loaded: Callable[[Any], None] | None = None,
) -> BuiltRetriever:
    """Build a retriever; the callback runs after model load, before indexing."""
    if on_embedder_loaded is not None and not callable(on_embedder_loaded):
        raise TypeError("on_embedder_loaded must be callable or None.")

    if isinstance(profile, KeywordRetrieverProfile):
        before = time.perf_counter()
        retriever = KeywordRetriever(chunks)
        return BuiltRetriever(retriever, 0.0, _elapsed_ms(before))

    if isinstance(profile, BM25RetrievalProfile):
        before = time.perf_counter()
        retriever = BM25Retriever(chunks)
        return BuiltRetriever(retriever, 0.0, _elapsed_ms(before))

    if isinstance(profile, HybridRetrievalProfile):
        dense_profile = profile.dense
    elif isinstance(profile, DenseRetrievalProfile):
        dense_profile = profile
    else:
        raise TypeError("Unsupported retriever profile.")

    before = time.perf_counter()
    embedder = load_cpu_embedder(dense_profile, local_files_only=local_files_only)
    model_load_ms = _elapsed_ms(before)

    if on_embedder_loaded is not None:
        on_embedder_loaded(embedder)

    before = time.perf_counter()
    dense_retriever = DenseRetriever(chunks, embedder=embedder)

    if isinstance(profile, HybridRetrievalProfile):
        retriever = HybridRetriever(
            lexical=BM25Retriever(chunks),
            dense=dense_retriever,
            candidate_k=profile.candidate_k,
            rank_constant=profile.rank_constant,
        )
    else:
        retriever = dense_retriever

    return BuiltRetriever(retriever, model_load_ms, _elapsed_ms(before), embedder)

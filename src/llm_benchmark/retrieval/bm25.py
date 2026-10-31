"""Deterministic benchmark adapter for rank-bm25's BM25Okapi."""

import math
import re
from importlib.metadata import version

from .models import Chunk, RankedRetrieval, RetrievalQuery
from .profile import BM25_PROFILE

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


class BM25Retriever:
    def __init__(self, chunks: tuple[Chunk, ...]) -> None:
        if not isinstance(chunks, tuple):
            raise TypeError("Chunks must be a tuple.")

        if any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError("Every corpus item must be a Chunk.")

        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Corpus chunk IDs must be unique.")

        # Canonical corpus order also stabilizes backend initialization.
        ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        self._chunk_ids = tuple(chunk.chunk_id for chunk in ordered)

        tokenized = [_tokenize(chunk.text) for chunk in ordered]
        self._document_terms = tuple(frozenset(tokens) for tokens in tokenized)
        self._backend = None

        # rank-bm25 cannot initialize an empty or entirely tokenless corpus.
        if not any(tokenized):
            return

        observed_version = version(BM25_PROFILE.backend)
        if observed_version != BM25_PROFILE.backend_version:
            raise ValueError("Installed rank-bm25 version does not match the profile.")

        from rank_bm25 import BM25Okapi

        self._backend = BM25Okapi(
            [list(tokens) for tokens in tokenized],
            k1=BM25_PROFILE.k1,
            b=BM25_PROFILE.b,
            epsilon=BM25_PROFILE.epsilon,
        )

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int,
    ) -> RankedRetrieval:
        if not isinstance(query, RetrievalQuery):
            raise TypeError("Expected RetrievalQuery.")

        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_terms = sorted(set(_tokenize(query.text)))

        if self._backend is None or not query_terms:
            return RankedRetrieval(
                query_id=query.query_id,
                chunk_ids=(),
            )

        candidates = [
            index
            for index, terms in enumerate(self._document_terms)
            if not terms.isdisjoint(query_terms)
        ]

        if not candidates:
            return RankedRetrieval(
                query_id=query.query_id,
                chunk_ids=(),
            )

        scores = self._backend.get_scores(query_terms)

        if len(scores) != len(self._chunk_ids):
            raise ValueError("BM25 score count does not match the corpus.")

        values = tuple(float(score) for score in scores)
        if any(not math.isfinite(score) for score in values):
            raise ValueError("BM25 returned non-finite scores.")

        ranked = sorted(
            candidates,
            key=lambda index: (
                -values[index],
                self._chunk_ids[index],
            ),
        )

        return RankedRetrieval(
            query_id=query.query_id,
            chunk_ids=tuple(self._chunk_ids[index] for index in ranked[:top_k]),
        )

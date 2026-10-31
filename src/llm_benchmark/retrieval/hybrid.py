"""Reciprocal rank fusion over two retrieval implementations."""

import math

from .models import RankedRetrieval, RetrievalQuery
from .retriever import Retriever


class HybridRetriever:
    def __init__(
        self,
        *,
        lexical: Retriever,
        dense: Retriever,
        candidate_k: int = 50,
        rank_constant: int = 60,
    ) -> None:
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer.")

        if type(rank_constant) is not int or rank_constant <= 0:
            raise ValueError("rank_constant must be a positive integer.")

        self._lexical = lexical
        self._dense = dense
        self._candidate_k = candidate_k
        self._rank_constant = rank_constant

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

        if top_k > self._candidate_k:
            raise ValueError("top_k must not exceed candidate_k.")

        contributions: dict[str, list[float]] = {}

        for retriever in (self._lexical, self._dense):
            result = retriever.retrieve(
                query,
                top_k=self._candidate_k,
            )

            if not isinstance(result, RankedRetrieval):
                raise TypeError("Child retriever must return RankedRetrieval.")

            if result.query_id != query.query_id:
                raise ValueError("Child result belongs to another query.")

            if len(result.chunk_ids) > self._candidate_k:
                raise ValueError("Child result exceeds candidate_k.")

            for rank, chunk_id in enumerate(result.chunk_ids, start=1):
                contributions.setdefault(chunk_id, []).append(
                    1.0 / (self._rank_constant + rank)
                )

        ordered = sorted(
            contributions,
            key=lambda chunk_id: (
                -math.fsum(contributions[chunk_id]),
                chunk_id,
            ),
        )

        return RankedRetrieval(
            query_id=query.query_id,
            chunk_ids=tuple(ordered[:top_k]),
        )

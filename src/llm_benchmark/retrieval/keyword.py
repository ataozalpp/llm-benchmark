"""Deterministic retrieval using unique-token overlap."""

import re

from .models import Chunk, RankedRetrieval, RetrievalQuery

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.casefold()))


class KeywordRetriever:
    def __init__(self, chunks: tuple[Chunk, ...]) -> None:
        if not isinstance(chunks, tuple):
            raise TypeError("Chunks must be a tuple.")

        if any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError("Every corpus item must be a Chunk.")

        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Corpus chunk IDs must be unique.")

        self._index = tuple(
            (chunk.chunk_id, _tokenize(chunk.text))
            for chunk in chunks
        )

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int,
    ) -> RankedRetrieval:
        if not isinstance(query, RetrievalQuery):
            raise TypeError("Expected a RetrievalQuery.")

        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_tokens = _tokenize(query.text)

        ranked: list[tuple[int, str]] = []

        for chunk_id, chunk_tokens in self._index:
            score = len(query_tokens & chunk_tokens)

            if score > 0:
                ranked.append((score, chunk_id))

        ranked.sort(key=lambda item: (-item[0], item[1]))

        return RankedRetrieval(
            query_id=query.query_id,
            chunk_ids=tuple(
                chunk_id
                for _, chunk_id in ranked[:top_k]
            ),
        )
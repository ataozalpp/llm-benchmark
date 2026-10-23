"""Exact cosine-similarity retrieval over an in-memory corpus."""

import math

from .embeddings import Embedder, Vector
from .models import Chunk, RankedRetrieval, RetrievalQuery


def _normalize(vector: Vector) -> Vector:
    if not isinstance(vector, tuple) or not vector:
        raise ValueError("Embedding vectors must be nonempty tuples.")

    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in vector
    ):
        raise ValueError("Embedding values must be finite numbers.")

    scale = max(abs(value) for value in vector)

    if scale == 0:
        raise ValueError("Zero embedding vectors are unsupported.")

    scaled = tuple(value / scale for value in vector)
    norm = math.sqrt(math.fsum(value * value for value in scaled))

    return tuple(value / norm for value in scaled)


class DenseRetriever:
    def __init__(
        self,
        chunks: tuple[Chunk, ...],
        *,
        embedder: Embedder,
    ) -> None:
        if not isinstance(chunks, tuple):
            raise TypeError("Chunks must be a tuple.")

        if any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError("Every corpus item must be a Chunk.")

        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Corpus chunk IDs must be unique.")

        self._embedder = embedder
        self._index: tuple[tuple[str, Vector], ...] = ()
        self._dimension: int | None = None

        if not chunks:
            return

        vectors = embedder.encode_documents(tuple(chunk.text for chunk in chunks))

        if not isinstance(vectors, tuple):
            raise ValueError("Document embeddings must be a tuple.")

        if len(vectors) != len(chunks):
            raise ValueError("Embedding count must match the corpus.")

        normalized = tuple(_normalize(vector) for vector in vectors)
        dimension = len(normalized[0])
        if any(len(vector) != dimension for vector in normalized):
            raise ValueError("Document embedding dimensions must match.")

        self._dimension = dimension
        self._index = tuple(zip(chunk_ids, normalized, strict=True))

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

        if not self._index:
            return RankedRetrieval(query.query_id, ())

        dimension = self._dimension
        if dimension is None:
            raise RuntimeError("Nonempty dense index has no embedding dimension.")

        query_vector = _normalize(self._embedder.encode_query(query.text))

        if len(query_vector) != dimension:
            raise ValueError("Query embedding dimension does not match corpus.")

        scored = [
            (
                math.fsum(
                    left * right
                    for left, right in zip(
                        query_vector,
                        document_vector,
                        strict=True,
                    )
                ),
                chunk_id,
            )
            for chunk_id, document_vector in self._index
        ]

        scored.sort(key=lambda item: (-item[0], item[1]))

        return RankedRetrieval(
            query_id=query.query_id,
            chunk_ids=tuple(chunk_id for _, chunk_id in scored[:top_k]),
        )

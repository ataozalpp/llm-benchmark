"""Model independent embedding contract for retrieval."""

from typing import Protocol

Vector = tuple[float, ...]


class Embedder(Protocol):
    def encode_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[Vector, ...]:
        """Return one vector per document, preserving input order."""
        ...

    def encode_query(self, text: str) -> Vector:
        """Return a query vector in the document embedding space."""
        ...

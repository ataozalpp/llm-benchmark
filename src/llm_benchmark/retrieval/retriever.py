"""Interface for model-independent retrieval implementations."""

from typing import Protocol

from .models import RankedRetrieval, RetrievalQuery


class Retriever(Protocol):
    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int,
    ) -> RankedRetrieval:
        """Return at most top_k unique chunk IDs in ranking order."""
        ...
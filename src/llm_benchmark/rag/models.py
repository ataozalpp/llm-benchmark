"""Validated contracts for RAG generation and evaluation."""

import json
from dataclasses import dataclass

from ..retrieval.models import Chunk, RetrievalQuery


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string.")


def _require_unique_ids(values: object, label: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple.")

    for value in values:
        _require_text(value, label)

    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique IDs.")


@dataclass(frozen=True)
class RagScenario:
    query: RetrievalQuery
    reference_answers: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    answerable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("Expected RetrievalQuery.")

        if type(self.answerable) is not bool:
            raise TypeError("answerable must be a boolean.")

        for answer in self.reference_answers:
            _require_text(answer, "Reference answer")

        if not isinstance(self.relevant_chunk_ids, frozenset):
            raise TypeError("Relevant chunk IDs must be a frozenset.")

        for chunk_id in self.relevant_chunk_ids:
            _require_text(chunk_id, "Relevant chunk ID")

        if self.answerable:
            if not self.reference_answers or not self.relevant_chunk_ids:
                raise ValueError(
                    "Answerable scenarios require reference answers and evidence."
                )
        elif self.reference_answers or self.relevant_chunk_ids:
            raise ValueError(
                "Unanswerable scenarios must not declare answers or gold evidence."
            )


@dataclass(frozen=True)
class RagContext:
    query_id: str
    chunks: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "Query ID")

        if not isinstance(self.chunks, tuple):
            raise TypeError("Context chunks must be a tuple.")

        if any(not isinstance(chunk, Chunk) for chunk in self.chunks):
            raise TypeError("Context items must be Chunk instances.")

        ids = tuple(chunk.chunk_id for chunk in self.chunks)
        _require_unique_ids(ids, "Context chunk IDs")

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    def render(self) -> str:
        return json.dumps(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True)
class RagGenerationRequest:
    query: RetrievalQuery
    context: RagContext

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("Expected RetrievalQuery.")

        if not isinstance(self.context, RagContext):
            raise TypeError("Expected RagContext.")

        if self.query.query_id != self.context.query_id:
            raise ValueError("Query and context IDs must match.")


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: tuple[str, ...]
    abstained: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError("Answer must be a string.")

        if type(self.abstained) is not bool:
            raise TypeError("abstained must be a boolean.")

        _require_unique_ids(self.citations, "Citation IDs")

        if self.abstained:
            if self.answer or self.citations:
                raise ValueError(
                    "An abstention must have an empty answer and no citations."
                )
        elif not self.answer.strip():
            raise ValueError("A non-abstaining answer must be nonblank.")

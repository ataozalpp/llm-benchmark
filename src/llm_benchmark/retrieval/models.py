"""Validated inputs for ranked chunk retrieval evaluation."""

from dataclasses import dataclass


def _validate_id(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("IDs must be non blank strings.")


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query_id: str
    relevant_chunk_ids: frozenset[str]

    def __post_init__(self) -> None:
        _validate_id(self.query_id)

        if not isinstance(self.relevant_chunk_ids, frozenset):
            raise TypeError("Relevant chunk IDs must be a frozenset.")

        if not self.relevant_chunk_ids:
            raise ValueError("At least one relevant chunk is required.")

        for chunk_id in self.relevant_chunk_ids:
            _validate_id(chunk_id)


@dataclass(frozen=True)
class RankedRetrieval:
    query_id: str
    chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.query_id)

        if not isinstance(self.chunk_ids, tuple):
            raise TypeError("Retrieved chunk IDs must be a tuple.")

        for chunk_id in self.chunk_ids:
            _validate_id(chunk_id)

        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("Retrieved chunk IDs must be unique.")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_id(self.chunk_id)
        _validate_id(self.document_id)

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Chunk text must be a non blank string.")


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_id(self.query_id)

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Query text must be a non blank string.")


@dataclass(frozen=True)
class RetrievalScenario:
    query: RetrievalQuery
    evaluation: RetrievalEvaluationCase

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("Expected a RetrievalQuery.")

        if not isinstance(self.evaluation, RetrievalEvaluationCase):
            raise TypeError("Expected a RetrievalEvaluationCase.")

        if self.query.query_id != self.evaluation.query_id:
            raise ValueError("Scenario query IDs must match.")

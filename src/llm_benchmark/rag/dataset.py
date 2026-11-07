"""Bounded, strict JSON loading for project-owned RAG QA suites."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, StrictInt, field_validator

from ..retrieval.dataset import _Chunk, _Record, _reject_constant, _unique_object
from ..retrieval.models import Chunk, RetrievalQuery
from .models import RagScenario

MAX_DATASET_BYTES = 32 * 1024 * 1024


class _RagScenario(_Record):
    query_id: str
    text: str
    answerable: bool = True
    reference_answers: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("reference_answers", "relevant_chunk_ids")
    @classmethod
    def validate_items(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Duplicate values.")
        for item in value:
            if not item.strip() or "\x00" in item:
                raise ValueError("Invalid text item.")
            item.encode("utf-8")
        return value


class _RagDataset(_Record):
    schema_version: StrictInt
    suite_id: str
    chunks: list[_Chunk] = Field(min_length=1)
    scenarios: list[_RagScenario] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value):
        if value != 1:
            raise ValueError("Unsupported suite version.")
        return value


@dataclass(frozen=True)
class RagDataset:
    suite_id: str
    chunks: tuple[Chunk, ...]
    scenarios: tuple[RagScenario, ...]


def load_rag_dataset(path: Path) -> RagDataset:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_DATASET_BYTES + 1)
        if len(raw) > MAX_DATASET_BYTES:
            raise ValueError("Suite too large.")

        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        data = _RagDataset.model_validate(payload)
        chunks = tuple(Chunk(**item.model_dump()) for item in data.chunks)
        scenarios = tuple(
            RagScenario(
                query=RetrievalQuery(item.query_id, item.text),
                reference_answers=tuple(item.reference_answers),
                relevant_chunk_ids=frozenset(item.relevant_chunk_ids),
                answerable=item.answerable,
            )
            for item in data.scenarios
        )
        corpus_ids = {item.chunk_id for item in chunks}

        if len(corpus_ids) != len(chunks):
            raise ValueError("Duplicate chunk IDs.")

        if len({item.query.query_id for item in scenarios}) != len(scenarios):
            raise ValueError("Duplicate query IDs.")

        if any(not item.relevant_chunk_ids <= corpus_ids for item in scenarios):
            raise ValueError("Unknown gold IDs.")

        return RagDataset(data.suite_id, chunks, scenarios)

    except (OSError, TypeError, ValueError, RecursionError):
        raise ValueError("Invalid RAG suite dataset.") from None
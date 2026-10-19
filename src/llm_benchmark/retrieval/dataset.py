"""Bounded, strict JSON loading for project-owned retrieval suites."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from .models import Chunk, RetrievalEvaluationCase, RetrievalQuery, RetrievalScenario

MAX_DATASET_BYTES = 4 * 1024 * 1024


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def validate_text(cls, value):
        if isinstance(value, str):
            if not value.strip() or "\x00" in value:
                raise ValueError("Invalid text.")
            value.encode("utf-8")
        return value


class _Chunk(_Record):
    chunk_id: str
    document_id: str
    text: str


class _Scenario(_Record):
    query_id: str
    text: str
    relevant_chunk_ids: list[str] = Field(min_length=1)

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_gold(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Duplicate gold IDs.")
        for item in value:
            if not item.strip() or "\x00" in item:
                raise ValueError("Invalid gold ID.")
            item.encode("utf-8")
        return value


class _Dataset(_Record):
    schema_version: StrictInt
    suite_id: str
    chunks: list[_Chunk] = Field(min_length=1)
    scenarios: list[_Scenario] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value):
        if value != 1:
            raise ValueError("Unsupported suite version.")
        return value


@dataclass(frozen=True)
class RetrievalDataset:
    suite_id: str
    chunks: tuple[Chunk, ...]
    scenarios: tuple[RetrievalScenario, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number.")


def load_retrieval_dataset(path: Path) -> RetrievalDataset:
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
        data = _Dataset.model_validate(payload)
        chunks = tuple(Chunk(**item.model_dump()) for item in data.chunks)
        scenarios = tuple(
            RetrievalScenario(
                RetrievalQuery(item.query_id, item.text),
                RetrievalEvaluationCase(
                    item.query_id, frozenset(item.relevant_chunk_ids)
                ),
            )
            for item in data.scenarios
        )
        corpus_ids = {item.chunk_id for item in chunks}
        if len(corpus_ids) != len(chunks):
            raise ValueError("Duplicate chunk IDs.")
        if len({item.query.query_id for item in scenarios}) != len(scenarios):
            raise ValueError("Duplicate query IDs.")
        if any(
            not item.evaluation.relevant_chunk_ids <= corpus_ids for item in scenarios
        ):
            raise ValueError("Unknown gold IDs.")
        return RetrievalDataset(data.suite_id, chunks, scenarios)
    except (OSError, ValueError, RecursionError):
        raise ValueError("Invalid retrieval suite dataset.") from None

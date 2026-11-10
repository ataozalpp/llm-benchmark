from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO, Literal, Protocol

from .dataset_adapters import (
    DatasetAdapterError,
    DatasetFileFormat,
    load_tabular_examples,
)
from .dataset_storage import DatasetStorageError, LocalDatasetStorage
from .db.records import DatasetRecord

IngestionFileFormat = DatasetFileFormat | Literal["rag_json"]

_ADAPTER_TYPES: dict[DatasetFileFormat, str] = {
    "csv": "tabular_mcq_csv_v1",
    "jsonl": "tabular_mcq_jsonl_v1",
}
RAG_SUITE_ADAPTER_TYPE = "rag_suite_json_v1"


class DatasetRegistrationRepository(Protocol):
    def create(
        self,
        *,
        name: str,
        source_type: str,
        source_uri: str,
        revision: str | None,
        split: str,
        task_type: str,
        adapter_type: str,
        license: str | None = None,
        checksum: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRecord: ...


@dataclass(frozen=True)
class DatasetIngestionRequest:
    name: str
    file_format: IngestionFileFormat
    split: str = "test"
    revision: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset name must not be empty")
        if not self.split.strip():
            raise ValueError("Dataset split must not be empty")


@dataclass(frozen=True)
class DatasetIngestionResult:
    dataset: DatasetRecord
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    sample_count: int
    categories: tuple[str, ...]


class DatasetIngestionService:
    def __init__(
        self,
        *,
        storage: LocalDatasetStorage,
        datasets: DatasetRegistrationRepository,
    ) -> None:
        self._storage = storage
        self._datasets = datasets

    def ingest(
        self,
        request: DatasetIngestionRequest,
        stream: BinaryIO,
    ) -> DatasetIngestionResult:
        if request.file_format == "rag_json":
            return self._ingest_rag_suite(request, stream)
        stored_file = self._storage.store(
            stream,
            request.file_format,
        )
        try:
            stored_path = self._storage.resolve(
                stored_file.storage_key,
            )
            examples = load_tabular_examples(
                stored_path,
                request.file_format,
            )

            adapter_type = _ADAPTER_TYPES[request.file_format]
            categories = tuple(sorted({example.category for example in examples}))

            dataset = self._datasets.create(
                name=request.name.strip(),
                source_type="uploaded",
                source_uri=stored_file.storage_key,
                revision=request.revision,
                split=request.split.strip(),
                task_type="multiple_choice",
                adapter_type=adapter_type,
                license=request.license,
                checksum=f"sha256:{stored_file.checksum_sha256}",
                metadata={
                    "schema_version": "tabular_mcq_upload_v1",
                    "format": request.file_format,
                    "size_bytes": stored_file.size_bytes,
                    "sample_count": len(examples),
                    "categories": list(categories),
                },
            )
        except Exception:
            if stored_file.created_new_file:
                try:
                    self._storage.remove(stored_file.storage_key)
                except DatasetStorageError:
                    pass
            raise

        return DatasetIngestionResult(
            dataset=dataset,
            storage_key=stored_file.storage_key,
            checksum_sha256=stored_file.checksum_sha256,
            size_bytes=stored_file.size_bytes,
            sample_count=len(examples),
            categories=categories,
        )

    def _ingest_rag_suite(
        self,
        request: DatasetIngestionRequest,
        stream: BinaryIO,
    ) -> DatasetIngestionResult:
        # Imported lazily so MCQ ingestion does not load the RAG package.
        from .rag.dataset import load_rag_dataset
        from .rag.identity import corpus_fingerprint, suite_fingerprint

        stored_file = self._storage.store(stream, "json")
        try:
            try:
                suite = load_rag_dataset(
                    self._storage.resolve(stored_file.storage_key)
                )
            except ValueError:
                raise DatasetAdapterError("Uploaded RAG suite is invalid") from None
            answerable = sum(item.answerable for item in suite.scenarios)
            dataset = self._datasets.create(
                name=request.name.strip(),
                source_type="uploaded",
                source_uri=stored_file.storage_key,
                revision=request.revision,
                split=request.split.strip(),
                task_type="rag",
                adapter_type=RAG_SUITE_ADAPTER_TYPE,
                license=request.license,
                checksum=f"sha256:{stored_file.checksum_sha256}",
                metadata={
                    "schema_version": "rag_suite_upload_v1",
                    "format": "rag_json",
                    "size_bytes": stored_file.size_bytes,
                    "suite_id": suite.suite_id,
                    "chunk_count": len(suite.chunks),
                    "scenario_count": len(suite.scenarios),
                    "answerable_count": answerable,
                    "unanswerable_count": len(suite.scenarios) - answerable,
                    "corpus_hash": corpus_fingerprint(suite.chunks),
                    "suite_hash": suite_fingerprint(suite.scenarios),
                },
            )
        except Exception:
            if stored_file.created_new_file:
                try:
                    self._storage.remove(stored_file.storage_key)
                except DatasetStorageError:
                    pass
            raise

        return DatasetIngestionResult(
            dataset=dataset,
            storage_key=stored_file.storage_key,
            checksum_sha256=stored_file.checksum_sha256,
            size_bytes=stored_file.size_bytes,
            sample_count=len(suite.scenarios),
            categories=(),
        )

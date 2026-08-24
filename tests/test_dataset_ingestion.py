from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config

from llm_benchmark.dataset_adapters import DatasetAdapterError
from llm_benchmark.dataset_ingestion import (
    DatasetIngestionRequest,
    DatasetIngestionService,
)
from llm_benchmark.dataset_storage import LocalDatasetStorage
from llm_benchmark.db import create_db_engine, create_session_factory
from llm_benchmark.db.records import DatasetRecord
from llm_benchmark.db.repositories import DatasetRepository


def _dataset_record() -> DatasetRecord:
    now = datetime.now(UTC)
    return DatasetRecord(
        id=1,
        name="uploaded-dataset",
        source_type="uploaded",
        source_uri="upload://sha256/" + ("a" * 64) + ".csv",
        revision=None,
        split="test",
        task_type="multiple_choice",
        adapter_type="tabular_mcq_csv_v1",
        license=None,
        checksum="sha256:" + ("a" * 64),
        metadata_json={},
        is_active=True,
        created_at=now,
        updated_at=now,
    )


class RecordingDatasetRepository:
    def __init__(self, result: DatasetRecord) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

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
    ) -> DatasetRecord:
        self.calls.append(
            {
                "name": name,
                "source_type": source_type,
                "source_uri": source_uri,
                "revision": revision,
                "split": split,
                "task_type": task_type,
                "adapter_type": adapter_type,
                "license": license,
                "checksum": checksum,
                "metadata": metadata,
            }
        )
        return self.result


class FailingDatasetRepository:
    def create(
        self,
        **_values: object,
    ) -> DatasetRecord:
        raise RuntimeError("Repository persistence failed")


def test_ingestion_persists_only_registration_metadata_in_migrated_sqlite(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "ingestion.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")

    engine = create_db_engine(database_url)
    repository = DatasetRepository(create_session_factory(engine))
    storage = LocalDatasetStorage(tmp_path / "datasets")
    service = DatasetIngestionService(
        storage=storage,
        datasets=repository,
    )
    content = (
        b"sample_id,question,option_A,option_B,correct_answer,category\n"
        b"unique-001,Unique ingestion question,Alpha choice,Beta choice,B,science\n"
    )

    result = service.ingest(
        DatasetIngestionRequest(
            name="sqlite-upload",
            file_format="csv",
        ),
        BytesIO(content),
    )

    persisted = repository.list_active()
    assert len(persisted) == 1
    dataset = persisted[0]
    assert dataset.id == result.dataset.id
    assert dataset.source_type == "uploaded"
    assert dataset.source_uri == result.storage_key
    assert dataset.source_uri.startswith("upload://sha256/")
    assert str(tmp_path) not in dataset.source_uri
    assert dataset.adapter_type == "tabular_mcq_csv_v1"
    assert dataset.checksum == f"sha256:{result.checksum_sha256}"
    assert dataset.metadata_json == {
        "schema_version": "tabular_mcq_upload_v1",
        "format": "csv",
        "size_bytes": len(content),
        "sample_count": 1,
        "categories": ["science"],
    }
    assert storage.resolve(result.storage_key).read_bytes() == content

    engine.dispose()
    database_bytes = database_path.read_bytes()
    assert content not in database_bytes
    assert b"Unique ingestion question" not in database_bytes
    assert b"Alpha choice" not in database_bytes
    assert b"Beta choice" not in database_bytes
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "outputs").exists()


def test_ingest_stores_validates_and_registers_csv(
        tmp_path: Path,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    repository = RecordingDatasetRepository(_dataset_record())
    service = DatasetIngestionService(
        storage=storage,
        datasets=repository,
    )
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,Question,Alpha,Beta,A,test\n"
    )

    result = service.ingest(
        DatasetIngestionRequest(
            name="uploaded-dataset",
            file_format="csv",
        ),
        BytesIO(content),
    )

    assert result.sample_count == 1
    assert result.categories == ("test",)
    assert result.size_bytes == len(content)
    assert result.storage_key.startswith("upload://sha256/")
    assert len(repository.calls) == 1

    call = repository.calls[0]

    assert call["name"] == "uploaded-dataset"
    assert call["source_type"] == "uploaded"
    assert call["source_uri"] == result.storage_key
    assert call["revision"] is None
    assert call["split"] == "test"
    assert call["task_type"] == "multiple_choice"
    assert call["adapter_type"] == "tabular_mcq_csv_v1"
    assert call["license"] is None
    assert call["checksum"] == f"sha256:{result.checksum_sha256}"
    assert call["metadata"] == {
        "schema_version": "tabular_mcq_upload_v1",
        "format": "csv",
        "size_bytes": len(content),
        "sample_count": 1,
        "categories": ["test"],
    }


def test_ingest_stores_validates_and_registers_jsonl(
        tmp_path: Path,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    repository = RecordingDatasetRepository(_dataset_record())
    service = DatasetIngestionService(
        storage=storage,
        datasets=repository,
    )
    content = (
        b'{"sample_id":"002","question":"Choose one",'
        b'"options":["Alpha","Beta"],'
        b'"correct_answer":"B","category":"science"}\n'
    )

    result = service.ingest(
        DatasetIngestionRequest(
            name="uploaded-jsonl",
            file_format="jsonl",
            split="validation",
            revision="revision-1",
            license="test-license",
        ),
        BytesIO(content),
    )

    assert result.sample_count == 1
    assert result.categories == ("science",)
    assert result.size_bytes == len(content)
    assert result.storage_key.endswith(".jsonl")
    assert len(repository.calls) == 1

    call = repository.calls[0]

    assert call["name"] == "uploaded-jsonl"
    assert call["source_type"] == "uploaded"
    assert call["source_uri"] == result.storage_key
    assert call["revision"] == "revision-1"
    assert call["split"] == "validation"
    assert call["task_type"] == "multiple_choice"
    assert call["adapter_type"] == "tabular_mcq_jsonl_v1"
    assert call["license"] == "test-license"
    assert call["checksum"] == f"sha256:{result.checksum_sha256}"


def test_ingest_reports_sample_count_and_sorted_categories(
    tmp_path: Path,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    repository = RecordingDatasetRepository(_dataset_record())
    service = DatasetIngestionService(
        storage=storage,
        datasets=repository,
    )
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,First,Alpha,Beta,A,science\n"
        b"002,Second,True,False,B,history\n"
        b"003,Third,Yes,No,A,science\n"
    )

    result = service.ingest(
        DatasetIngestionRequest(
            name="multi-category",
            file_format="csv",
        ),
        BytesIO(content),
    )

    assert result.sample_count == 3
    assert result.categories == ("history", "science")

    metadata = repository.calls[0]["metadata"]

    assert metadata == {
        "schema_version": "tabular_mcq_upload_v1",
        "format": "csv",
        "size_bytes": len(content),
        "sample_count": 3,
        "categories": ["history", "science"],
    }


def test_ingestion_request_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="Dataset name must not be empty",
    ):
        DatasetIngestionRequest(
            name="   ",
            file_format="csv",
        )


def test_ingestion_request_rejects_empty_split() -> None:
    with pytest.raises(
        ValueError,
        match="Dataset split must not be empty",
    ):
        DatasetIngestionRequest(
            name="dataset",
            file_format="csv",
            split="   ",
        )


def test_invalid_dataset_removes_newly_stored_file(
        tmp_path: Path
) -> None:
    storage_root = tmp_path / "datasets"
    storage = LocalDatasetStorage(storage_root)
    repository = RecordingDatasetRepository(_dataset_record())
    service = DatasetIngestionService(
        storage=storage,
        datasets=repository,
    )
    invalid_content = b"not, a, valid, dataset\n"

    with pytest.raises(DatasetAdapterError) as raised:
        service.ingest(
            DatasetIngestionRequest(
                name="invalid",
                file_format="csv",
            ),
            BytesIO(invalid_content),
        )

    assert repository.calls == []
    assert str(raised.value) == (
        "CSV dataset is missing required columns: "
        "category, correct_answer, question, sample_id"
    )
    assert list(storage_root.rglob("*.csv")) == []


def test_repository_failure_preserves_reused_file(
    tmp_path: Path,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,Question,Alpha,Beta,A,test\n"
    )

    existing = storage.store(
        BytesIO(content),
        "csv",
    )

    assert existing.created_new_file is True

    duplicate = storage.store(
        BytesIO(content),
        "csv",
    )

    assert duplicate.storage_key == existing.storage_key
    assert duplicate.created_new_file is False

    existing_path = storage.resolve(existing.storage_key)

    assert existing_path.exists()

    service = DatasetIngestionService(
        storage=storage,
        datasets=FailingDatasetRepository(),
    )

    with pytest.raises(
        RuntimeError,
        match="Repository persistence failed",
    ) as raised:
        service.ingest(
            DatasetIngestionRequest(
                name="reused-dataset",
                file_format="csv",
            ),
            BytesIO(content),
        )

    assert existing_path.exists()
    assert str(raised.value) == "Repository persistence failed"


def test_repository_failure_removes_newly_stored_files(
        tmp_path: Path
) -> None:
    storage_root = tmp_path / "datasets"
    storage = LocalDatasetStorage(storage_root)
    service = DatasetIngestionService(
        storage=storage,
        datasets=FailingDatasetRepository(),
    )
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,Question,Alpha,Beta,A,test\n"
    )

    with pytest.raises(
        RuntimeError,
        match="Repository persistence failed",
    ) as raised:
        service.ingest(
            DatasetIngestionRequest(
                name="new-dataset",
                file_format="csv"
            ),
            BytesIO(content),
        )

    assert str(raised.value) == "Repository persistence failed"
    assert list(storage_root.rglob("*.csv")) == []

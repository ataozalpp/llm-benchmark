from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from llm_benchmark.api import create_app
from llm_benchmark.api.app_dependencies import get_dataset_ingestion_service
from llm_benchmark.dataset_storage import (
    DatasetFileTooLargeError,
    DatasetFinalizationError,
    DatasetStorageWriteError,
    DatasetStreamError,
    LocalDatasetStorage,
)
from llm_benchmark.db.registry import create_registry_repositories


@pytest.fixture
def upload_api_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, Path]]:
    database_path = tmp_path / "upload-api.sqlite"
    database_url = (
        f"sqlite:///{database_path.as_posix()}"
    )

    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option(
        "sqlalchemy.url",
        database_url,
    )
    command.upgrade(alembic_config, "head")

    registry = create_registry_repositories(database_url)
    storage_root = tmp_path / "datasets"

    app = create_app(
        registry,
        dataset_storage_root=storage_root,
    )

    with TestClient(app) as client:
        yield client, storage_root


def test_upload_csv_registers_and_stores_dataset(
    upload_api_client: tuple[TestClient, Path],
    tmp_path: Path,
) -> None:
    client, storage_root = upload_api_client
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,Which planet is red?,Venus,Mars,B,science\n"
    )

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "uploaded-science",
            "file_format": "csv",
            "split": "test",
        },
        files={
            "file": (
                "science.csv",
                content,
                "text/csv",
            ),
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["name"] == "uploaded-science"
    assert payload["source_type"] == "uploaded"
    assert payload["source_uri"].startswith(
        "upload://sha256/"
    )
    assert payload["source_uri"].endswith(".csv")
    assert payload["revision"] is None
    assert payload["split"] == "test"
    assert payload["task_type"] == "multiple_choice"
    assert (
        payload["adapter_type"]
        == "tabular_mcq_csv_v1"
    )
    assert payload["checksum"].startswith("sha256:")
    assert payload["metadata"]["format"] == "csv"
    assert payload["metadata"]["sample_count"] == 1
    assert payload["metadata"]["categories"] == [
        "science"
    ]
    assert payload["metadata"]["size_bytes"] == len(
        content
    )
    assert payload["is_active"] is True

    storage = LocalDatasetStorage(storage_root)
    stored_path = storage.resolve(
        payload["source_uri"]
    )

    assert stored_path.read_bytes() == content
    assert stored_path.is_relative_to(storage_root)

    retrieved = client.get(
        f"/api/v1/datasets/{payload['id']}"
    )

    assert retrieved.status_code == 200
    assert retrieved.json() == payload

    listed = client.get("/api/v1/datasets")

    assert listed.status_code == 200
    assert [
        item["id"]
        for item in listed.json()
    ] == [payload["id"]]

    serialized = response.text.lower()

    assert str(tmp_path).lower() not in serialized
    assert "c:\\users\\" not in serialized


def test_upload_jsonl_registers_and_stores_dataset(
    upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client
    content = (
        b'{"sample_id":"002",'
        b'"question":"Choose the correct option",'
        b'"options":["Alpha","Beta"],'
        b'"correct_answer":"B",'
        b'"category":"logic"}\n'
    )

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "uploaded-jsonl",
            "file_format": "jsonl",
            "split": "validation",
            "revision": "revision-1",
            "license": "CC0-1.0",
        },
        files={
            "file": (
                "questions.jsonl",
                content,
                "application/x-ndjson",
            ),
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["name"] == "uploaded-jsonl"
    assert payload["source_type"] == "uploaded"
    assert payload["source_uri"].startswith(
        "upload://sha256/"
    )
    assert payload["source_uri"].endswith(".jsonl")
    assert payload["revision"] == "revision-1"
    assert payload["split"] == "validation"
    assert payload["license"] == "CC0-1.0"
    assert (
        payload["adapter_type"]
        == "tabular_mcq_jsonl_v1"
    )
    assert payload["metadata"]["format"] == "jsonl"
    assert payload["metadata"]["sample_count"] == 1
    assert payload["metadata"]["categories"] == [
        "logic"
    ]

    storage = LocalDatasetStorage(storage_root)

    assert (
        storage.resolve(
            payload["source_uri"]
        ).read_bytes()
        == content
    )


def test_invalid_csv_creates_no_registration_or_file(
    upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client
    invalid_content = b"invalid,columns\nvalue,data\n"

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "invalid-csv",
            "file_format": "csv",
            "split": "test",
        },
        files={
            "file": (
                "invalid.csv",
                invalid_content,
                "text/csv",
            ),
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded dataset is invalid",
        "code": "invalid_uploaded_dataset",
    }

    datasets = client.get("/api/v1/datasets")

    assert datasets.status_code == 200
    assert datasets.json() == []

    if storage_root.exists():
        assert [
            path
            for path in storage_root.rglob("*")
            if path.is_file()
        ] == []

    serialized = response.text.lower()

    assert "invalid,columns" not in serialized
    assert "value,data" not in serialized
    assert str(storage_root).lower() not in serialized


def test_empty_csv_creates_no_registration_or_file(
    upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
    )

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "empty-csv",
            "file_format": "csv",
        },
        files={
            "file": (
                "empty.csv",
                content,
                "text/csv",
            ),
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded dataset is invalid",
        "code": "invalid_uploaded_dataset",
    }

    assert client.get(
        "/api/v1/datasets"
    ).json() == []

    if storage_root.exists():
        assert [
            path
            for path in storage_root.rglob("*")
            if path.is_file()
        ] == []


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_payload"),
    [
        (
            DatasetFileTooLargeError,
            413,
            {
                "detail": "Dataset exceeds the upload size limit",
                "code": "dataset_file_too_large",
            },
        ),
        (
            DatasetStreamError,
            422,
            {
                "detail": "Dataset upload could not be read",
                "code": "dataset_stream_failed",
            },
        ),
        (
            DatasetStorageWriteError,
            500,
            {
                "detail": "Dataset storage failed",
                "code": "dataset_storage_failed",
            },
        ),
        (
            DatasetFinalizationError,
            500,
            {
                "detail": "Dataset storage failed",
                "code": "dataset_storage_failed",
            },
        ),
    ],
)
def test_upload_storage_errors_are_safely_mapped(
    tmp_path: Path,
    error_type: type[Exception],
    expected_status: int,
    expected_payload: dict[str, str],
) -> None:
    sensitive_detail = (
        r"C:\Users\person\dataset.csv /private/data/dataset.jsonl "
        ".dataset-upload-secret.tmp SELECT * FROM datasets "
        "Traceback api_key=abc123 Authorization: Bearer token-value "
        "password=secret-value uploaded-question-content"
    )

    class FailingIngestionService:
        def ingest(self, *_: object) -> None:
            raise error_type(sensitive_detail)

    storage_root = tmp_path / "datasets"
    app = create_app(dataset_storage_root=storage_root)
    app.dependency_overrides[get_dataset_ingestion_service] = (
        lambda: FailingIngestionService()
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/datasets/upload",
            data={
                "name": "error-mapping",
                "file_format": "csv",
            },
            files={
                "file": (
                    "questions.csv",
                    b"uploaded-question-content",
                    "text/csv",
                ),
            },
        )

    assert response.status_code == expected_status
    assert response.json() == expected_payload
    serialized = response.text.lower()
    for forbidden in (
        "c:\\\\users",
        "/private/data",
        ".dataset-upload-secret.tmp",
        "select * from",
        "traceback",
        "abc123",
        "token-value",
        "secret-value",
        "uploaded-question-content",
    ):
        assert forbidden not in serialized
    assert not storage_root.exists()


def test_upload_rejects_unsupported_format_before_ingestion(
    upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "unsupported",
            "file_format": "xlsx",
        },
        files={
            "file": (
                "questions.xlsx",
                b"not-an-excel-file",
                (
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
            ),
        },
    )

    assert response.status_code == 422
    assert client.get(
        "/api/v1/datasets"
    ).json() == []
    assert not storage_root.exists()


@pytest.mark.parametrize(
    "server_controlled_field",
    [
        "source_type",
        "source_uri",
        "task_type",
        "adapter_type",
        "checksum",
        "sample_count",
        "categories",
        "storage_root",
    ],
)
def test_upload_rejects_server_controlled_fields(
    upload_api_client: tuple[TestClient, Path],
    server_controlled_field: str,
) -> None:
    client, storage_root = upload_api_client
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,Question,Alpha,Beta,A,test\n"
    )

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "controlled-fields",
            "file_format": "csv",
            server_controlled_field: "client-value",
        },
        files={
            "file": (
                "questions.csv",
                content,
                "text/csv",
            ),
        },
    )

    assert response.status_code == 422
    assert "client-value" not in response.text
    assert response.json()["detail"] == [
        {
            "loc": ["body", server_controlled_field],
            "msg": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]

    assert client.get(
        "/api/v1/datasets"
    ).json() == []

    assert not storage_root.exists()


def test_upload_rejects_unknown_multipart_field(
    upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "unknown-field",
            "file_format": "csv",
            "unexpected": "submitted-value",
        },
        files={
            "file": (
                "questions.csv",
                b"content",
                "text/csv",
            ),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "loc": ["body", "unexpected"],
            "msg": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]
    assert "submitted-value" not in response.text
    assert not storage_root.exists()


@pytest.mark.parametrize("field_name", ["name", "split"])
def test_upload_rejects_blank_required_text_fields(
    upload_api_client: tuple[TestClient, Path],
    field_name: str,
) -> None:
    client, storage_root = upload_api_client
    data = {
        "name": "blank-field",
        "file_format": "csv",
        "split": "test",
    }
    data[field_name] = "   "

    response = client.post(
        "/api/v1/datasets/upload",
        data=data,
        files={
            "file": (
                "questions.csv",
                b"content",
                "text/csv",
            ),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == [
        "body",
        field_name,
    ]
    assert not storage_root.exists()


def test_upload_openapi_and_factory_have_no_storage_side_effects(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "datasets"

    app = create_app(
        dataset_storage_root=storage_root,
    )

    document = app.openapi()

    assert (
        "/api/v1/datasets/upload"
        in document["paths"]
    )
    request_schema = document["paths"][
        "/api/v1/datasets/upload"
    ]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    schema_name = request_schema["$ref"].rsplit("/", 1)[-1]
    properties = document["components"]["schemas"][
        schema_name
    ]["properties"]

    assert set(properties) == {
        "file",
        "name",
        "file_format",
        "split",
        "revision",
        "license",
    }
    assert "form" not in properties
    assert not storage_root.exists()


def test_empty_jsonl_creates_no_registration_or_file(
    upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client

    response = client.post(
        "/api/v1/datasets/upload",
        data={
            "name": "empty-jsonl",
            "file_format": "jsonl",
        },
        files={
            "file": (
                "empty.jsonl",
                b"",
                "application/x-ndjson",
            ),
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded dataset is invalid",
        "code": "invalid_uploaded_dataset",
    }

    assert client.get(
        "/api/v1/datasets"
    ).json() == []

    if storage_root.exists():
        assert [
            path
            for path in storage_root.rglob("*")
            if path.is_file()
        ] == []


def test_duplicate_upload_preserves_reused_file(
        upload_api_client: tuple[TestClient, Path],
) -> None:
    client, storage_root = upload_api_client
    content = (
        b"sample_id,question,option_A,option_B,"
        b"correct_answer,category\n"
        b"001,Question,Alpha,Beta,A,test\n"
    )
    request_data = {
        "name": "duplicate-upload",
        "file_format": "csv",
        "split": "test",
    }
    request_files = {
        "file": (
            "questions.csv",
            content,
            "text/csv",
        ),
    }

    first = client.post(
        "/api/v1/datasets/upload",
        data=request_data,
        files=request_files,
    )

    assert first.status_code == 201

    second = client.post(
        "/api/v1/datasets/upload",
        data=request_data,
        files=request_files,
    )

    assert second.status_code == 409
    assert second.json() == {
        "detail": "Resource already exists"
    }

    source_uri = first.json()["source_uri"]
    stored_path = LocalDatasetStorage(
        storage_root
    ).resolve(source_uri)

    assert stored_path.read_bytes() == content

    registered = client.get(
        "/api/v1/datasets"
    ).json()

    assert len(registered) == 1

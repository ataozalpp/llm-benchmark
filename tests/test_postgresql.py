from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from llm_benchmark.api import create_app
from llm_benchmark.db import DATABASE_URL_ENV, create_db_engine
from llm_benchmark.db.errors import UniquenessConflictError
from llm_benchmark.db.records import SampleResultCreate
from llm_benchmark.db.registry import create_registry_repositories
from llm_benchmark.worker import create_worker

TEST_POSTGRES_URL_ENV = "LLM_BENCHMARK_TEST_POSTGRES_URL"
EXPECTED_TABLES = {
    "provider_endpoints",
    "models",
    "datasets",
    "benchmark_runs",
    "sample_results",
}
EXPECTED_REVISION = "20260817_0001"
APPLICATION_TABLES_IN_DELETE_ORDER = (
    "sample_results",
    "benchmark_runs",
    "models",
    "datasets",
    "provider_endpoints",
)


def _validate_postgres_test_url(database_url: str) -> str:
    parsed = make_url(database_url)
    if not parsed.drivername.startswith("postgresql"):
        raise ValueError("PostgreSQL integration tests require a PostgreSQL URL")
    if parsed.database is None or not parsed.database.endswith("_test"):
        raise ValueError("PostgreSQL integration tests require a database ending in _test")
    return database_url


def _configured_postgres_test_url() -> str:
    database_url = os.getenv(TEST_POSTGRES_URL_ENV)
    if not database_url:
        pytest.skip(f"{TEST_POSTGRES_URL_ENV} is not configured")
    return _validate_postgres_test_url(database_url)


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _table_names(database_url: str) -> set[str]:
    engine = create_db_engine(database_url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _truncate_application_tables(database_url: str) -> None:
    _validate_postgres_test_url(database_url)
    engine = create_db_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE "
                    + ", ".join(APPLICATION_TABLES_IN_DELETE_ORDER)
                    + " RESTART IDENTITY CASCADE"
                )
            )
    finally:
        engine.dispose()


@pytest.fixture
def postgres_database_url(monkeypatch: pytest.MonkeyPatch) -> str:
    database_url = _configured_postgres_test_url()
    # Alembic intentionally reads the runtime variable, but the destructive
    # target is sourced exclusively from the separately validated test URL.
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    command.upgrade(_alembic_config(database_url), "head")
    _truncate_application_tables(database_url)
    yield database_url
    _truncate_application_tables(database_url)


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite:///runtime/test.db",
        "postgresql+psycopg://user:development@127.0.0.1:5432/llm_benchmark",
        "postgresql+psycopg://user:development@127.0.0.1:5432/production",
    ],
)
def test_postgres_validation_rejects_unsafe_database_urls(database_url: str) -> None:
    with pytest.raises(ValueError):
        _validate_postgres_test_url(database_url)


def test_postgres_validation_accepts_explicit_test_database() -> None:
    database_url = "postgresql+psycopg://user:development@127.0.0.1:5432/llm_benchmark_test"
    assert _validate_postgres_test_url(database_url) == database_url


def test_postgres_tests_never_fall_back_to_runtime_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TEST_POSTGRES_URL_ENV, raising=False)
    monkeypatch.setenv(
        DATABASE_URL_ENV,
        "postgresql+psycopg://user:development@127.0.0.1:5432/llm_benchmark",
    )
    with pytest.raises(pytest.skip.Exception):
        _configured_postgres_test_url()


@pytest.mark.postgres
def test_postgresql_migration_upgrade_downgrade_reupgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _configured_postgres_test_url()
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    config = _alembic_config(database_url)
    try:
        command.downgrade(config, "base")
        command.upgrade(config, "head")
        assert EXPECTED_TABLES <= _table_names(database_url)
        engine = create_db_engine(database_url)
        try:
            with engine.connect() as connection:
                revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        finally:
            engine.dispose()
        assert revision == EXPECTED_REVISION

        command.downgrade(config, "base")
        assert EXPECTED_TABLES.isdisjoint(_table_names(database_url))

        command.upgrade(config, "head")
        assert EXPECTED_TABLES <= _table_names(database_url)
    finally:
        command.upgrade(config, "head")


@pytest.mark.postgres
def test_postgresql_schema_constraints(postgres_database_url: str) -> None:
    engine = create_db_engine(postgres_database_url)
    try:
        inspector = inspect(engine)
        assert EXPECTED_TABLES <= set(inspector.get_table_names())
        expected_foreign_keys = {
            ("models", "endpoint_id", "provider_endpoints"),
            ("benchmark_runs", "model_id", "models"),
            ("benchmark_runs", "dataset_id", "datasets"),
            ("sample_results", "run_id", "benchmark_runs"),
        }
        actual_foreign_keys = {
            (table, fk["constrained_columns"][0], fk["referred_table"])
            for table in EXPECTED_TABLES
            for fk in inspector.get_foreign_keys(table)
        }
        assert expected_foreign_keys <= actual_foreign_keys

        unique_constraints = {
            (table, tuple(constraint["column_names"]))
            for table in EXPECTED_TABLES
            for constraint in inspector.get_unique_constraints(table)
        }
        assert ("provider_endpoints", ("name",)) in unique_constraints
        assert ("models", ("endpoint_id", "model_identifier")) in unique_constraints
        assert ("sample_results", ("run_id", "sample_id")) in unique_constraints

        check_names = {
            constraint["name"]
            for table in ("provider_endpoints", "models", "benchmark_runs")
            for constraint in inspector.get_check_constraints(table)
        }
        assert any(name.endswith("_provider_type") for name in check_names)
        assert any(name.endswith("_reasoning_policy") for name in check_names)
        assert any(name.endswith("_run_status") for name in check_names)

        sample_columns = {column["name"]: column for column in inspector.get_columns("sample_results")}
        assert sample_columns["ttft_ms"]["nullable"] is True
        assert sample_columns["throughput_tokens_per_second"]["nullable"] is True

        all_columns = {
            column["name"]
            for table in EXPECTED_TABLES
            for column in inspector.get_columns(table)
        }
        assert "credential_env_var" in all_columns
        assert {"api_key", "password", "secret", "bearer_token"}.isdisjoint(all_columns)
    finally:
        engine.dispose()


def _sample(sample_id: str, *, ttft_ms: float | None = None) -> SampleResultCreate:
    return SampleResultCreate(
        sample_id=sample_id,
        category="logic",
        correct_answer="B",
        parsed_answer="B",
        raw_response="B",
        request_status="succeeded",
        parse_status="normalized_label",
        evaluation_status="correct",
        is_correct=True,
        input_tokens=10,
        output_tokens=1,
        reasoning_tokens=0,
        total_tokens=11,
        latency_ms=2.5,
        ttft_ms=ttft_ms,
        throughput_tokens_per_second=None,
    )


@pytest.mark.postgres
def test_postgresql_repository_crud_json_history_and_samples(postgres_database_url: str) -> None:
    registry = create_registry_repositories(postgres_database_url)
    suffix = uuid4().hex
    endpoint = registry.endpoints.create(
        name=f"endpoint-{suffix}", provider_type="mock", base_url="mock://provider"
    )
    assert registry.endpoints.get_by_id(endpoint.id) == endpoint
    updated_endpoint = registry.endpoints.update(endpoint.id, base_url="mock://updated")
    assert updated_endpoint.base_url == "mock://updated"

    model = registry.models.create(
        name="PostgreSQL Mock",
        model_identifier=f"mock-{suffix}",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
        capabilities={"reasoning_output": False},
        default_generation_config={"scenario_cycle": ["correct"]},
        metadata={"runtime": "postgres-test"},
    )
    dataset = registry.datasets.create(
        name=f"dataset-{suffix}",
        source_type="local",
        source_uri=f"fixture://{suffix}",
        revision=None,
        split="test",
        task_type="multiple_choice",
        adapter_type="local_jsonl",
        metadata={"categories": ["logic"]},
    )
    assert registry.datasets.get_by_id(dataset.id).revision is None
    assert registry.models.get_by_id(model.id).metadata_json == {"runtime": "postgres-test"}

    run = registry.runs.create_queued(
        experiment_name="postgres-repository",
        model_id=model.id,
        dataset_id=dataset.id,
        resolved_config={"schema_version": 1, "test": {"nested": True}},
        config_hash="a" * 64,
        seed=42,
        sample_count=2,
        artifact_directory="outputs/test",
    )
    running = registry.runs.claim_next_queued()
    assert running is not None
    assert running.id == run.id
    assert running.started_at is not None
    persisted_samples = registry.samples.add_many(
        run.id, [_sample("one"), _sample("two", ttft_ms=1.25)]
    )
    assert persisted_samples[0].ttft_ms is None
    assert persisted_samples[0].throughput_tokens_per_second is None
    assert persisted_samples[1].ttft_ms == 1.25
    completed = registry.runs.transition_status(
        run.id,
        "completed",
        summary={"overall": {"accuracy": 1.0}},
        artifact_directory="outputs/test/completed",
    )
    assert completed.completed_at is not None
    assert completed.summary_json == {"overall": {"accuracy": 1.0}}
    assert completed.artifact_directory == "outputs/test/completed"
    assert completed.resolved_config_json["test"]["nested"] is True

    registry.endpoints.soft_delete(endpoint.id)
    registry.models.soft_delete(model.id)
    registry.datasets.soft_delete(dataset.id)
    assert registry.endpoints.list_active() == []
    assert registry.models.list_active() == []
    assert registry.datasets.list_active() == []
    assert registry.runs.get_by_id(run.id).status.value == "completed"
    assert len(registry.samples.list_by_run_id(run.id)) == 2


@pytest.mark.postgres
def test_postgresql_repository_conflicts_foreign_keys_and_bulk_rollback(
    postgres_database_url: str,
) -> None:
    registry = create_registry_repositories(postgres_database_url)
    endpoint = registry.endpoints.create(
        name="unique-endpoint", provider_type="mock", base_url="mock://provider"
    )
    with pytest.raises(UniquenessConflictError):
        registry.endpoints.create(
            name="unique-endpoint", provider_type="mock", base_url="mock://other"
        )

    engine = create_db_engine(postgres_database_url)
    try:
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO models "
                    "(name, model_identifier, endpoint_id, reasoning_policy, "
                    "capabilities_json, default_generation_config_json, metadata_json, is_active) "
                    "VALUES ('orphan', 'orphan', 999999, 'unsupported', '{}', '{}', '{}', true)"
                )
            )
    finally:
        engine.dispose()

    model = registry.models.create(
        name="Mock",
        model_identifier="mock",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
    )
    dataset = registry.datasets.create(
        name="Dataset",
        source_type="local",
        source_uri="fixture://rollback",
        revision=None,
        split="test",
        task_type="multiple_choice",
        adapter_type="local_jsonl",
    )
    run = registry.runs.create_queued(
        experiment_name="rollback",
        model_id=model.id,
        dataset_id=dataset.id,
        resolved_config={"schema_version": 1},
        config_hash="b" * 64,
        seed=42,
        sample_count=2,
        artifact_directory="outputs/test",
    )
    with pytest.raises(UniquenessConflictError):
        registry.samples.add_many(run.id, [_sample("duplicate"), _sample("duplicate")])
    assert registry.samples.list_by_run_id(run.id) == []


@pytest.mark.postgres
def test_postgresql_fastapi_uploaded_dataset_mock_run(
    postgres_database_url: str,
    tmp_path: Path,
) -> None:
    registry = create_registry_repositories(postgres_database_url)
    dataset_root = tmp_path / "datasets"
    output_root = tmp_path / "outputs"
    app = create_app(
        registry,
        dataset_storage_root=dataset_root,
        run_output_root=output_root,
    )
    csv_content = (
        b"sample_id,question,option_A,option_B,correct_answer,category\n"
        b"pg-001,PostgreSQL synthetic question,No,Yes,B,synthetic\n"
    )
    with TestClient(app) as client:
        endpoint_response = client.post(
            "/api/v1/endpoints",
            json={"name": "postgres-mock", "provider_type": "mock", "base_url": "mock://provider"},
        )
        assert endpoint_response.status_code == 201
        model_response = client.post(
            "/api/v1/models",
            json={
                "name": "PostgreSQL Mock",
                "model_identifier": "postgres-mock-model",
                "endpoint_id": endpoint_response.json()["id"],
                "reasoning_policy": "unsupported",
                "default_generation_config": {"scenario_cycle": ["correct"]},
            },
        )
        assert model_response.status_code == 201
        upload_response = client.post(
            "/api/v1/datasets/upload",
            data={"name": "postgres-upload", "file_format": "csv", "split": "test"},
            files={"file": ("dataset.csv", BytesIO(csv_content), "text/csv")},
        )
        assert upload_response.status_code == 201
        upload = upload_response.json()
        assert upload["source_uri"].startswith("upload://sha256/")
        assert str(tmp_path) not in upload_response.text

        run_response = client.post(
            "/api/v1/runs",
            json={
                "experiment_name": "postgres-upload-smoke",
                "model_id": model_response.json()["id"],
                "dataset_id": upload["id"],
                "seed": 42,
                "profile": "smoke",
                "sample_size": 1,
                "sample_ids": ["pg-001"],
                "category_filter": [],
            },
        )
        assert run_response.status_code == 201
        public_run = run_response.json()
        assert public_run["status"] == "queued"
        assert public_run["summary"] is None
        assert public_run["started_at"] is None
        assert public_run["completed_at"] is None
        assert public_run["sample_count"] == 1
        assert str(tmp_path) not in run_response.text
        queued_results = client.get(f"/api/v1/runs/{public_run['id']}/results")
        assert queued_results.status_code == 200
        assert queued_results.json() == []

        worker = create_worker(
            registry_factory=lambda: registry,
            dataset_storage_root=dataset_root,
        )
        assert worker.run_once() is True

        completed_response = client.get(f"/api/v1/runs/{public_run['id']}")
        assert completed_response.status_code == 200
        public_run = completed_response.json()
        assert public_run["status"] == "completed"
        assert public_run["summary"] is not None
        assert public_run["started_at"] is not None
        assert public_run["completed_at"] is not None
        results_response = client.get(f"/api/v1/runs/{public_run['id']}/results")
        assert results_response.status_code == 200
        results = results_response.json()
        assert len(results) == 1
        assert results[0]["sample_id"] == "pg-001"
        assert results[0]["ttft_ms"] is None
        assert results[0]["throughput_tokens_per_second"] is None

    stored_dataset = registry.datasets.get_by_id(upload["id"])
    assert stored_dataset.source_uri == upload["source_uri"]
    assert stored_dataset.metadata_json["sample_count"] == 1
    assert "PostgreSQL synthetic question" not in repr(stored_dataset.model_dump())
    persisted_run = registry.runs.get_by_id(public_run["id"])
    assert persisted_run.status.value == "completed"
    assert persisted_run.summary_json is not None
    assert persisted_run.artifact_directory.startswith(str(output_root))
    assert str(dataset_root) not in repr(persisted_run.resolved_config_json)
    artifact_directory = Path(persisted_run.artifact_directory)
    assert (artifact_directory / "results.jsonl").is_file()
    assert (artifact_directory / "summary.json").is_file()
    assert (artifact_directory / "dataset_manifest.json").is_file()
    assert (artifact_directory / "resolved_config.json").is_file()

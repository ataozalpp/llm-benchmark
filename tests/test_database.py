from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from llm_benchmark.db import (
    Base,
    DATABASE_URL_ENV,
    DEFAULT_DATABASE_URL,
    create_db_engine,
    create_session_factory,
    get_database_url,
)
from llm_benchmark.db.models import ProviderEndpoint, RegisteredModel
from llm_benchmark.db.schemas import ModelCapabilities


EXPECTED_TABLES = {
    "provider_endpoints",
    "models",
    "datasets",
    "benchmark_runs",
    "sample_results",
}


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_database_url_uses_runtime_default_and_environment_override() -> None:
    assert get_database_url({}) == DEFAULT_DATABASE_URL
    assert get_database_url({DATABASE_URL_ENV: "sqlite:///custom.db"}) == "sqlite:///custom.db"


def test_database_imports_do_not_create_runtime_files(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop(DATABASE_URL_ENV, None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import llm_benchmark.db; import llm_benchmark.db.models; import llm_benchmark.db.engine",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "runtime").exists()
    assert not list(tmp_path.glob("*.db"))
    assert not list(tmp_path.glob("*.sqlite*"))


def test_engine_session_and_sqlite_foreign_keys(tmp_path: Path) -> None:
    engine = create_db_engine(sqlite_url(tmp_path / "session.sqlite"))
    Base.metadata.create_all(engine)

    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        endpoint = ProviderEndpoint(
            name="local-test",
            provider_type="lm_studio_native",
            base_url="http://127.0.0.1:1234",
        )
        session.add(endpoint)
        session.commit()
        assert endpoint.id is not None

    with session_factory() as session:
        session.add(
            RegisteredModel(
                name="orphan",
                model_identifier="missing-endpoint-model",
                endpoint_id=999_999,
                reasoning_policy="unsupported",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_initial_migration_creates_registry_schema(tmp_path: Path) -> None:
    database_url = sqlite_url(tmp_path / "migration.sqlite")
    config = alembic_config(database_url)

    command.upgrade(config, "head")

    engine = create_db_engine(database_url)
    inspector = inspect(engine)
    assert EXPECTED_TABLES <= set(inspector.get_table_names())

    with engine.connect() as connection:
        revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
    assert revision == "20260817_0001"

def test_alembic_honors_database_url_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = tmp_path / "environment-selected.sqlite"
    database_url = sqlite_url(database_path)

    monkeypatch.setenv(DATABASE_URL_ENV, database_url)

    config = Config("alembic.ini")
    command.upgrade(config, "head")

    assert database_path.exists()

    engine = create_db_engine(database_url)
    try:
        inspector = inspect(engine)
        assert EXPECTED_TABLES <= set(inspector.get_table_names())

        with engine.connect() as connection:
            revision = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
        assert revision == "20260817_0001"
    finally:
        engine.dispose()

def test_initial_migration_supports_upgrade_downgrade_reupgrade(tmp_path: Path) -> None:
    database_url = sqlite_url(tmp_path / "migration-cycle.sqlite")
    config = alembic_config(database_url)

    command.upgrade(config, "head")
    assert EXPECTED_TABLES <= set(inspect(create_db_engine(database_url)).get_table_names())

    command.downgrade(config, "base")
    assert EXPECTED_TABLES.isdisjoint(inspect(create_db_engine(database_url)).get_table_names())

    command.upgrade(config, "head")
    assert EXPECTED_TABLES <= set(inspect(create_db_engine(database_url)).get_table_names())


def test_migration_has_expected_foreign_keys_and_uniqueness(tmp_path: Path) -> None:
    database_url = sqlite_url(tmp_path / "constraints.sqlite")
    config = alembic_config(database_url)
    command.upgrade(config, "head")

    inspector = inspect(create_db_engine(database_url))

    model_foreign_keys = inspector.get_foreign_keys("models")
    assert {(item["referred_table"], tuple(item["constrained_columns"])) for item in model_foreign_keys} == {
        ("provider_endpoints", ("endpoint_id",))
    }

    run_foreign_keys = inspector.get_foreign_keys("benchmark_runs")
    assert {(item["referred_table"], tuple(item["constrained_columns"])) for item in run_foreign_keys} == {
        ("datasets", ("dataset_id",)),
        ("models", ("model_id",)),
    }

    result_foreign_keys = inspector.get_foreign_keys("sample_results")
    assert {(item["referred_table"], tuple(item["constrained_columns"])) for item in result_foreign_keys} == {
        ("benchmark_runs", ("run_id",))
    }

    endpoint_uniques = {
        tuple(item["column_names"]) for item in inspector.get_unique_constraints("provider_endpoints")
    }
    model_uniques = {
        tuple(item["column_names"]) for item in inspector.get_unique_constraints("models")
    }
    result_uniques = {
        tuple(item["column_names"]) for item in inspector.get_unique_constraints("sample_results")
    }
    assert ("name",) in endpoint_uniques
    assert ("endpoint_id", "model_identifier") in model_uniques
    assert ("run_id", "sample_id") in result_uniques

    result_columns = {item["name"]: item for item in inspector.get_columns("sample_results")}
    assert result_columns["ttft_ms"]["nullable"] is True
    assert result_columns["throughput_tokens_per_second"]["nullable"] is True


def test_historical_foreign_keys_do_not_cascade(tmp_path: Path) -> None:
    database_url = sqlite_url(tmp_path / "ondelete.sqlite")
    config = alembic_config(database_url)
    command.upgrade(config, "head")
    inspector = inspect(create_db_engine(database_url))

    assert inspector.get_foreign_keys("models")[0]["options"].get("ondelete") is None
    for foreign_key in inspector.get_foreign_keys("benchmark_runs"):
        assert foreign_key["options"].get("ondelete") is None
    assert inspector.get_foreign_keys("sample_results")[0]["options"].get("ondelete") == "CASCADE"


def test_registry_schema_has_no_secret_value_columns_and_uses_portable_enums(tmp_path: Path) -> None:
    forbidden_columns = {"api_key", "bearer_token", "password", "secret", "secret_value"}
    orm_columns = {column.name for table in Base.metadata.tables.values() for column in table.columns}
    assert forbidden_columns.isdisjoint(orm_columns)
    assert "credential_env_var" in orm_columns

    enum_columns = [
        column
        for table in Base.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, SqlEnum)
    ]
    assert enum_columns
    assert all(column.type.native_enum is False for column in enum_columns)
    assert all(column.type.create_constraint is True for column in enum_columns)

    database_url = sqlite_url(tmp_path / "schema-security.sqlite")
    config = alembic_config(database_url)
    command.upgrade(config, "head")
    inspector = inspect(create_db_engine(database_url))
    migration_columns = {
        column["name"]
        for table_name in EXPECTED_TABLES
        for column in inspector.get_columns(table_name)
    }
    assert forbidden_columns.isdisjoint(migration_columns)
    assert "credential_env_var" in migration_columns

    check_constraints = {
        constraint["name"]
        for table_name in ("provider_endpoints", "models", "benchmark_runs")
        for constraint in inspector.get_check_constraints(table_name)
    }
    assert any(name.endswith("_provider_type") for name in check_constraints)
    assert any(name.endswith("_reasoning_policy") for name in check_constraints)
    assert any(name.endswith("_run_status") for name in check_constraints)


def test_model_capabilities_validate_known_shape() -> None:
    capabilities = ModelCapabilities(reasoning_output=True, streaming=False)
    assert capabilities.reasoning_output is True
    assert capabilities.streaming is False

    with pytest.raises(ValidationError):
        ModelCapabilities(unknown_capability=True)

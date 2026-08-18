from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest

from llm_benchmark.api import create_app
from llm_benchmark.db.errors import RepositoryError
from llm_benchmark.db.registry import create_registry_repositories


@pytest.fixture
def api_client(tmp_path: Path) -> TestClient:
    database_url = f"sqlite:///{(tmp_path / 'api.sqlite').as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")
    app = create_app(create_registry_repositories(database_url))
    with TestClient(app) as client:
        yield client


def endpoint_payload(name: str = "openai-local") -> dict[str, Any]:
    return {
        "name": name,
        "provider_type": "openai_compatible",
        "base_url": "http://127.0.0.1:1234/v1",
        "credential_env_var": "LOCAL_LLM_API_KEY",
    }


def model_payload(endpoint_id: int, identifier: str = "registered-model") -> dict[str, Any]:
    return {
        "name": "Registered model",
        "model_identifier": identifier,
        "endpoint_id": endpoint_id,
        "reasoning_policy": "toggle",
        "capabilities": {"reasoning_output": True, "streaming": False},
        "default_generation_config": {"temperature": 0},
        "metadata": {"family": "test"},
    }


def dataset_payload(name: str = "fixture") -> dict[str, Any]:
    return {
        "name": name,
        "source_type": "local",
        "source_uri": "data/fixtures/mcq_fixture.jsonl",
        "revision": None,
        "split": "test",
        "task_type": "multiple_choice",
        "adapter_type": "local_jsonl",
        "license": "CC0-1.0",
        "checksum": None,
        "metadata": {"purpose": "test"},
    }


def test_api_import_factory_and_openapi_have_no_database_side_effects(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("LLM_BENCHMARK_DATABASE_URL", None)
    code = (
        "from llm_benchmark.api import create_app; "
        "app = create_app(); "
        "assert app.openapi()['info']['title'] == 'LLM Benchmark Registry API'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
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


def test_openapi_has_registry_routes_and_no_secret_fields_or_local_paths() -> None:
    document = create_app().openapi()
    expected_paths = {
        "/api/v1/endpoints",
        "/api/v1/endpoints/{endpoint_id}",
        "/api/v1/models",
        "/api/v1/models/{model_id}",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}",
    }
    assert expected_paths <= set(document["paths"])
    serialized = json.dumps(document).lower()
    assert "c:\\users\\" not in serialized

    property_names = {
        property_name
        for schema in document["components"]["schemas"].values()
        for property_name in schema.get("properties", {})
    }
    assert {"api_key", "secret", "password", "bearer_token"}.isdisjoint(property_names)


def test_endpoint_crud_active_list_history_and_duplicate(api_client: TestClient) -> None:
    created = api_client.post("/api/v1/endpoints", json=endpoint_payload())
    assert created.status_code == 201
    endpoint = created.json()
    assert endpoint["provider_type"] == "openai_compatible"
    assert endpoint["credential_env_var"] == "LOCAL_LLM_API_KEY"

    assert api_client.get(f"/api/v1/endpoints/{endpoint['id']}").json() == endpoint
    assert [item["id"] for item in api_client.get("/api/v1/endpoints").json()] == [endpoint["id"]]

    updated = api_client.patch(
        f"/api/v1/endpoints/{endpoint['id']}",
        json={"name": "openai-updated", "credential_env_var": None},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "openai-updated"
    assert updated.json()["credential_env_var"] is None

    duplicate = api_client.post("/api/v1/endpoints", json={**endpoint_payload(), "name": "openai-updated"})
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Resource already exists"}

    deleted = api_client.delete(f"/api/v1/endpoints/{endpoint['id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert api_client.get("/api/v1/endpoints").json() == []
    historical = api_client.get(f"/api/v1/endpoints/{endpoint['id']}")
    assert historical.status_code == 200
    assert historical.json()["is_active"] is False


def test_model_crud_capabilities_active_list_and_duplicate(api_client: TestClient) -> None:
    endpoint_id = api_client.post("/api/v1/endpoints", json=endpoint_payload()).json()["id"]
    created = api_client.post("/api/v1/models", json=model_payload(endpoint_id))
    assert created.status_code == 201
    model = created.json()
    assert model["capabilities"] == {"reasoning_output": True, "streaming": False}
    assert api_client.get(f"/api/v1/models/{model['id']}").json() == model
    assert len(api_client.get("/api/v1/models").json()) == 1

    updated = api_client.patch(
        f"/api/v1/models/{model['id']}",
        json={
            "reasoning_policy": "provider_managed",
            "capabilities": {"tool_calling": True},
            "default_generation_config": {"temperature": 0.2},
            "metadata": {"family": "updated"},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["reasoning_policy"] == "provider_managed"
    assert updated.json()["capabilities"] == {"tool_calling": True}

    duplicate = api_client.post("/api/v1/models", json=model_payload(endpoint_id))
    assert duplicate.status_code == 409

    assert api_client.delete(f"/api/v1/models/{model['id']}").status_code == 204
    assert api_client.get("/api/v1/models").json() == []
    assert api_client.get(f"/api/v1/models/{model['id']}").json()["is_active"] is False


def test_dataset_crud_nullable_revision_active_list_and_duplicate(api_client: TestClient) -> None:
    created = api_client.post("/api/v1/datasets", json=dataset_payload())
    assert created.status_code == 201
    dataset = created.json()
    assert dataset["revision"] is None
    assert api_client.get(f"/api/v1/datasets/{dataset['id']}").json() == dataset
    assert len(api_client.get("/api/v1/datasets").json()) == 1

    updated = api_client.patch(
        f"/api/v1/datasets/{dataset['id']}",
        json={"name": "fixture-updated", "license": None, "metadata": {"purpose": "updated"}},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "fixture-updated"
    assert updated.json()["license"] is None

    duplicate = api_client.post("/api/v1/datasets", json=dataset_payload("another-display-name"))
    assert duplicate.status_code == 409

    assert api_client.delete(f"/api/v1/datasets/{dataset['id']}").status_code == 204
    assert api_client.get("/api/v1/datasets").json() == []
    assert api_client.get(f"/api/v1/datasets/{dataset['id']}").json()["is_active"] is False


def test_missing_records_and_inactive_endpoint_model_dependency(api_client: TestClient) -> None:
    for resource in ("endpoints", "models", "datasets"):
        response = api_client.get(f"/api/v1/{resource}/999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    endpoint_id = api_client.post("/api/v1/endpoints", json=endpoint_payload()).json()["id"]
    assert api_client.delete(f"/api/v1/endpoints/{endpoint_id}").status_code == 204
    rejected = api_client.post("/api/v1/models", json=model_payload(endpoint_id))
    assert rejected.status_code == 409
    assert rejected.json()["detail"] == "provider endpoint is inactive"


@pytest.mark.parametrize("resource", ["endpoints", "models", "datasets"])
def test_empty_patch_is_rejected(api_client: TestClient, resource: str) -> None:
    assert api_client.patch(f"/api/v1/{resource}/1", json={}).status_code == 422


@pytest.mark.parametrize("field", ["api_key", "secret", "password", "bearer_token", "bearer-token"])
def test_endpoint_secret_value_fields_are_rejected(api_client: TestClient, field: str) -> None:
    response = api_client.post("/api/v1/endpoints", json={**endpoint_payload(), field: "sensitive-value"})
    assert response.status_code == 422
    assert "sensitive-value" not in response.text


def test_unknown_fields_and_invalid_capabilities_are_rejected(api_client: TestClient) -> None:
    assert api_client.post(
        "/api/v1/endpoints", json={**endpoint_payload(), "unknown_field": True}
    ).status_code == 422
    endpoint_id = api_client.post("/api/v1/endpoints", json=endpoint_payload("capability-endpoint")).json()["id"]
    invalid = model_payload(endpoint_id)
    invalid["capabilities"] = {"unknown_capability": True}
    assert api_client.post("/api/v1/models", json=invalid).status_code == 422


@pytest.mark.parametrize("sensitive_key", ["api_key", "password", "secret", "authorization", "token"])
def test_nested_metadata_secrets_are_rejected(api_client: TestClient, sensitive_key: str) -> None:
    payload = dataset_payload()
    payload["metadata"] = {"nested": {sensitive_key: "sensitive-value"}}
    response = api_client.post("/api/v1/datasets", json=payload)
    assert response.status_code == 422
    assert "sensitive-value" not in response.text


def test_internal_errors_are_sanitized() -> None:
    class FailingEndpointRepository:
        def list_active(self) -> list[Any]:
            raise RepositoryError(
                "SELECT password FROM secrets at C:\\Users\\private\\database.sqlite api_key=secret"
            )

    registry = SimpleNamespace(endpoints=FailingEndpointRepository(), models=None, datasets=None)
    client = TestClient(create_app(registry), raise_server_exceptions=False)
    response = client.get("/api/v1/endpoints")
    assert response.status_code == 500
    assert response.json() == {"detail": "Registry persistence failed"}
    serialized = response.text.lower()
    assert "select" not in serialized
    assert "password" not in serialized
    assert "c:\\users" not in serialized
    assert "secret" not in serialized


def test_routes_do_not_import_sqlalchemy_or_orm_models() -> None:
    route_source = Path("src/llm_benchmark/api/routes.py").read_text(encoding="utf-8").lower()
    assert "sqlalchemy" not in route_source
    assert "db.models" not in route_source

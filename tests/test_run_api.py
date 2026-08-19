from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest

from llm_benchmark.api import create_app
from llm_benchmark.config import RunConfig
from llm_benchmark.db.registry import create_registry_repositories
from llm_benchmark.reproducibility import canonical_hash
from llm_benchmark.runner import PipelineExecution


@pytest.fixture
def run_api(tmp_path: Path) -> dict[str, Any]:
    database_url = f"sqlite:///{(tmp_path / 'run-api.sqlite').as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")
    registry = create_registry_repositories(database_url)
    output_root = tmp_path / "artifacts"
    app = create_app(registry, run_output_root=output_root)
    with TestClient(app) as client:
        yield {"client": client, "registry": registry, "output_root": output_root, "tmp_path": tmp_path}


def register_fixture(context: dict[str, Any], *, scenario: str = "correct") -> tuple[int, int, int]:
    registry = context["registry"]
    endpoint = registry.endpoints.create(
        name=f"mock-endpoint-{scenario}",
        provider_type="mock",
        base_url="mock://provider",
    )
    model = registry.models.create(
        name=f"Mock model {scenario}",
        model_identifier=f"mock-model-{scenario}",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
        capabilities={},
        default_generation_config={"scenario_cycle": [scenario], "mock_latency_ms": 3.5},
    )
    dataset = registry.datasets.create(
        name="synthetic-mcq-fixture",
        source_type="local",
        source_uri="data/fixtures/mcq_fixture.jsonl",
        revision=None,
        split="test",
        task_type="multiple_choice",
        adapter_type="local_jsonl",
        license="project-owned synthetic fixture",
    )
    return endpoint.id, model.id, dataset.id


def run_payload(model_id: int, dataset_id: int) -> dict[str, Any]:
    return {
        "experiment_name": "api-fixture-smoke",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "seed": 42,
        "profile": "smoke",
        "sample_size": 1,
        "sample_ids": ["q01"],
        "category_filter": [],
    }


def test_run_routes_appear_in_openapi_without_server_controlled_request_fields() -> None:
    document = create_app().openapi()
    assert {
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/results",
    } <= set(document["paths"])
    properties = document["components"]["schemas"]["RunCreate"]["properties"]
    forbidden = {
        "endpoint_id",
        "provider",
        "base_url",
        "credential_env_var",
        "api_key",
        "output_dir",
        "artifact_directory",
        "model_identifier",
        "dataset_path",
    }
    assert forbidden.isdisjoint(properties)
    assert "error_message" not in document["components"]["schemas"]["RunResponse"]["properties"]


def test_successful_run_creation_listing_detail_and_results(run_api: dict[str, Any]) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    client = run_api["client"]
    created = client.post("/api/v1/runs", json=run_payload(model_id, dataset_id))
    assert created.status_code == 201
    run = created.json()
    assert run["status"] == "completed"
    assert run["sample_count"] == 1
    assert run["summary"]["overall"]["correct_count"] == 1
    assert "artifact_directory" not in run
    assert "resolved_config" not in run

    assert client.get("/api/v1/runs").json() == [run]
    assert client.get(f"/api/v1/runs/{run['id']}").json() == run
    results = client.get(f"/api/v1/runs/{run['id']}/results").json()
    assert len(results) == 1
    assert results[0]["sample_id"] == "q01"
    assert results[0]["evaluation_status"] == "correct"
    assert results[0]["ttft_ms"] is None
    assert results[0]["throughput_tokens_per_second"] is None

    persisted = run_api["registry"].runs.get_by_id(run["id"])
    assert persisted.status.value == "completed"
    assert persisted.started_at is not None
    assert persisted.completed_at is not None
    assert persisted.config_hash == canonical_hash(persisted.resolved_config_json)
    resolved = RunConfig.model_validate(persisted.resolved_config_json)
    assert resolved.output_dir == run_api["output_root"]
    assert resolved.models[0].provider == "mock"
    assert resolved.models[0].model_id == "mock-model-correct"
    assert resolved.dataset.path == Path("data/fixtures/mcq_fixture.jsonl")
    artifact_directory = Path(persisted.artifact_directory)
    assert artifact_directory.parent == run_api["output_root"]
    assert (artifact_directory / "results.jsonl").is_file()
    assert (artifact_directory / "summary.json").is_file()


@pytest.mark.parametrize(
    "field",
    [
        "endpoint_id",
        "provider",
        "base_url",
        "credential_env_var",
        "api_key",
        "output_dir",
        "artifact_directory",
        "model_identifier",
        "dataset_path",
    ],
)
def test_server_controlled_run_fields_are_rejected(run_api: dict[str, Any], field: str) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    payload = {**run_payload(model_id, dataset_id), field: "client-controlled-value"}
    response = run_api["client"].post("/api/v1/runs", json=payload)
    assert response.status_code == 422
    assert "client-controlled-value" not in response.text
    assert run_api["registry"].runs.list_runs() == []


@pytest.mark.parametrize("scenario", ["unparseable", "request_failed"])
def test_sample_outcomes_still_complete_the_run(run_api: dict[str, Any], scenario: str) -> None:
    _, model_id, dataset_id = register_fixture(run_api, scenario=scenario)
    response = run_api["client"].post("/api/v1/runs", json=run_payload(model_id, dataset_id))
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "completed"
    result = run_api["client"].get(f"/api/v1/runs/{run['id']}/results").json()[0]
    assert result["evaluation_status"] == scenario


@pytest.mark.parametrize("dependency", ["endpoint", "model", "dataset"])
def test_inactive_dependencies_are_rejected_without_run_creation(
    run_api: dict[str, Any], dependency: str
) -> None:
    endpoint_id, model_id, dataset_id = register_fixture(run_api)
    repository = getattr(run_api["registry"], f"{dependency}s")
    repository.soft_delete({"endpoint": endpoint_id, "model": model_id, "dataset": dataset_id}[dependency])
    response = run_api["client"].post("/api/v1/runs", json=run_payload(model_id, dataset_id))
    assert response.status_code == 409
    assert run_api["registry"].runs.list_runs() == []


@pytest.mark.parametrize(("field", "value"), [("model_id", 999999), ("dataset_id", 999999)])
def test_missing_registered_resources_return_404(
    run_api: dict[str, Any], field: str, value: int
) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    payload = run_payload(model_id, dataset_id)
    payload[field] = value
    response = run_api["client"].post("/api/v1/runs", json=payload)
    assert response.status_code == 404
    assert run_api["registry"].runs.list_runs() == []


def test_unsupported_dataset_registration_returns_sanitized_conflict(run_api: dict[str, Any]) -> None:
    registry = run_api["registry"]
    endpoint = registry.endpoints.create(name="unsupported", provider_type="mock", base_url="mock://provider")
    model = registry.models.create(
        name="Mock",
        model_identifier="mock",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
    )
    dataset = registry.datasets.create(
        name="unsupported",
        source_type="local",
        source_uri="private/path.jsonl",
        revision=None,
        split="test",
        task_type="free_text",
        adapter_type="local_jsonl",
    )
    response = run_api["client"].post("/api/v1/runs", json=run_payload(model.id, dataset.id))
    assert response.status_code == 409
    assert response.json() == {"detail": "Registered run configuration is invalid"}
    assert "private/path" not in response.text


@pytest.mark.parametrize(
    "internal_detail",
    [
        r"C:\\Users\\developer\\private\\run.py",
        "/home/developer/private/run.py",
        "SELECT * FROM benchmark_runs WHERE id = 1",
        "Prompt: confidential benchmark question",
        'Traceback (most recent call last): File "runner.py", line 10',
        "api_key=private-api-key-value",
        "Authorization: Bearer private-bearer-value",
        "password=private-password-value",
        "secret=private-secret-value",
    ],
)
def test_pipeline_system_failure_hides_internal_details_from_public_run_response(
    tmp_path: Path, internal_detail: str
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'failure.sqlite').as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")
    registry = create_registry_repositories(database_url)

    def fail(_: RunConfig) -> PipelineExecution:
        raise RuntimeError(internal_detail)

    app = create_app(registry, benchmark_executor=fail, run_output_root=tmp_path / "artifacts")
    context = {"registry": registry}
    _, model_id, dataset_id = register_fixture(context)
    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json=run_payload(model_id, dataset_id))
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "failed"
    assert run["error_type"] == "RuntimeError"
    assert "error_message" not in run
    assert internal_detail not in json.dumps(run)
    persisted = registry.runs.get_by_id(run["id"])
    assert persisted.status.value == "failed"
    assert persisted.error_type == "RuntimeError"
    assert persisted.error_message is not None
    assert registry.samples.list_by_run_id(run["id"]) == []


def test_missing_run_and_results_return_404(run_api: dict[str, Any]) -> None:
    assert run_api["client"].get("/api/v1/runs/999999").status_code == 404
    assert run_api["client"].get("/api/v1/runs/999999/results").status_code == 404


def test_api_package_has_no_sqlalchemy_or_orm_imports() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in Path("src/llm_benchmark/api").glob("*.py")
    )
    assert "sqlalchemy" not in source
    assert "db.models" not in source

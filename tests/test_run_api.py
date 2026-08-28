from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from llm_benchmark.api import create_app
from llm_benchmark.api.app_dependencies import get_run_preflight_service
from llm_benchmark.application import BenchmarkApplicationService
from llm_benchmark.config import RunConfig
from llm_benchmark.dataset_storage import LocalDatasetStorage
from llm_benchmark.db.registry import create_registry_repositories
from llm_benchmark.reproducibility import canonical_hash
from llm_benchmark.run_preflight import RunApiGuardrailPolicy, RunPreflightService
from llm_benchmark.runner import PipelineExecution
from llm_benchmark.worker import BenchmarkWorker


@pytest.fixture
def run_api(tmp_path: Path) -> dict[str, Any]:
    database_path = tmp_path / "run-api.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")
    registry = create_registry_repositories(database_url)
    output_root = tmp_path / "artifacts"
    app = create_app(registry, run_output_root=output_root)
    with TestClient(app) as client:
        yield {
            "client": client,
            "registry": registry,
            "output_root": output_root,
            "tmp_path": tmp_path,
            "database_path": database_path,
        }


def sample_row_count(context: dict[str, Any]) -> int:
    with sqlite3.connect(context["database_path"]) as connection:
        value = connection.execute("SELECT COUNT(*) FROM sample_results").fetchone()
    assert value is not None
    return int(value[0])


def run_one_worker_job(context: dict[str, Any]) -> bool:
    registry = context["registry"]
    app = context["client"].app
    service = BenchmarkApplicationService(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        runs=registry.runs,
        samples=registry.samples,
        executor=app.state.benchmark_executor,
    )
    return BenchmarkWorker(runs=registry.runs, service=service).run_once()


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
    assert run["status"] == "queued"
    assert run["sample_count"] == 1
    assert run["summary"] is None
    assert run["started_at"] is None
    assert run["completed_at"] is None
    assert "artifact_directory" not in run
    assert "resolved_config" not in run
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()

    assert run_one_worker_job(run_api) is True
    run = client.get(f"/api/v1/runs/{run['id']}").json()
    assert run["status"] == "completed"
    assert run["summary"]["overall"]["correct_count"] == 1

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


def test_run_submission_is_queued_without_execution_or_artifacts(
    run_api: dict[str, Any],
) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("submission must not execute the benchmark")

    run_api["client"].app.state.benchmark_executor = forbidden_executor
    response = run_api["client"].post(
        "/api/v1/runs", json=run_payload(model_id, dataset_id)
    )

    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "queued"
    assert run["sample_count"] == 1
    assert run["summary"] is None
    assert run["started_at"] is None
    assert run["completed_at"] is None
    assert executor_calls == 0
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()


@pytest.mark.parametrize(
    ("payload_update", "expected_code"),
    [
        ({"profile": "full"}, "run_guardrail_violation"),
        ({"sample_size": 101, "sample_ids": []}, "run_guardrail_violation"),
        (
            {"sample_size": 101, "sample_ids": [f"private-{index}" for index in range(101)]},
            "run_guardrail_violation",
        ),
        ({"sample_ids": ["private-missing-id"]}, "invalid_dataset_selection"),
        ({"category_filter": ["private-missing-category"]}, "invalid_dataset_selection"),
        ({"category_filter": ["math", "private-missing-category"]}, "invalid_dataset_selection"),
    ],
)
def test_preflight_rejection_has_no_execution_records_or_artifacts(
    run_api: dict[str, Any], payload_update: dict[str, Any], expected_code: str
) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("executor must not be invoked")

    run_api["client"].app.state.benchmark_executor = forbidden_executor
    payload = {**run_payload(model_id, dataset_id), **payload_update}
    if "sample_ids" in payload_update and "sample_size" not in payload_update:
        payload["sample_size"] = len(payload_update["sample_ids"])
    response = run_api["client"].post("/api/v1/runs", json=payload)
    assert response.status_code == 422
    assert response.json()["code"] == expected_code
    response_text = response.text
    assert "private-" not in response_text
    assert "data/fixtures" not in response_text
    assert executor_calls == 0
    assert run_api["registry"].runs.list_runs() == []
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()


def test_injected_policy_limit_applies_to_exact_selected_count(run_api: dict[str, Any]) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    run_api["client"].app.state.run_guardrail_policy = RunApiGuardrailPolicy(
        max_selected_samples=2
    )
    payload = run_payload(model_id, dataset_id)
    payload.update({"sample_size": None, "sample_ids": []})
    response = run_api["client"].post("/api/v1/runs", json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Run request exceeds the synchronous API limits",
        "code": "run_guardrail_violation",
    }
    assert run_api["registry"].runs.list_runs() == []
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()


def test_duplicate_sample_ids_keep_request_validation_behavior(run_api: dict[str, Any]) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    payload = run_payload(model_id, dataset_id)
    payload.update({"sample_size": 2, "sample_ids": ["private-id", "private-id"]})
    response = run_api["client"].post("/api/v1/runs", json=payload)
    assert response.status_code == 422
    assert "private-id" not in response.text
    assert run_api["registry"].runs.list_runs() == []
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()


def test_expected_dataset_preflight_failure_maps_to_safe_409_without_side_effects(
    run_api: dict[str, Any],
) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    executor_calls = 0

    def fail_loader(_: object) -> list[object]:
        raise OSError(r"C:\Users\private\dataset.jsonl api_key=private-value")

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("executor must not be invoked")

    run_api["client"].app.dependency_overrides[get_run_preflight_service] = lambda: RunPreflightService(
        loader=fail_loader  # type: ignore[arg-type]
    )
    run_api["client"].app.state.benchmark_executor = forbidden_executor
    response = run_api["client"].post(
        "/api/v1/runs", json=run_payload(model_id, dataset_id)
    )
    assert response.status_code == 409
    assert response.json() == {
        "detail": "Registered dataset is unavailable for preflight",
        "code": "dataset_preflight_failed",
    }
    assert "private" not in response.text.lower()
    assert executor_calls == 0
    assert run_api["registry"].runs.list_runs() == []
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()


@pytest.mark.parametrize(
    "error",
    [
        TypeError("private prompt and api_key=private-key"),
        AttributeError("private dataset path /home/private/data.jsonl"),
        AssertionError("private traceback SELECT * FROM sample_results"),
    ],
)
def test_unexpected_preflight_errors_map_to_sanitized_500_without_side_effects(
    run_api: dict[str, Any], error: Exception
) -> None:
    _, model_id, dataset_id = register_fixture(run_api)
    executor_calls = 0

    def fail_loader(_: object) -> list[object]:
        raise error

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("executor must not be invoked")

    app = run_api["client"].app
    app.dependency_overrides[get_run_preflight_service] = lambda: RunPreflightService(
        loader=fail_loader  # type: ignore[arg-type]
    )
    app.state.benchmark_executor = forbidden_executor
    with TestClient(app, raise_server_exceptions=False) as safe_client:
        response = safe_client.post(
            "/api/v1/runs", json=run_payload(model_id, dataset_id)
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "private" not in response.text.lower()
    assert executor_calls == 0
    assert run_api["registry"].runs.list_runs() == []
    assert sample_row_count(run_api) == 0
    assert not run_api["output_root"].exists()


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
    assert run["status"] == "queued"
    assert run_one_worker_job(run_api) is True
    run = run_api["client"].get(f"/api/v1/runs/{run['id']}").json()
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
        assert run["status"] == "queued"
        service = BenchmarkApplicationService(
            endpoints=registry.endpoints,
            models=registry.models,
            datasets=registry.datasets,
            runs=registry.runs,
            samples=registry.samples,
            executor=fail,
        )
        assert BenchmarkWorker(runs=registry.runs, service=service).run_once() is True
        run = client.get(f"/api/v1/runs/{run['id']}").json()
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


def test_uploaded_csv_runs_end_to_end_with_portable_provenance(tmp_path: Path) -> None:
    database_path = tmp_path / "uploaded-run.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")
    registry = create_registry_repositories(database_url)
    storage_root = tmp_path / "datasets"
    output_root = tmp_path / "artifacts"
    app = create_app(
        registry,
        dataset_storage_root=storage_root,
        run_output_root=output_root,
    )
    endpoint = registry.endpoints.create(
        name="uploaded-mock",
        provider_type="mock",
        base_url="mock://provider",
    )
    model = registry.models.create(
        name="Uploaded Mock",
        model_identifier="uploaded-mock-model",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
        default_generation_config={"scenario_cycle": ["correct"]},
    )
    content = (
        b"sample_id,question,option_A,option_B,correct_answer,category\n"
        b"002,Second question,No,Yes,B,logic\n"
        b"001,First question,Yes,No,A,science\n"
    )

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/datasets/upload",
            data={"name": "uploaded-run-dataset", "file_format": "csv", "split": "test"},
            files={"file": ("questions.csv", content, "text/csv")},
        )
        assert upload.status_code == 201
        dataset = upload.json()
        payload = run_payload(model.id, dataset["id"])
        payload.update({"sample_size": 2, "sample_ids": ["001", "002"]})
        response = client.post("/api/v1/runs", json=payload)

        assert response.status_code == 201
        run = response.json()
        assert run["status"] == "queued"
        assert run_one_worker_job({"registry": registry, "client": client}) is True
        run = client.get(f"/api/v1/runs/{run['id']}").json()
        assert run["status"] == "completed"
        results = client.get(f"/api/v1/runs/{run['id']}/results").json()

    assert [result["sample_id"] for result in results] == ["001", "002"]
    assert all(result["evaluation_status"] == "correct" for result in results)
    assert all(result["ttft_ms"] is None for result in results)
    assert all(result["throughput_tokens_per_second"] is None for result in results)
    persisted = registry.runs.get_by_id(run["id"])
    dataset_config = persisted.resolved_config_json["dataset"]
    assert dataset_config["source"] == "uploaded"
    assert dataset_config["storage_key"] == dataset["source_uri"]
    assert dataset_config["checksum"] == dataset["checksum"]
    assert dataset_config["adapter_type"] == "tabular_mcq_csv_v1"
    assert dataset_config["path"] is None
    assert str(storage_root) not in repr(dataset_config)
    artifact_directory = Path(persisted.artifact_directory)
    manifest = json.loads((artifact_directory / "dataset_manifest.json").read_text("utf-8"))
    assert manifest["storage_key"] == dataset["source_uri"]
    assert manifest["checksum"] == dataset["checksum"]
    assert str(storage_root) not in repr(manifest)
    assert (artifact_directory / "resolved_config.json").is_file()
    assert (artifact_directory / "results.jsonl").is_file()
    assert (artifact_directory / "summary.json").is_file()


@pytest.mark.parametrize("failure", ["missing", "tampered", "invalid"])
def test_uploaded_preflight_failures_are_safe_and_have_no_side_effects(
    tmp_path: Path,
    failure: str,
) -> None:
    database_path = tmp_path / f"uploaded-{failure}.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_config, "head")
    registry = create_registry_repositories(database_url)
    storage_root = tmp_path / "datasets"
    output_root = tmp_path / "artifacts"
    storage = LocalDatasetStorage(storage_root)
    valid = (
        b"sample_id,question,option_A,option_B,correct_answer,category\n"
        b"001,Question,Yes,No,A,test\n"
    )
    content = b"invalid,uploaded,content\n" if failure == "invalid" else valid
    stored = storage.store(BytesIO(content), "csv")
    if failure == "missing":
        storage.remove(stored.storage_key)
    elif failure == "tampered":
        storage.resolve(stored.storage_key).write_bytes(b"private tampered dataset row")
    endpoint = registry.endpoints.create(
        name=f"uploaded-{failure}", provider_type="mock", base_url="mock://provider"
    )
    model = registry.models.create(
        name="Mock",
        model_identifier="mock",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
    )
    dataset = registry.datasets.create(
        name="uploaded-invalid",
        source_type="uploaded",
        source_uri=stored.storage_key,
        revision=None,
        split="test",
        task_type="multiple_choice",
        adapter_type="tabular_mcq_csv_v1",
        checksum=f"sha256:{stored.checksum_sha256}",
    )
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("executor must not run")

    app = create_app(
        registry,
        dataset_storage_root=storage_root,
        run_output_root=output_root,
        benchmark_executor=forbidden_executor,
    )
    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json=run_payload(model.id, dataset.id))

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Registered dataset is unavailable for preflight",
        "code": "dataset_preflight_failed",
    }
    assert str(tmp_path) not in response.text
    assert "tampered" not in response.text
    assert executor_calls == 0
    assert registry.runs.list_runs() == []
    assert sample_row_count({"database_path": database_path}) == 0
    assert not output_root.exists()

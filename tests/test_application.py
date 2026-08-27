from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update

from llm_benchmark.application import (
    BenchmarkApplicationService,
    ClaimedRunStateError,
    RegistrationMismatchError,
)
from llm_benchmark.config import RunConfig, load_config
from llm_benchmark.db import Base, create_db_engine, create_session_factory
from llm_benchmark.db.errors import InactiveDependencyError, RecordNotFoundError
from llm_benchmark.db.models import BenchmarkRun, RunStatus
from llm_benchmark.db.repositories import (
    BenchmarkRunRepository,
    DatasetRepository,
    ModelRepository,
    ProviderEndpointRepository,
    SampleResultRepository,
)
from llm_benchmark.reproducibility import canonical_hash
from llm_benchmark.runner import PipelineExecution, execute_benchmark


@pytest.fixture
def service_context(tmp_path: Path) -> dict[str, Any]:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'application.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    endpoints = ProviderEndpointRepository(session_factory)
    models = ModelRepository(session_factory)
    datasets = DatasetRepository(session_factory)
    runs = BenchmarkRunRepository(session_factory)
    samples = SampleResultRepository(session_factory)
    return {
        "engine": engine,
        "session_factory": session_factory,
        "endpoints": endpoints,
        "models": models,
        "datasets": datasets,
        "runs": runs,
        "samples": samples,
        "service": BenchmarkApplicationService(
            endpoints=endpoints,
            models=models,
            datasets=datasets,
            runs=runs,
            samples=samples,
        ),
    }


def fixture_config(tmp_path: Path, *, scenario: str | None = None) -> RunConfig:
    source = load_config(Path("configs/mock_smoke.yaml"))
    model = source.models[0]
    if scenario is not None:
        model = model.model_copy(update={"scenario_cycle": [scenario]})
    return source.model_copy(update={"models": [model], "output_dir": tmp_path / "outputs"})


def register_config(context: dict[str, Any], config: RunConfig) -> tuple[Any, Any, Any]:
    model_config = config.models[0]
    endpoint = context["endpoints"].create(
        name=model_config.endpoint_alias,
        provider_type="mock",
        base_url="mock://provider",
    )
    model = context["models"].create(
        name="Mock model",
        model_identifier=model_config.model_id,
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
        capabilities={},
    )
    dataset = context["datasets"].create(
        name=config.dataset.name,
        source_type=config.dataset.source,
        source_uri=config.dataset.path.as_posix() if config.dataset.path else config.dataset.name,
        revision=config.dataset.revision,
        split=config.dataset.split,
        task_type="multiple_choice",
        adapter_type="local_jsonl",
        license="CC0-1.0",
    )
    return endpoint, model, dataset


def execute_registered(context: dict[str, Any], config: RunConfig) -> Any:
    endpoint, model, dataset = register_config(context, config)
    result = context["service"].execute(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
    )
    return endpoint, model, dataset, result


def enqueue_and_claim(
    context: dict[str, Any],
    config: RunConfig,
    *,
    selected_sample_count: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    endpoint, model, dataset = register_config(context, config)
    queued = context["service"].enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=(
            selected_sample_count
            if selected_sample_count is not None
            else config.dataset.sample_size or len(config.dataset.sample_ids)
        ),
    )
    running = context["runs"].claim_next_queued()
    assert running is not None
    assert running.id == queued.run.id
    return endpoint, model, dataset, running


def update_run_record(context: dict[str, Any], run_id: int, **values: Any) -> Any:
    with context["session_factory"]() as session, session.begin():
        session.execute(update(BenchmarkRun).where(BenchmarkRun.id == run_id).values(**values))
    return context["runs"].get_by_id(run_id)


def test_enqueue_creates_only_queued_run_with_exact_count_and_stable_hash(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("enqueue must not call the executor")

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=forbidden_executor,
    )
    result = service.enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=3,
    )

    resolved = config.model_dump(mode="json")
    assert result.run.status is RunStatus.QUEUED
    assert result.run.started_at is None
    assert result.run.completed_at is None
    assert result.run.sample_count == 3
    assert result.run.resolved_config_json == resolved
    assert result.run.config_hash == canonical_hash(resolved)
    assert result.samples == ()
    assert result.summary is None
    assert executor_calls == 0
    assert service_context["samples"].list_by_run_id(result.run.id) == []
    assert not config.output_dir.exists()


def test_execute_claimed_requires_running_run(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    queued = service_context["service"].enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=8,
    )

    with pytest.raises(ClaimedRunStateError):
        service_context["service"].execute_claimed(queued.run)


def test_execute_claimed_reconstructs_config_and_completes(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    _, _, _, running = enqueue_and_claim(service_context, config, selected_sample_count=8)

    result = service_context["service"].execute_claimed(running)

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.sample_count == 8
    assert len(result.samples) == 8
    assert result.summary == result.run.summary_json


def test_invalid_persisted_config_fails_before_execution(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    _, _, _, running = enqueue_and_claim(service_context, config)
    corrupted = update_run_record(
        service_context,
        running.id,
        resolved_config_json={"schema_version": 999},
    )
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("invalid config must not execute")

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=forbidden_executor,
    )
    result = service.execute_claimed(corrupted)

    assert result.run.status is RunStatus.FAILED
    assert result.run.error_type == "PersistedRunConfigError"
    assert executor_calls == 0


def test_persisted_config_hash_mismatch_fails_before_execution(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    _, _, _, running = enqueue_and_claim(service_context, config)
    corrupted = update_run_record(service_context, running.id, config_hash="f" * 64)
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("hash mismatch must not execute")

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=forbidden_executor,
    )
    result = service.execute_claimed(corrupted)

    assert result.run.status is RunStatus.FAILED
    assert result.run.error_type == "PersistedRunConfigError"
    assert executor_calls == 0


@pytest.mark.parametrize("dependency", ["endpoint", "model", "dataset"])
def test_execute_claimed_rejects_inactive_registration_before_execution(
    service_context: dict[str, Any], tmp_path: Path, dependency: str
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset, running = enqueue_and_claim(service_context, config)
    {
        "endpoint": service_context["endpoints"],
        "model": service_context["models"],
        "dataset": service_context["datasets"],
    }[dependency].soft_delete(
        {"endpoint": endpoint.id, "model": model.id, "dataset": dataset.id}[dependency]
    )
    executor_calls = 0

    def forbidden_executor(_: RunConfig) -> PipelineExecution:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError("inactive registration must not execute")

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=forbidden_executor,
    )
    result = service.execute_claimed(running)

    assert result.run.status is RunStatus.FAILED
    assert result.run.error_type == "InactiveDependencyError"
    assert executor_calls == 0


def test_execute_claimed_revalidates_registration_identity(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, _, _, running = enqueue_and_claim(service_context, config)
    service_context["endpoints"].update(endpoint.id, name="renamed-after-enqueue")

    result = service_context["service"].execute_claimed(running)

    assert result.run.status is RunStatus.FAILED
    assert result.run.error_type == "RegistrationMismatchError"


def test_successful_registered_execution_persists_run_samples_and_artifacts(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    _, _, _, result = execute_registered(service_context, config)

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.started_at is not None
    assert result.run.completed_at is not None
    assert result.run.resolved_config_json == config.model_dump(mode="json")
    assert result.run.config_hash == result.summary["resolved_config_hash"]
    assert result.run.summary_json == result.summary
    assert result.run.artifact_directory == str(Path(result.run.artifact_directory))
    assert Path(result.run.artifact_directory).is_dir()
    assert len(result.samples) == result.run.sample_count == 8

    artifact_rows = [
        json.loads(line)
        for line in (Path(result.run.artifact_directory) / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item.sample_id for item in result.samples] == [item["sample_id"] for item in artifact_rows]
    for persisted, artifact in zip(result.samples, artifact_rows, strict=True):
        assert persisted.raw_response == artifact["raw_response"]
        assert persisted.parse_status == artifact["parse_status"]
        assert persisted.request_status == artifact["request_status"]
        assert persisted.evaluation_status == artifact["evaluation_status"]
        assert persisted.input_tokens == artifact["input_tokens"]
        assert persisted.output_tokens == artifact["total_output_tokens"]
        assert persisted.total_tokens == artifact["total_tokens"]
        assert persisted.latency_ms == artifact["logical_request_latency_ms"]
        assert persisted.ttft_ms is None
        assert persisted.throughput_tokens_per_second is None


def test_execution_failure_transitions_to_failed_with_sanitized_details(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)

    def failing_executor(_: RunConfig) -> PipelineExecution:
        raise RuntimeError("authorization = sensitive-value provider execution failed")

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=failing_executor,
    )
    queued = service.enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=8,
    )
    running = service_context["runs"].claim_next_queued()
    assert running is not None
    assert running.id == queued.run.id
    result = service.execute_claimed(running)

    assert result.run.status is RunStatus.FAILED
    assert result.run.error_type == "RuntimeError"
    assert "sensitive-value" not in (result.run.error_message or "")
    assert result.samples == ()
    assert service_context["samples"].list_by_run_id(result.run.id) == []


@pytest.mark.parametrize("dependency", ["endpoint", "model", "dataset"])
def test_inactive_registration_is_rejected_before_run_creation(
    service_context: dict[str, Any], tmp_path: Path, dependency: str
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    {
        "endpoint": service_context["endpoints"],
        "model": service_context["models"],
        "dataset": service_context["datasets"],
    }[dependency].soft_delete({"endpoint": endpoint.id, "model": model.id, "dataset": dataset.id}[dependency])

    with pytest.raises(InactiveDependencyError):
        service_context["service"].execute(
            endpoint_id=endpoint.id,
            model_id=model.id,
            dataset_id=dataset.id,
            config=config,
        )
    assert service_context["runs"].list_runs() == []


@pytest.mark.parametrize("dependency", ["endpoint", "model", "dataset"])
def test_missing_registration_is_rejected_before_run_creation(
    service_context: dict[str, Any], tmp_path: Path, dependency: str
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    identifiers = {"endpoint": endpoint.id, "model": model.id, "dataset": dataset.id}
    identifiers[dependency] = 999_999
    with pytest.raises(RecordNotFoundError):
        service_context["service"].execute(
            endpoint_id=identifiers["endpoint"],
            model_id=identifiers["model"],
            dataset_id=identifiers["dataset"],
            config=config,
        )
    assert service_context["runs"].list_runs() == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "different-model"),
        ("endpoint_alias", "different-endpoint"),
        ("provider", "lm_studio"),
    ],
)
def test_registered_model_identity_mismatch_is_rejected(
    service_context: dict[str, Any], tmp_path: Path, field: str, value: str
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    mismatched_model = config.models[0].model_copy(update={field: value})
    mismatched_config = config.model_copy(update={"models": [mismatched_model]})

    with pytest.raises(RegistrationMismatchError):
        service_context["service"].execute(
            endpoint_id=endpoint.id,
            model_id=model.id,
            dataset_id=dataset.id,
            config=mismatched_config,
        )
    assert service_context["runs"].list_runs() == []


def test_registered_dataset_identity_mismatch_is_rejected(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    mismatched_dataset = config.dataset.model_copy(update={"split": "validation"})

    with pytest.raises(RegistrationMismatchError):
        service_context["service"].execute(
            endpoint_id=endpoint.id,
            model_id=model.id,
            dataset_id=dataset.id,
            config=config.model_copy(update={"dataset": mismatched_dataset}),
        )


@pytest.mark.parametrize("scenario", ["unparseable", "request_failed"])
def test_sample_outcomes_complete_the_claimed_benchmark(
    service_context: dict[str, Any], tmp_path: Path, scenario: str
) -> None:
    config = fixture_config(tmp_path, scenario=scenario)
    _, _, _, running = enqueue_and_claim(service_context, config, selected_sample_count=8)
    result = service_context["service"].execute_claimed(running)
    assert result.run.status is RunStatus.COMPLETED
    assert all(item.evaluation_status == scenario for item in result.samples)
    if scenario == "unparseable":
        assert all(item.parse_status == "ambiguous_multiple_answers" for item in result.samples)


def test_sample_persistence_failure_is_atomic_and_marks_run_failed(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)

    def duplicate_executor(run_config: RunConfig) -> PipelineExecution:
        execution = execute_benchmark(run_config)
        return PipelineExecution(
            run_dir=execution.run_dir,
            summary=execution.summary,
            results=(execution.results[0], execution.results[0]),
        )

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=duplicate_executor,
    )
    queued = service.enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=8,
    )
    running = service_context["runs"].claim_next_queued()
    assert running is not None
    assert running.id == queued.run.id
    result = service.execute_claimed(running)
    assert result.run.status is RunStatus.FAILED
    assert result.run.error_type == "UniquenessConflictError"
    assert service_context["samples"].list_by_run_id(result.run.id) == []
    assert Path(result.run.artifact_directory).is_dir()


def test_no_database_connection_is_held_during_execution(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    observed_checked_out: list[int] = []

    def observing_executor(run_config: RunConfig) -> PipelineExecution:
        observed_checked_out.append(service_context["engine"].pool.checkedout())
        return execute_benchmark(run_config)

    service = BenchmarkApplicationService(
        endpoints=service_context["endpoints"],
        models=service_context["models"],
        datasets=service_context["datasets"],
        runs=service_context["runs"],
        samples=service_context["samples"],
        executor=observing_executor,
    )
    queued = service.enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=8,
    )
    running = service_context["runs"].claim_next_queued()
    assert running is not None
    assert running.id == queued.run.id
    result = service.execute_claimed(running)
    assert result.run.status is RunStatus.COMPLETED
    assert observed_checked_out == [0]


def test_backward_compatible_execute_transitions_only_its_own_run(
    service_context: dict[str, Any], tmp_path: Path
) -> None:
    config = fixture_config(tmp_path)
    endpoint, model, dataset = register_config(service_context, config)
    older = service_context["service"].enqueue(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
        selected_sample_count=8,
    )

    result = service_context["service"].execute(
        endpoint_id=endpoint.id,
        model_id=model.id,
        dataset_id=dataset.id,
        config=config,
    )

    assert result.run.id != older.run.id
    assert result.run.status is RunStatus.COMPLETED
    assert service_context["runs"].get_by_id(older.run.id).status is RunStatus.QUEUED

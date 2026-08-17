from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError

from llm_benchmark.db import Base, create_db_engine, create_session_factory
from llm_benchmark.db.errors import (
    InactiveDependencyError,
    InvalidStateTransitionError,
    RecordNotFoundError,
    UniquenessConflictError,
)
from llm_benchmark.db.models import ProviderEndpoint, RunStatus
from llm_benchmark.db.records import SampleResultCreate
from llm_benchmark.db.repositories import (
    BenchmarkRunRepository,
    DatasetRepository,
    ModelRepository,
    ProviderEndpointRepository,
    SampleResultRepository,
)


@pytest.fixture
def repositories(tmp_path: Path) -> dict[str, Any]:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'repositories.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    return {
        "engine": engine,
        "session_factory": session_factory,
        "endpoints": ProviderEndpointRepository(session_factory),
        "models": ModelRepository(session_factory),
        "datasets": DatasetRepository(session_factory),
        "runs": BenchmarkRunRepository(session_factory),
        "samples": SampleResultRepository(session_factory),
    }


def create_dependencies(repositories: dict[str, Any], suffix: str = "") -> tuple[Any, Any, Any]:
    endpoint = repositories["endpoints"].create(
        name=f"endpoint{suffix}",
        provider_type="lm_studio_native",
        base_url="http://127.0.0.1:1234",
    )
    model = repositories["models"].create(
        name=f"model{suffix}",
        model_identifier=f"model-id{suffix}",
        endpoint_id=endpoint.id,
        reasoning_policy="toggle",
        capabilities={"reasoning_output": True},
        default_generation_config={"temperature": 0},
        metadata={"family": "test"},
    )
    dataset = repositories["datasets"].create(
        name=f"dataset{suffix}",
        source_type="fixture",
        source_uri=f"data/fixture{suffix}.jsonl",
        revision="fixture-v1",
        split="test",
        task_type="multiple_choice",
        adapter_type="local_jsonl",
        license="CC0-1.0",
        checksum=f"checksum{suffix}",
    )
    return endpoint, model, dataset


def create_run(repositories: dict[str, Any], suffix: str = "") -> Any:
    _, model, dataset = create_dependencies(repositories, suffix)
    return repositories["runs"].create_queued(
        experiment_name=f"experiment{suffix}",
        model_id=model.id,
        dataset_id=dataset.id,
        resolved_config={"schema_version": 1, "suffix": suffix},
        config_hash=("a" * 63) + (suffix[-1:] or "a"),
        seed=42,
        sample_count=2,
        artifact_directory=f"outputs/run{suffix}",
    )


def sample(sample_id: str, *, ttft_ms: float | None = None) -> SampleResultCreate:
    return SampleResultCreate(
        sample_id=sample_id,
        category="test",
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
        latency_ms=25.0,
        ttft_ms=ttft_ms,
        throughput_tokens_per_second=None,
    )


def test_endpoint_crud_active_listing_soft_delete_and_conflict(repositories: dict[str, Any]) -> None:
    repository = repositories["endpoints"]
    endpoint = repository.create(
        name="local",
        provider_type="lm_studio_native",
        base_url="http://127.0.0.1:1234",
        credential_env_var="LOCAL_API_KEY",
    )
    assert repository.get_by_id(endpoint.id).name == "local"
    assert [item.id for item in repository.list_active()] == [endpoint.id]

    updated = repository.update(endpoint.id, name="local-updated", credential_env_var=None)
    assert updated.name == "local-updated"
    assert updated.credential_env_var is None
    assert updated.updated_at >= endpoint.updated_at

    with pytest.raises(UniquenessConflictError):
        repository.create(
            name="local-updated",
            provider_type="mock",
            base_url="mock://provider",
        )

    deleted = repository.soft_delete(endpoint.id)
    assert deleted.is_active is False
    assert repository.list_active() == []
    assert repository.get_by_id(endpoint.id).is_active is False

    with pytest.raises(RecordNotFoundError):
        repository.get_by_id(999_999)


def test_model_crud_capability_validation_and_soft_delete(repositories: dict[str, Any]) -> None:
    endpoint = repositories["endpoints"].create(
        name="model-endpoint", provider_type="mock", base_url="mock://provider"
    )
    repository = repositories["models"]
    model = repository.create(
        name="model",
        model_identifier="model-1",
        endpoint_id=endpoint.id,
        reasoning_policy="unsupported",
        capabilities={"streaming": False},
    )
    updated = repository.update(
        model.id,
        reasoning_policy="toggle",
        capabilities={"reasoning_output": True, "streaming": True},
        default_generation_config={"temperature": 0.5},
        metadata={"owner": "team"},
    )
    assert updated.capabilities_json == {"reasoning_output": True, "streaming": True}
    assert updated.default_generation_config_json == {"temperature": 0.5}
    assert updated.metadata_json == {"owner": "team"}
    assert repository.get_by_id(model.id).reasoning_policy.value == "toggle"

    with pytest.raises(ValidationError):
        repository.update(model.id, capabilities={"unknown": True})
    with pytest.raises(UniquenessConflictError):
        repository.create(
            name="duplicate",
            model_identifier="model-1",
            endpoint_id=endpoint.id,
            reasoning_policy="unsupported",
        )

    repository.soft_delete(model.id)
    assert repository.list_active() == []
    assert repository.get_by_id(model.id).is_active is False


def test_dataset_crud_and_both_identity_policies(repositories: dict[str, Any]) -> None:
    repository = repositories["datasets"]
    versioned = repository.create(
        name="versioned",
        source_type="huggingface",
        source_uri="TIGER-Lab/MMLU-Pro",
        revision="abc123",
        split="test",
        task_type="multiple_choice",
        adapter_type="mmlu_pro",
    )
    updated = repository.update(
        versioned.id,
        name="versioned-updated",
        license="MIT",
        checksum="sha256:value",
        metadata={"homepage": "example"},
    )
    assert updated.license == "MIT"
    assert updated.metadata_json == {"homepage": "example"}

    versioned_kwargs = {
        "name": "duplicate",
        "source_type": "huggingface",
        "source_uri": "TIGER-Lab/MMLU-Pro",
        "revision": "abc123",
        "split": "test",
        "task_type": "multiple_choice",
        "adapter_type": "mmlu_pro",
    }
    with pytest.raises(UniquenessConflictError):
        repository.create(**versioned_kwargs)

    unversioned_kwargs = {
        "name": "unversioned",
        "source_type": "local",
        "source_uri": "data/local.jsonl",
        "revision": None,
        "split": "test",
        "task_type": "multiple_choice",
        "adapter_type": "local_jsonl",
    }
    unversioned = repository.create(**unversioned_kwargs)
    with pytest.raises(UniquenessConflictError):
        repository.create(**unversioned_kwargs)

    repository.soft_delete(unversioned.id)
    assert [item.id for item in repository.list_active()] == [versioned.id]
    assert repository.get_by_id(unversioned.id).is_active is False


def test_inactive_dependencies_reject_new_models_and_runs(repositories: dict[str, Any]) -> None:
    endpoint, model, dataset = create_dependencies(repositories, "-inactive")
    repositories["endpoints"].soft_delete(endpoint.id)
    with pytest.raises(InactiveDependencyError):
        repositories["models"].create(
            name="blocked",
            model_identifier="blocked",
            endpoint_id=endpoint.id,
            reasoning_policy="unsupported",
        )
    with pytest.raises(InactiveDependencyError):
        repositories["runs"].create_queued(
            experiment_name="blocked-endpoint",
            model_id=model.id,
            dataset_id=dataset.id,
            resolved_config={},
            config_hash="b" * 64,
            seed=42,
            sample_count=1,
            artifact_directory="outputs/blocked",
        )

    endpoint2, model2, dataset2 = create_dependencies(repositories, "-model")
    repositories["models"].soft_delete(model2.id)
    with pytest.raises(InactiveDependencyError):
        repositories["runs"].create_queued(
            experiment_name="blocked-model",
            model_id=model2.id,
            dataset_id=dataset2.id,
            resolved_config={},
            config_hash="c" * 64,
            seed=42,
            sample_count=1,
            artifact_directory="outputs/blocked",
        )

    endpoint3, model3, dataset3 = create_dependencies(repositories, "-dataset")
    repositories["datasets"].soft_delete(dataset3.id)
    with pytest.raises(InactiveDependencyError):
        repositories["runs"].create_queued(
            experiment_name="blocked-dataset",
            model_id=model3.id,
            dataset_id=dataset3.id,
            resolved_config={},
            config_hash="d" * 64,
            seed=42,
            sample_count=1,
            artifact_directory="outputs/blocked",
        )


def test_queued_run_persists_immutable_inputs_and_lists(repositories: dict[str, Any]) -> None:
    _, model, dataset = create_dependencies(repositories, "-run")
    resolved_config = {"schema_version": 1, "provider": {"type": "mock"}}
    run = repositories["runs"].create_queued(
        experiment_name="queued-run",
        model_id=model.id,
        dataset_id=dataset.id,
        resolved_config=resolved_config,
        config_hash="e" * 64,
        seed=42,
        sample_count=14,
        artifact_directory="outputs/queued",
    )
    resolved_config["schema_version"] = 99

    persisted = repositories["runs"].get_by_id(run.id)
    assert persisted.status is RunStatus.QUEUED
    assert persisted.resolved_config_json["schema_version"] == 1
    assert persisted.config_hash == "e" * 64
    assert persisted.artifact_directory == "outputs/queued"
    assert [item.id for item in repositories["runs"].list_runs()] == [run.id]


@pytest.mark.parametrize("terminal", [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED])
def test_valid_running_to_terminal_transitions(repositories: dict[str, Any], terminal: RunStatus) -> None:
    run = create_run(repositories, f"-{terminal.value}")
    running = repositories["runs"].transition_status(run.id, RunStatus.RUNNING)
    assert running.started_at is not None
    assert running.completed_at is None

    terminal_run = repositories["runs"].transition_status(
        run.id,
        terminal,
        summary={"accuracy": 0.5},
        artifact_directory=f"outputs/{terminal.value}",
        error_type="server_error" if terminal is RunStatus.FAILED else None,
        error_message="authorization = sensitive-value\nprovider failed" if terminal is RunStatus.FAILED else None,
    )
    assert terminal_run.completed_at is not None
    assert terminal_run.summary_json == {"accuracy": 0.5}
    assert terminal_run.artifact_directory == f"outputs/{terminal.value}"
    if terminal is RunStatus.FAILED:
        assert terminal_run.error_type == "server_error"
        assert "sensitive-value" not in (terminal_run.error_message or "")


@pytest.mark.parametrize("terminal", [RunStatus.FAILED, RunStatus.CANCELLED])
def test_valid_queued_to_terminal_transitions(repositories: dict[str, Any], terminal: RunStatus) -> None:
    run = create_run(repositories, f"-queued-{terminal.value}")
    result = repositories["runs"].transition_status(run.id, terminal)
    assert result.status is terminal
    assert result.started_at is None
    assert result.completed_at is not None


def test_invalid_and_terminal_run_transitions(repositories: dict[str, Any]) -> None:
    run = create_run(repositories, "-invalid")
    with pytest.raises(InvalidStateTransitionError):
        repositories["runs"].transition_status(run.id, RunStatus.QUEUED)
    completed = repositories["runs"].transition_status(run.id, RunStatus.RUNNING)
    completed = repositories["runs"].transition_status(completed.id, RunStatus.COMPLETED)
    with pytest.raises(InvalidStateTransitionError):
        repositories["runs"].transition_status(completed.id, RunStatus.FAILED)


def test_soft_delete_preserves_referenced_history_and_foreign_keys(repositories: dict[str, Any]) -> None:
    endpoint, model, dataset = create_dependencies(repositories, "-history")
    run = repositories["runs"].create_queued(
        experiment_name="history",
        model_id=model.id,
        dataset_id=dataset.id,
        resolved_config={},
        config_hash="f" * 64,
        seed=42,
        sample_count=1,
        artifact_directory="outputs/history",
    )
    repositories["endpoints"].soft_delete(endpoint.id)
    repositories["models"].soft_delete(model.id)
    repositories["datasets"].soft_delete(dataset.id)
    assert repositories["runs"].get_by_id(run.id).id == run.id

    with repositories["session_factory"]() as session:
        with pytest.raises(IntegrityError):
            with session.begin():
                session.execute(delete(ProviderEndpoint).where(ProviderEndpoint.id == endpoint.id))


def test_single_and_bulk_sample_results_preserve_nullable_metrics(repositories: dict[str, Any]) -> None:
    run = create_run(repositories, "-samples")
    first = repositories["samples"].add_one(run.id, sample("one"))
    assert first.ttft_ms is None
    assert first.throughput_tokens_per_second is None

    unsafe_error = sample("three").model_copy(
        update={"provider_error_message": "token: sensitive-value provider failed"}
    )
    bulk = repositories["samples"].add_many(
        run.id,
        [
            sample("two", ttft_ms=12.5),
            unsafe_error,
        ],
    )
    assert [item.sample_id for item in bulk] == ["two", "three"]
    persisted = repositories["samples"].list_by_run_id(run.id)
    assert [item.sample_id for item in persisted] == ["one", "two", "three"]
    assert persisted[1].ttft_ms == 12.5
    assert "sensitive-value" not in (persisted[2].provider_error_message or "")


def test_bulk_sample_failure_rolls_back_entire_transaction(repositories: dict[str, Any]) -> None:
    run = create_run(repositories, "-rollback")
    with pytest.raises(UniquenessConflictError):
        repositories["samples"].add_many(run.id, [sample("duplicate"), sample("duplicate")])
    assert repositories["samples"].list_by_run_id(run.id) == []

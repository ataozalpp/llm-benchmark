"""Registered RAG run lifecycle for the API and worker.

Mirrors BenchmarkApplicationService: short transactions for queueing, the
claim, result persistence and the final status, with no database
transaction open while retrieval and generation run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from ..application import (
    ClaimedRunStateError,
    PersistedRunConfigError,
    RegistrationMismatchError,
)
from ..dataset_storage import LocalDatasetStorage
from ..db.models import RunStatus, TaskType
from ..db.records import (
    BenchmarkRunRecord,
    DatasetRecord,
    ModelRecord,
    ProviderEndpointRecord,
    RagCaseResultCreate,
    RagCaseResultRecord,
)
from ..reproducibility import canonical_hash
from .dataset import load_rag_dataset
from .run_config import LoadedRagRunConfig, RagRunConfig
from .run_resolution import ResolvedRagRun, validate_registrations
from .run_service import RagPipelineExecution, execute_rag_benchmark


class EndpointReader(Protocol):
    def get_by_id(self, endpoint_id: int) -> ProviderEndpointRecord: ...


class ModelReader(Protocol):
    def get_by_id(self, model_id: int) -> ModelRecord: ...


class DatasetReader(Protocol):
    def get_by_id(self, dataset_id: int) -> DatasetRecord: ...


class RagCaseWriter(Protocol):
    def add_many(
        self, run_id: int, cases: list[RagCaseResultCreate]
    ) -> list[RagCaseResultRecord]: ...


class RagServiceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run: BenchmarkRunRecord
    cases: tuple[RagCaseResultRecord, ...] = ()
    summary: dict[str, object] | None = None


class RagApplicationService:
    def __init__(
        self,
        *,
        endpoints: EndpointReader,
        models: ModelReader,
        datasets: DatasetReader,
        runs,
        rag_cases: RagCaseWriter,
        storage: LocalDatasetStorage,
        executor: Callable[[LoadedRagRunConfig], RagPipelineExecution] = (
            execute_rag_benchmark
        ),
    ) -> None:
        self._endpoints = endpoints
        self._models = models
        self._datasets = datasets
        self._runs = runs
        self._rag_cases = rag_cases
        self._storage = storage
        self._executor = executor

    def enqueue(self, resolved: ResolvedRagRun, scenario_count: int) -> RagServiceResult:
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        resolved_config = resolved.config.model_dump(mode="json")
        run = self._runs.create_queued(
            experiment_name=resolved.experiment_name,
            model_id=resolved.model.id,
            dataset_id=resolved.dataset.id,
            resolved_config=resolved_config,
            config_hash=canonical_hash(resolved_config),
            seed=None,
            sample_count=scenario_count,
            artifact_directory=resolved.config.output_dir,
            task_type=TaskType.RAG,
        )
        return RagServiceResult(run=run)

    def execute_claimed(self, run: BenchmarkRunRecord) -> RagServiceResult:
        if run.status is not RunStatus.RUNNING:
            raise ClaimedRunStateError(
                "Claimed execution requires a running benchmark run"
            )
        if run.task_type is not TaskType.RAG:
            raise ClaimedRunStateError("RAG execution requires a RAG run")

        execution: RagPipelineExecution | None = None
        try:
            try:
                config = RagRunConfig.model_validate(run.resolved_config_json)
            except (TypeError, ValueError, ValidationError) as error:
                raise PersistedRunConfigError(
                    "Persisted run configuration is invalid"
                ) from error
            if canonical_hash(config.model_dump(mode="json")) != run.config_hash:
                raise PersistedRunConfigError(
                    "Persisted run configuration hash does not match"
                )

            model = self._models.get_by_id(run.model_id)
            endpoint = self._endpoints.get_by_id(model.endpoint_id)
            dataset = self._datasets.get_by_id(run.dataset_id)
            validate_registrations(endpoint, model, dataset)
            _validate_identity(config, endpoint, model, dataset)

            suite_path = self._storage.verify(dataset.source_uri, dataset.checksum)
            answerable = {
                scenario.query.query_id: scenario.answerable
                for scenario in load_rag_dataset(suite_path).scenarios
            }
            execution = self._executor(
                LoadedRagRunConfig(config, suite_path, Path(config.output_dir))
            )
            cases = self._rag_cases.add_many(
                run.id, [_to_case(row, answerable) for row in execution.rows]
            )
            run = self._runs.transition_status(
                run.id,
                "completed",
                summary=execution.summary,
                artifact_directory=str(execution.run_dir),
                sample_count=len(cases),
            )
            return RagServiceResult(
                run=run, cases=tuple(cases), summary=execution.summary
            )
        except Exception as error:
            run = self._runs.transition_status(
                run.id,
                "failed",
                artifact_directory=(
                    str(execution.run_dir)
                    if execution is not None
                    else run.artifact_directory
                ),
                error_type=type(error).__name__,
                error_message=str(error),
            )
            return RagServiceResult(run=run)


def _validate_identity(
    config: RagRunConfig,
    endpoint: ProviderEndpointRecord,
    model: ModelRecord,
    dataset: DatasetRecord,
) -> None:
    mismatches = []
    if config.suite_path != dataset.source_uri:
        mismatches.append("suite_path")
    if endpoint.provider_type.value == "mock":
        if config.provider != "synthetic":
            mismatches.append("provider")
    else:
        configured = config.model
        if config.provider != "openai_compatible" or configured is None:
            mismatches.append("provider")
        else:
            if configured.endpoint_alias != endpoint.name:
                mismatches.append("model.endpoint_alias")
            if configured.model_id != model.model_identifier:
                mismatches.append("model.model_id")
            if (configured.base_url or "").rstrip("/") != endpoint.base_url.rstrip("/"):
                mismatches.append("model.base_url")
    if mismatches:
        raise RegistrationMismatchError(
            "Configuration conflicts with selected registrations: "
            + ", ".join(mismatches)
        )


def _to_case(row: dict[str, object], answerable: dict[str, bool]) -> RagCaseResultCreate:
    response = row.get("response") or {}
    return RagCaseResultCreate(
        query_id=row["query_id"],
        answerable=answerable[row["query_id"]],
        status=row["status"],
        failure_code=row.get("failure_code"),
        retrieved_chunk_ids=list(row["retrieved_chunk_ids"]),
        context_chunk_ids=list(row["context_chunk_ids"]),
        raw_response=response.get("text"),
        finish_reason=response.get("finish_reason"),
        parsed_answer=row.get("answer"),
        evaluation=row.get("evaluation"),
        input_tokens=response.get("input_tokens"),
        output_tokens=response.get("output_tokens"),
        retrieval_latency_ms=row.get("retrieval_latency_ms"),
        generation_latency_ms=row.get("generation_latency_ms"),
    )

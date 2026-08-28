"""Framework-independent orchestration for registered benchmark execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .config import RunConfig
from .db.errors import InactiveDependencyError
from .db.models import RunStatus
from .db.records import (
    BenchmarkRunRecord,
    DatasetRecord,
    ModelRecord,
    ProviderEndpointRecord,
    SampleResultCreate,
    SampleResultRecord,
)
from .reproducibility import canonical_hash
from .runner import PipelineExecution, execute_benchmark


class ApplicationServiceError(Exception):
    """Base class for application-service validation failures."""


class RegistrationMismatchError(ApplicationServiceError):
    """Raised when a selected registration conflicts with the run config."""


class ClaimedRunStateError(ApplicationServiceError):
    """Raised when execution is requested for a run that is not running."""


class PersistedRunConfigError(ApplicationServiceError):
    """Raised when persisted run configuration cannot be trusted."""


class EndpointReader(Protocol):
    def get_by_id(self, endpoint_id: int) -> ProviderEndpointRecord: ...


class ModelReader(Protocol):
    def get_by_id(self, model_id: int) -> ModelRecord: ...


class DatasetReader(Protocol):
    def get_by_id(self, dataset_id: int) -> DatasetRecord: ...


class RunWriter(Protocol):
    def create_queued(
        self,
        *,
        experiment_name: str,
        model_id: int,
        dataset_id: int,
        resolved_config: dict[str, object],
        config_hash: str,
        seed: int,
        sample_count: int,
        artifact_directory: str,
    ) -> BenchmarkRunRecord: ...

    def transition_status(
        self,
        run_id: int,
        requested_status: str,
        *,
        summary: dict[str, object] | None = None,
        artifact_directory: str | None = None,
        sample_count: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> BenchmarkRunRecord: ...


class SampleWriter(Protocol):
    def add_many(self, run_id: int, samples: list[SampleResultCreate]) -> list[SampleResultRecord]: ...


class BenchmarkServiceResult(BaseModel):
    """Immutable result returned to API, CLI, or worker adapters in later phases."""

    model_config = ConfigDict(frozen=True)

    run: BenchmarkRunRecord
    samples: tuple[SampleResultRecord, ...] = ()
    summary: dict[str, object] | None = None


class BenchmarkApplicationService:
    def __init__(
        self,
        *,
        endpoints: EndpointReader,
        models: ModelReader,
        datasets: DatasetReader,
        runs: RunWriter,
        samples: SampleWriter,
        executor: Callable[[RunConfig], PipelineExecution] = execute_benchmark,
    ) -> None:
        self._endpoints = endpoints
        self._models = models
        self._datasets = datasets
        self._runs = runs
        self._samples = samples
        self._executor = executor

    def enqueue(
        self,
        *,
        endpoint_id: int,
        model_id: int,
        dataset_id: int,
        config: RunConfig,
        selected_sample_count: int,
    ) -> BenchmarkServiceResult:
        if selected_sample_count < 0:
            raise ValueError("selected_sample_count must not be negative")
        endpoint = self._endpoints.get_by_id(endpoint_id)
        model = self._models.get_by_id(model_id)
        dataset = self._datasets.get_by_id(dataset_id)
        self._validate_active(endpoint, model, dataset)
        self._validate_identity(config, endpoint, model, dataset)

        resolved_config = config.model_dump(mode="json")
        config_hash = canonical_hash(resolved_config)
        run = self._runs.create_queued(
            experiment_name=config.experiment_name,
            model_id=model.id,
            dataset_id=dataset.id,
            resolved_config=resolved_config,
            config_hash=config_hash,
            seed=config.seed,
            sample_count=selected_sample_count,
            artifact_directory=str(config.output_dir),
        )
        return BenchmarkServiceResult(run=run)

    def execute_claimed(self, run: BenchmarkRunRecord) -> BenchmarkServiceResult:
        if run.status is not RunStatus.RUNNING:
            raise ClaimedRunStateError("Claimed execution requires a running benchmark run")

        pipeline_execution: PipelineExecution | None = None
        try:
            try:
                config = RunConfig.model_validate(run.resolved_config_json)
            except (TypeError, ValueError) as error:
                raise PersistedRunConfigError("Persisted run configuration is invalid") from error
            if canonical_hash(config.model_dump(mode="json")) != run.config_hash:
                raise PersistedRunConfigError("Persisted run configuration hash does not match")

            model = self._models.get_by_id(run.model_id)
            endpoint = self._endpoints.get_by_id(model.endpoint_id)
            dataset = self._datasets.get_by_id(run.dataset_id)
            self._validate_active(endpoint, model, dataset)
            self._validate_identity(config, endpoint, model, dataset)

            pipeline_execution = self._executor(config)
            sample_inputs = [_to_sample_input(result) for result in pipeline_execution.results]
            persisted_samples = self._samples.add_many(run.id, sample_inputs)
            run = self._runs.transition_status(
                run.id,
                "completed",
                summary=pipeline_execution.summary,
                artifact_directory=str(pipeline_execution.run_dir),
                sample_count=len(persisted_samples),
            )
            return BenchmarkServiceResult(
                run=run,
                samples=tuple(persisted_samples),
                summary=pipeline_execution.summary,
            )
        except Exception as error:
            artifact_directory = (
                str(pipeline_execution.run_dir) if pipeline_execution is not None else run.artifact_directory
            )
            run = self._runs.transition_status(
                run.id,
                "failed",
                artifact_directory=artifact_directory,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            return BenchmarkServiceResult(run=run)

    def execute(
        self,
        *,
        endpoint_id: int,
        model_id: int,
        dataset_id: int,
        config: RunConfig,
    ) -> BenchmarkServiceResult:
        queued = self.enqueue(
            endpoint_id=endpoint_id,
            model_id=model_id,
            dataset_id=dataset_id,
            config=config,
            selected_sample_count=_planned_sample_count(config),
        )
        running = self._runs.transition_status(queued.run.id, "running")
        return self.execute_claimed(running)

    @staticmethod
    def _validate_active(
        endpoint: ProviderEndpointRecord,
        model: ModelRecord,
        dataset: DatasetRecord,
    ) -> None:
        if not endpoint.is_active:
            raise InactiveDependencyError("provider endpoint", endpoint.id)
        if not model.is_active:
            raise InactiveDependencyError("model", model.id)
        if not dataset.is_active:
            raise InactiveDependencyError("dataset", dataset.id)

    @staticmethod
    def _validate_identity(
        config: RunConfig,
        endpoint: ProviderEndpointRecord,
        model: ModelRecord,
        dataset: DatasetRecord,
    ) -> None:
        if len(config.models) != 1:
            raise RegistrationMismatchError("Registered execution requires exactly one configured model")
        configured_model = config.models[0]
        expected_provider = {
            "mock": "mock",
            "lm_studio_native": "lm_studio",
            "openai_compatible": "openai_compatible",
        }[endpoint.provider_type.value]
        mismatches: list[str] = []
        if model.endpoint_id != endpoint.id:
            mismatches.append("model.endpoint_id")
        if configured_model.provider != expected_provider:
            mismatches.append("model.provider")
        if configured_model.endpoint_alias != endpoint.name:
            mismatches.append("model.endpoint_alias")
        if configured_model.model_id != model.model_identifier:
            mismatches.append("model.model_id")
        if configured_model.base_url is not None and configured_model.base_url.rstrip("/") != endpoint.base_url.rstrip("/"):
            mismatches.append("model.base_url")
        if config.dataset.name != dataset.name:
            mismatches.append("dataset.name")
        if config.dataset.source != dataset.source_type:
            mismatches.append("dataset.source")
        if config.dataset.revision != dataset.revision:
            mismatches.append("dataset.revision")
        if config.dataset.split != dataset.split:
            mismatches.append("dataset.split")
        if config.dataset.source == "local":
            configured_source_uri = _normalized_path(config.dataset.path)
        elif config.dataset.source == "uploaded":
            configured_source_uri = config.dataset.storage_key
        else:
            configured_source_uri = config.dataset.name
        registered_source_uri = (
            _normalized_path(Path(dataset.source_uri))
            if dataset.source_type == "local"
            else dataset.source_uri
        )
        if configured_source_uri != registered_source_uri:
            mismatches.append("dataset.source_uri")
        if config.dataset.source == "uploaded":
            if config.dataset.adapter_type != dataset.adapter_type:
                mismatches.append("dataset.adapter_type")
            if config.dataset.checksum != dataset.checksum:
                mismatches.append("dataset.checksum")
            if config.dataset.license != dataset.license:
                mismatches.append("dataset.license")
        if mismatches:
            raise RegistrationMismatchError(
                "Configuration conflicts with selected registrations: " + ", ".join(mismatches)
            )


def _planned_sample_count(config: RunConfig) -> int:
    if config.dataset.sample_ids:
        return len(config.dataset.sample_ids)
    if config.dataset.sample_size is not None:
        return config.dataset.sample_size
    return 0


def _normalized_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(PurePosixPath(path.as_posix()))


def _to_sample_input(result: object) -> SampleResultCreate:
    from .models import BenchmarkResult

    if not isinstance(result, BenchmarkResult):
        raise TypeError("Pipeline returned an unsupported sample result type")
    return SampleResultCreate(
        sample_id=result.sample_id,
        category=result.category,
        correct_answer=result.correct_answer,
        parsed_answer=result.parsed_answer,
        raw_response=result.raw_response,
        request_status=result.request_status,
        parse_status=result.parse_status,
        evaluation_status=result.evaluation_status,
        is_correct=result.is_correct,
        input_tokens=result.input_tokens if result.input_tokens is not None else result.prompt_tokens,
        output_tokens=(
            result.total_output_tokens
            if result.total_output_tokens is not None
            else result.completion_tokens
        ),
        reasoning_tokens=result.reasoning_output_tokens,
        total_tokens=result.total_tokens,
        latency_ms=result.logical_request_latency_ms,
        ttft_ms=result.time_to_first_token_ms,
        throughput_tokens_per_second=result.tokens_per_second,
        error_type=result.error_type,
        provider_error_message=result.provider_error_message,
    )

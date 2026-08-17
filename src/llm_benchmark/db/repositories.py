"""Synchronous repository boundaries for benchmark registry persistence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, TypeVar

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .base import utc_now
from .errors import (
    InactiveDependencyError,
    InvalidStateTransitionError,
    RecordNotFoundError,
    RepositoryError,
    UniquenessConflictError,
)
from .models import (
    BenchmarkRun,
    ProviderEndpoint,
    ProviderType,
    ReasoningPolicy,
    RegisteredDataset,
    RegisteredModel,
    RunStatus,
    SampleResult,
)
from .records import (
    BenchmarkRunRecord,
    DatasetRecord,
    ModelRecord,
    ProviderEndpointRecord,
    SampleResultCreate,
    SampleResultRecord,
)
from .schemas import ModelCapabilities


_T = TypeVar("_T")
_UNSET = object()
_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[-_ ]?key|authorization|token|secret|password)\b\s*[:=]\s*[^\s,;]+"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_MAX_ERROR_MESSAGE_LENGTH = 512

_ALLOWED_TRANSITIONS = {
    RunStatus.QUEUED: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}


class _Repository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _require(session: Session, model_type: type[_T], record_id: int, entity: str) -> _T:
        record = session.get(model_type, record_id)
        if record is None:
            raise RecordNotFoundError(entity, record_id)
        return record


class ProviderEndpointRepository(_Repository):
    def create(
        self,
        *,
        name: str,
        provider_type: ProviderType | str,
        base_url: str,
        credential_env_var: str | None = None,
    ) -> ProviderEndpointRecord:
        _validate_credential_reference(credential_env_var)
        with self._session_factory() as session:
            try:
                with session.begin():
                    endpoint = ProviderEndpoint(
                        name=name,
                        provider_type=ProviderType(provider_type),
                        base_url=base_url,
                        credential_env_var=credential_env_var,
                    )
                    session.add(endpoint)
                    session.flush()
                    result = ProviderEndpointRecord.model_validate(endpoint)
            except IntegrityError as error:
                _raise_integrity_error(error, "provider endpoint", name)
        return result

    def get_by_id(self, endpoint_id: int) -> ProviderEndpointRecord:
        with self._session_factory() as session:
            return ProviderEndpointRecord.model_validate(
                self._require(session, ProviderEndpoint, endpoint_id, "provider endpoint")
            )

    def list_active(self) -> list[ProviderEndpointRecord]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(ProviderEndpoint).where(ProviderEndpoint.is_active.is_(True)).order_by(ProviderEndpoint.id)
            ).all()
            return [ProviderEndpointRecord.model_validate(row) for row in rows]

    def update(
        self,
        endpoint_id: int,
        *,
        name: str | None = None,
        provider_type: ProviderType | str | None = None,
        base_url: str | None = None,
        credential_env_var: str | None | object = _UNSET,
    ) -> ProviderEndpointRecord:
        if credential_env_var is not _UNSET:
            if credential_env_var is not None and not isinstance(credential_env_var, str):
                raise ValueError("credential_env_var must be a string or null")
            _validate_credential_reference(credential_env_var)
        with self._session_factory() as session:
            try:
                with session.begin():
                    endpoint = self._require(session, ProviderEndpoint, endpoint_id, "provider endpoint")
                    if name is not None:
                        endpoint.name = name
                    if provider_type is not None:
                        endpoint.provider_type = ProviderType(provider_type)
                    if base_url is not None:
                        endpoint.base_url = base_url
                    if credential_env_var is not _UNSET:
                        endpoint.credential_env_var = credential_env_var  # type: ignore[assignment]
                    session.flush()
                    result = ProviderEndpointRecord.model_validate(endpoint)
            except IntegrityError as error:
                _raise_integrity_error(error, "provider endpoint", name or str(endpoint_id))
        return result

    def soft_delete(self, endpoint_id: int) -> ProviderEndpointRecord:
        with self._session_factory() as session, session.begin():
            endpoint = self._require(session, ProviderEndpoint, endpoint_id, "provider endpoint")
            endpoint.is_active = False
            session.flush()
            return ProviderEndpointRecord.model_validate(endpoint)


class ModelRepository(_Repository):
    def create(
        self,
        *,
        name: str,
        model_identifier: str,
        endpoint_id: int,
        reasoning_policy: ReasoningPolicy | str,
        capabilities: ModelCapabilities | dict[str, Any] | None = None,
        default_generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRecord:
        capabilities_json = _validated_capabilities(capabilities)
        with self._session_factory() as session:
            try:
                with session.begin():
                    endpoint = self._require(session, ProviderEndpoint, endpoint_id, "provider endpoint")
                    if not endpoint.is_active:
                        raise InactiveDependencyError("provider endpoint", endpoint_id)
                    model = RegisteredModel(
                        name=name,
                        model_identifier=model_identifier,
                        endpoint_id=endpoint_id,
                        reasoning_policy=ReasoningPolicy(reasoning_policy),
                        capabilities_json=capabilities_json,
                        default_generation_config_json=default_generation_config or {},
                        metadata_json=metadata or {},
                    )
                    session.add(model)
                    session.flush()
                    result = ModelRecord.model_validate(model)
            except IntegrityError as error:
                _raise_integrity_error(error, "model", f"endpoint={endpoint_id}, model={model_identifier}")
        return result

    def get_by_id(self, model_id: int) -> ModelRecord:
        with self._session_factory() as session:
            return ModelRecord.model_validate(self._require(session, RegisteredModel, model_id, "model"))

    def list_active(self) -> list[ModelRecord]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RegisteredModel).where(RegisteredModel.is_active.is_(True)).order_by(RegisteredModel.id)
            ).all()
            return [ModelRecord.model_validate(row) for row in rows]

    def update(
        self,
        model_id: int,
        *,
        name: str | None = None,
        reasoning_policy: ReasoningPolicy | str | None = None,
        capabilities: ModelCapabilities | dict[str, Any] | None = None,
        default_generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRecord:
        capabilities_json = _validated_capabilities(capabilities) if capabilities is not None else None
        with self._session_factory() as session, session.begin():
            model = self._require(session, RegisteredModel, model_id, "model")
            if name is not None:
                model.name = name
            if reasoning_policy is not None:
                model.reasoning_policy = ReasoningPolicy(reasoning_policy)
            if capabilities_json is not None:
                model.capabilities_json = capabilities_json
            if default_generation_config is not None:
                model.default_generation_config_json = default_generation_config
            if metadata is not None:
                model.metadata_json = metadata
            session.flush()
            return ModelRecord.model_validate(model)

    def soft_delete(self, model_id: int) -> ModelRecord:
        with self._session_factory() as session, session.begin():
            model = self._require(session, RegisteredModel, model_id, "model")
            model.is_active = False
            session.flush()
            return ModelRecord.model_validate(model)


class DatasetRepository(_Repository):
    def create(
        self,
        *,
        name: str,
        source_type: str,
        source_uri: str,
        revision: str | None,
        split: str,
        task_type: str,
        adapter_type: str,
        license: str | None = None,
        checksum: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRecord:
        identity = _dataset_identity_query(
            source_type=source_type,
            source_uri=source_uri,
            revision=revision,
            split=split,
            task_type=task_type,
            adapter_type=adapter_type,
        )
        identity_text = f"{source_type}:{source_uri}@{revision!r}/{split}/{task_type}/{adapter_type}"
        with self._session_factory() as session:
            try:
                with session.begin():
                    if session.scalar(identity) is not None:
                        raise UniquenessConflictError("dataset", identity_text)
                    dataset = RegisteredDataset(
                        name=name,
                        source_type=source_type,
                        source_uri=source_uri,
                        revision=revision,
                        split=split,
                        task_type=task_type,
                        adapter_type=adapter_type,
                        license=license,
                        checksum=checksum,
                        metadata_json=metadata or {},
                    )
                    session.add(dataset)
                    session.flush()
                    result = DatasetRecord.model_validate(dataset)
            except IntegrityError as error:
                _raise_integrity_error(error, "dataset", identity_text)
        return result

    def get_by_id(self, dataset_id: int) -> DatasetRecord:
        with self._session_factory() as session:
            return DatasetRecord.model_validate(self._require(session, RegisteredDataset, dataset_id, "dataset"))

    def list_active(self) -> list[DatasetRecord]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RegisteredDataset)
                .where(RegisteredDataset.is_active.is_(True))
                .order_by(RegisteredDataset.id)
            ).all()
            return [DatasetRecord.model_validate(row) for row in rows]

    def update(
        self,
        dataset_id: int,
        *,
        name: str | None = None,
        license: str | None | object = _UNSET,
        checksum: str | None | object = _UNSET,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRecord:
        with self._session_factory() as session, session.begin():
            dataset = self._require(session, RegisteredDataset, dataset_id, "dataset")
            if name is not None:
                dataset.name = name
            if license is not _UNSET:
                dataset.license = license  # type: ignore[assignment]
            if checksum is not _UNSET:
                dataset.checksum = checksum  # type: ignore[assignment]
            if metadata is not None:
                dataset.metadata_json = metadata
            session.flush()
            return DatasetRecord.model_validate(dataset)

    def soft_delete(self, dataset_id: int) -> DatasetRecord:
        with self._session_factory() as session, session.begin():
            dataset = self._require(session, RegisteredDataset, dataset_id, "dataset")
            dataset.is_active = False
            session.flush()
            return DatasetRecord.model_validate(dataset)


class BenchmarkRunRepository(_Repository):
    def create_queued(
        self,
        *,
        experiment_name: str,
        model_id: int,
        dataset_id: int,
        resolved_config: dict[str, Any],
        config_hash: str,
        seed: int,
        sample_count: int,
        artifact_directory: str,
    ) -> BenchmarkRunRecord:
        with self._session_factory() as session, session.begin():
            model = self._require(session, RegisteredModel, model_id, "model")
            dataset = self._require(session, RegisteredDataset, dataset_id, "dataset")
            endpoint = self._require(session, ProviderEndpoint, model.endpoint_id, "provider endpoint")
            if not endpoint.is_active:
                raise InactiveDependencyError("provider endpoint", endpoint.id)
            if not model.is_active:
                raise InactiveDependencyError("model", model.id)
            if not dataset.is_active:
                raise InactiveDependencyError("dataset", dataset.id)
            run = BenchmarkRun(
                experiment_name=experiment_name,
                model_id=model_id,
                dataset_id=dataset_id,
                status=RunStatus.QUEUED,
                resolved_config_json=resolved_config,
                config_hash=config_hash,
                seed=seed,
                sample_count=sample_count,
                artifact_directory=artifact_directory,
            )
            session.add(run)
            session.flush()
            return BenchmarkRunRecord.model_validate(run)

    def get_by_id(self, run_id: int) -> BenchmarkRunRecord:
        with self._session_factory() as session:
            return BenchmarkRunRecord.model_validate(self._require(session, BenchmarkRun, run_id, "benchmark run"))

    def list_runs(self) -> list[BenchmarkRunRecord]:
        with self._session_factory() as session:
            rows = session.scalars(select(BenchmarkRun).order_by(BenchmarkRun.id)).all()
            return [BenchmarkRunRecord.model_validate(row) for row in rows]

    def transition_status(
        self,
        run_id: int,
        requested_status: RunStatus | str,
        *,
        summary: dict[str, Any] | None = None,
        artifact_directory: str | None = None,
        sample_count: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> BenchmarkRunRecord:
        target = RunStatus(requested_status)
        with self._session_factory() as session, session.begin():
            run = self._require(session, BenchmarkRun, run_id, "benchmark run")
            current = RunStatus(run.status)
            if target not in _ALLOWED_TRANSITIONS[current]:
                raise InvalidStateTransitionError(current.value, target.value)
            now = utc_now()
            run.status = target
            if target is RunStatus.RUNNING:
                run.started_at = now
            if target in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                run.completed_at = now
            if summary is not None:
                run.summary_json = summary
            if artifact_directory is not None:
                run.artifact_directory = artifact_directory
            if sample_count is not None:
                run.sample_count = sample_count
            if target is RunStatus.FAILED:
                run.error_type = error_type
                run.error_message = _sanitize_error_message(error_message)
            session.flush()
            return BenchmarkRunRecord.model_validate(run)


class SampleResultRepository(_Repository):
    def add_one(self, run_id: int, sample: SampleResultCreate) -> SampleResultRecord:
        return self.add_many(run_id, [sample])[0]

    def add_many(self, run_id: int, samples: Iterable[SampleResultCreate]) -> list[SampleResultRecord]:
        inputs = list(samples)
        with self._session_factory() as session:
            try:
                with session.begin():
                    self._require(session, BenchmarkRun, run_id, "benchmark run")
                    payloads = [item.model_dump() for item in inputs]
                    for payload in payloads:
                        payload["provider_error_message"] = _sanitize_error_message(
                            payload["provider_error_message"]
                        )
                    rows = [SampleResult(run_id=run_id, **payload) for payload in payloads]
                    session.add_all(rows)
                    session.flush()
                    results = [SampleResultRecord.model_validate(row) for row in rows]
            except IntegrityError as error:
                _raise_integrity_error(error, "sample result", f"run={run_id}")
        return results

    def list_by_run_id(self, run_id: int) -> list[SampleResultRecord]:
        with self._session_factory() as session:
            self._require(session, BenchmarkRun, run_id, "benchmark run")
            rows = session.scalars(
                select(SampleResult).where(SampleResult.run_id == run_id).order_by(SampleResult.id)
            ).all()
            return [SampleResultRecord.model_validate(row) for row in rows]


def _validated_capabilities(
    capabilities: ModelCapabilities | dict[str, Any] | None,
) -> dict[str, Any]:
    if capabilities is None:
        validated = ModelCapabilities()
    elif isinstance(capabilities, ModelCapabilities):
        validated = capabilities
    else:
        validated = ModelCapabilities.model_validate(capabilities)
    return validated.model_dump(exclude_none=True)


def _dataset_identity_query(
    *,
    source_type: str,
    source_uri: str,
    revision: str | None,
    split: str,
    task_type: str,
    adapter_type: str,
) -> Select[tuple[RegisteredDataset]]:
    revision_clause = (
        RegisteredDataset.revision.is_(None)
        if revision is None
        else RegisteredDataset.revision == revision
    )
    return select(RegisteredDataset).where(
        RegisteredDataset.source_type == source_type,
        RegisteredDataset.source_uri == source_uri,
        revision_clause,
        RegisteredDataset.split == split,
        RegisteredDataset.task_type == task_type,
        RegisteredDataset.adapter_type == adapter_type,
    )


def _validate_credential_reference(value: str | None) -> None:
    if value is not None and not _ENV_VAR_PATTERN.fullmatch(value):
        raise ValueError("credential_env_var must be an environment-variable name, not a credential value")


def _sanitize_error_message(message: str | None) -> str | None:
    if message is None:
        return None
    sanitized = _BEARER_PATTERN.sub("Bearer [REDACTED]", message)
    sanitized = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", sanitized)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) > _MAX_ERROR_MESSAGE_LENGTH:
        sanitized = sanitized[: _MAX_ERROR_MESSAGE_LENGTH - 3] + "..."
    return sanitized


def _raise_integrity_error(error: IntegrityError, entity: str, identity: str) -> None:
    message = str(error.orig).lower()
    if "unique" in message or "duplicate" in message:
        raise UniquenessConflictError(entity, identity) from error
    raise RepositoryError(f"Unable to persist {entity} because a database constraint failed") from error

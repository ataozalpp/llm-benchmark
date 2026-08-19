"""Resolve registered resources into a safe immutable benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from .config import DatasetConfig, EvaluationConfig, ModelConfig, RunConfig
from .db.errors import InactiveDependencyError
from .db.records import DatasetRecord, ModelRecord, ProviderEndpointRecord


class RunConfigResolutionError(Exception):
    """Raised when registrations cannot produce a supported safe RunConfig."""


class EndpointReader(Protocol):
    def get_by_id(self, endpoint_id: int) -> ProviderEndpointRecord: ...


class ModelReader(Protocol):
    def get_by_id(self, model_id: int) -> ModelRecord: ...


class DatasetReader(Protocol):
    def get_by_id(self, dataset_id: int) -> DatasetRecord: ...


@dataclass(frozen=True)
class RunConfigRequest:
    experiment_name: str
    model_id: int
    dataset_id: int
    seed: int = 42
    profile: Literal["smoke", "poc", "full"] = "smoke"
    sample_size: int | None = None
    sample_ids: tuple[str, ...] = ()
    category_filter: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedRegisteredRun:
    endpoint_id: int
    model_id: int
    dataset_id: int
    config: RunConfig


_SERVER_CONTROLLED_MODEL_FIELDS = {
    "provider",
    "endpoint_alias",
    "model_id",
    "base_url",
    "credential_env_var",
}
_SUPPORTED_DATASET_ADAPTERS = {
    ("local", "local_jsonl"),
    ("huggingface", "huggingface"),
    ("huggingface", "mmlu_pro"),
}


class RegisteredRunConfigResolver:
    def __init__(
        self,
        *,
        endpoints: EndpointReader,
        models: ModelReader,
        datasets: DatasetReader,
        output_root: Path,
    ) -> None:
        self._endpoints = endpoints
        self._models = models
        self._datasets = datasets
        self._output_root = output_root

    def resolve(self, request: RunConfigRequest) -> ResolvedRegisteredRun:
        model = self._models.get_by_id(request.model_id)
        endpoint = self._endpoints.get_by_id(model.endpoint_id)
        dataset = self._datasets.get_by_id(request.dataset_id)
        self._validate_active(endpoint, model, dataset)
        self._validate_dataset(dataset)

        defaults = dict(model.default_generation_config_json)
        forbidden = sorted(_SERVER_CONTROLLED_MODEL_FIELDS & defaults.keys())
        if forbidden:
            raise RunConfigResolutionError(
                "Registered default generation config contains server-controlled fields: "
                + ", ".join(forbidden)
            )
        provider = {
            "mock": "mock",
            "lm_studio_native": "lm_studio",
            "openai_compatible": "openai_compatible",
        }[endpoint.provider_type.value]
        model_values: dict[str, Any] = {
            **defaults,
            "provider": provider,
            "endpoint_alias": endpoint.name,
            "model_id": model.model_identifier,
            "base_url": endpoint.base_url,
        }
        if endpoint.credential_env_var is not None:
            model_values["credential_env_var"] = endpoint.credential_env_var

        dataset_values: dict[str, Any] = {
            "source": dataset.source_type,
            "name": dataset.name,
            "revision": dataset.revision,
            "split": dataset.split,
            "profile": request.profile,
            "sample_size": request.sample_size,
            "sample_ids": list(request.sample_ids),
            "category_filter": list(request.category_filter),
        }
        if dataset.source_type == "local":
            dataset_values["path"] = Path(dataset.source_uri)

        try:
            config = RunConfig(
                schema_version=1,
                experiment_name=request.experiment_name,
                seed=request.seed,
                output_dir=self._output_root,
                dataset=DatasetConfig.model_validate(dataset_values),
                models=[ModelConfig.model_validate(model_values)],
                evaluation=EvaluationConfig(),
            )
        except (ValidationError, ValueError) as error:
            raise RunConfigResolutionError("Registered resources cannot produce a valid run configuration") from error
        return ResolvedRegisteredRun(
            endpoint_id=endpoint.id,
            model_id=model.id,
            dataset_id=dataset.id,
            config=config,
        )

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
    def _validate_dataset(dataset: DatasetRecord) -> None:
        if dataset.task_type != "multiple_choice":
            raise RunConfigResolutionError("Only multiple-choice datasets are supported")
        if (dataset.source_type, dataset.adapter_type) not in _SUPPORTED_DATASET_ADAPTERS:
            raise RunConfigResolutionError("Dataset source and adapter are not supported")

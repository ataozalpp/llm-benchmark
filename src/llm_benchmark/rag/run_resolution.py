"""Build a RAG run configuration from registered resources.

Clients choose registered model and dataset IDs plus evaluation settings and
an embedding profile name. Endpoint URLs, model identifiers, storage keys,
embedding model revisions and output paths come from the server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ..dataset_ingestion import RAG_SUITE_ADAPTER_TYPE
from ..db.records import DatasetRecord, ModelRecord, ProviderEndpointRecord
from ..retrieval.run_config import DenseEmbeddingConfig
from ..run_resolution import RunConfigResolutionError
from .run_config import RagRunConfig

DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 60.0


class RagRunRequestError(ValueError):
    """Raised when client-selected RAG settings are invalid together."""


class EndpointReader(Protocol):
    def get_by_id(self, endpoint_id: int) -> ProviderEndpointRecord: ...


class ModelReader(Protocol):
    def get_by_id(self, model_id: int) -> ModelRecord: ...


class DatasetReader(Protocol):
    def get_by_id(self, dataset_id: int) -> DatasetRecord: ...


@dataclass(frozen=True)
class RagRunRequest:
    experiment_name: str
    model_id: int
    dataset_id: int
    retriever: str
    k: int
    max_context_chars: int
    embedding_profile: str | None = None
    hybrid: dict[str, int] | None = None
    prompt_version: str | None = None
    structured_output: str = "none"


@dataclass(frozen=True)
class ResolvedRagRun:
    experiment_name: str
    endpoint: ProviderEndpointRecord
    model: ModelRecord
    dataset: DatasetRecord
    config: RagRunConfig


class RegisteredRagRunConfigResolver:
    def __init__(
        self,
        *,
        endpoints: EndpointReader,
        models: ModelReader,
        datasets: DatasetReader,
        output_root: Path,
        embedding_profiles: dict[str, DenseEmbeddingConfig],
    ) -> None:
        self._endpoints = endpoints
        self._models = models
        self._datasets = datasets
        self._output_root = output_root
        self._embedding_profiles = embedding_profiles

    def resolve(self, request: RagRunRequest) -> ResolvedRagRun:
        model = self._models.get_by_id(request.model_id)
        endpoint = self._endpoints.get_by_id(model.endpoint_id)
        dataset = self._datasets.get_by_id(request.dataset_id)
        validate_registrations(endpoint, model, dataset)

        values: dict[str, Any] = {
            "schema_version": 1,
            # The opaque storage key; the worker verifies and resolves it.
            "suite_path": dataset.source_uri,
            "retriever": request.retriever,
            "k": request.k,
            "max_context_chars": request.max_context_chars,
            "structured_output": request.structured_output,
            "output_dir": str(self._output_root / "rag"),
        }
        if request.prompt_version is not None:
            values["prompt_version"] = request.prompt_version
        if request.hybrid is not None:
            values["hybrid"] = request.hybrid
        if request.embedding_profile is not None:
            profile = self._embedding_profiles.get(request.embedding_profile)
            if profile is None:
                raise RagRunRequestError("Unknown embedding profile.")
            values["embedding"] = profile.model_dump()

        if endpoint.provider_type.value == "mock":
            values["provider"] = "synthetic"
        else:
            values["provider"] = "openai_compatible"
            values["model"] = _model_values(endpoint, model)

        try:
            config = RagRunConfig.model_validate(values)
        except ValidationError:
            raise RagRunRequestError("RAG run settings are invalid.") from None

        return ResolvedRagRun(
            experiment_name=request.experiment_name,
            endpoint=endpoint,
            model=model,
            dataset=dataset,
            config=config,
        )


def validate_registrations(
    endpoint: ProviderEndpointRecord, model: ModelRecord, dataset: DatasetRecord
) -> None:
    if not (endpoint.is_active and model.is_active and dataset.is_active):
        raise RunConfigResolutionError("Registered resources must be active")
    if model.endpoint_id != endpoint.id:
        raise RunConfigResolutionError("Model does not belong to the endpoint")
    if endpoint.provider_type.value not in {"mock", "openai_compatible"}:
        raise RunConfigResolutionError(
            "RAG runs support mock and OpenAI-compatible endpoints only"
        )
    if endpoint.credential_env_var is not None:
        raise RunConfigResolutionError("RAG generation does not support credentials")
    if (
        dataset.task_type != "rag"
        or dataset.source_type != "uploaded"
        or dataset.adapter_type != RAG_SUITE_ADAPTER_TYPE
        or dataset.checksum is None
    ):
        raise RunConfigResolutionError("Dataset is not an uploaded RAG suite")


def _model_values(
    endpoint: ProviderEndpointRecord, model: ModelRecord
) -> dict[str, Any]:
    defaults = model.default_generation_config_json
    temperature = defaults.get("temperature", 0)
    if temperature != 0:
        raise RunConfigResolutionError("RAG generation uses temperature zero")
    return {
        "provider": "openai_compatible",
        "endpoint_alias": endpoint.name,
        "model_id": model.model_identifier,
        "base_url": endpoint.base_url,
        "temperature": 0,
        "max_output_tokens": defaults.get(
            "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS
        ),
        "timeout_seconds": defaults.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
    }

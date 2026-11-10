"""FastAPI dependency adapters with lazy default repository construction."""

from __future__ import annotations

from fastapi import Request

from llm_benchmark.application import BenchmarkApplicationService
from llm_benchmark.dataset_ingestion import (
    DatasetIngestionService,
)
from llm_benchmark.dataset_storage import (
    LocalDatasetStorage,
)
from llm_benchmark.db.registry import RegistryRepositories
from llm_benchmark.rag.application import RagApplicationService
from llm_benchmark.rag.run_preflight import RagRunPreflightService
from llm_benchmark.rag.run_resolution import RegisteredRagRunConfigResolver
from llm_benchmark.run_preflight import RunPreflightService
from llm_benchmark.run_resolution import RegisteredRunConfigResolver


def get_registry(request: Request) -> RegistryRepositories:
    registry = request.app.state.registry
    if registry is not None:
        return registry
    with request.app.state.registry_lock:
        registry = request.app.state.registry
        if registry is None:
            registry = request.app.state.registry_factory()
            request.app.state.registry = registry
    return registry


def get_dataset_ingestion_service(
    request: Request,
) -> DatasetIngestionService:
    registry = get_registry(request)
    storage = LocalDatasetStorage(request.app.state.dataset_storage_root)
    return DatasetIngestionService(
        storage=storage,
        datasets=registry.datasets,
    )


def get_benchmark_service(request: Request) -> BenchmarkApplicationService:
    registry = get_registry(request)
    return BenchmarkApplicationService(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        runs=registry.runs,
        samples=registry.samples,
        executor=request.app.state.benchmark_executor,
    )


def get_run_config_resolver(request: Request) -> RegisteredRunConfigResolver:
    registry = get_registry(request)
    return RegisteredRunConfigResolver(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        output_root=request.app.state.run_output_root,
    )


def get_run_preflight_service(request: Request) -> RunPreflightService:
    return RunPreflightService(
        policy=request.app.state.run_guardrail_policy,
        loader=request.app.state.dataset_loader,
    )


def get_rag_run_resolver(request: Request) -> RegisteredRagRunConfigResolver:
    registry = get_registry(request)
    return RegisteredRagRunConfigResolver(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        output_root=request.app.state.run_output_root,
        embedding_profiles=request.app.state.embedding_profiles,
    )


def get_rag_preflight_service(request: Request) -> RagRunPreflightService:
    return RagRunPreflightService(
        policy=request.app.state.rag_guardrail_policy,
        storage=LocalDatasetStorage(request.app.state.dataset_storage_root),
        embeddings_check=request.app.state.embeddings_check,
    )


def get_rag_service(request: Request) -> RagApplicationService:
    registry = get_registry(request)
    return RagApplicationService(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        runs=registry.runs,
        rag_cases=registry.rag_cases,
        storage=LocalDatasetStorage(request.app.state.dataset_storage_root),
    )

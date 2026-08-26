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
    storage = LocalDatasetStorage(
        request.app.state.dataset_storage_root
    )
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

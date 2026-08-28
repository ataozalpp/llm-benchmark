"""Thin benchmark-run API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from llm_benchmark.application import BenchmarkApplicationService
from llm_benchmark.db.registry import RegistryRepositories
from llm_benchmark.run_preflight import RunPreflightService
from llm_benchmark.run_resolution import RegisteredRunConfigResolver, RunConfigRequest

from .app_dependencies import (
    get_benchmark_service,
    get_registry,
    get_run_config_resolver,
    get_run_preflight_service,
)
from .schemas import RunCreate, RunResponse, SampleResultResponse

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunCreate,
    resolver: RegisteredRunConfigResolver = Depends(get_run_config_resolver),
    preflight: RunPreflightService = Depends(get_run_preflight_service),
    service: BenchmarkApplicationService = Depends(get_benchmark_service),
) -> RunResponse:
    resolved = resolver.resolve(
        RunConfigRequest(
            experiment_name=payload.experiment_name,
            model_id=payload.model_id,
            dataset_id=payload.dataset_id,
            seed=payload.seed,
            profile=payload.profile,
            sample_size=payload.sample_size,
            sample_ids=tuple(payload.sample_ids),
            category_filter=tuple(payload.category_filter),
        )
    )
    preflight_result = preflight.preflight(resolved.config)

    result = service.enqueue(
        endpoint_id=resolved.endpoint_id,
        model_id=resolved.model_id,
        dataset_id=resolved.dataset_id,
        config=resolved.config,
        selected_sample_count=preflight_result.selected_sample_count,
    )
    return RunResponse.from_record(result.run)


@router.get("", response_model=list[RunResponse])
def list_runs(registry: RegistryRepositories = Depends(get_registry)) -> list[RunResponse]:
    return [RunResponse.from_record(record) for record in registry.runs.list_runs()]


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: int, registry: RegistryRepositories = Depends(get_registry)) -> RunResponse:
    return RunResponse.from_record(registry.runs.get_by_id(run_id))


@router.get("/{run_id}/results", response_model=list[SampleResultResponse])
def list_run_results(
    run_id: int,
    registry: RegistryRepositories = Depends(get_registry),
) -> list[SampleResultResponse]:
    return [
        SampleResultResponse.from_record(record)
        for record in registry.samples.list_by_run_id(run_id)
    ]

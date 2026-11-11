"""Thin RAG run API routes.

Runs are listed and polled through the shared ``/runs`` routes; RAG case
results have their own shape and route.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from llm_benchmark.db.registry import RegistryRepositories
from llm_benchmark.rag.application import RagApplicationService
from llm_benchmark.rag.run_preflight import RagRunPreflightService
from llm_benchmark.rag.run_resolution import (
    RagRunRequest,
    RegisteredRagRunConfigResolver,
)

from .app_dependencies import (
    get_rag_preflight_service,
    get_rag_run_resolver,
    get_rag_service,
    get_registry,
)
from .schemas import RagCaseResultResponse, RagRunCreate, RunResponse

router = APIRouter(prefix="/rag-runs", tags=["rag-runs"])


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_rag_run(
    payload: RagRunCreate,
    resolver: RegisteredRagRunConfigResolver = Depends(get_rag_run_resolver),
    preflight: RagRunPreflightService = Depends(get_rag_preflight_service),
    service: RagApplicationService = Depends(get_rag_service),
) -> RunResponse:
    resolved = resolver.resolve(
        RagRunRequest(
            experiment_name=payload.experiment_name,
            model_id=payload.model_id,
            dataset_id=payload.dataset_id,
            retriever=payload.retriever,
            k=payload.k,
            max_context_chars=payload.max_context_chars,
            embedding_profile=payload.embedding_profile,
            hybrid=None if payload.hybrid is None else payload.hybrid.model_dump(),
            prompt_version=payload.prompt_version,
            structured_output=payload.structured_output,
        )
    )
    result = preflight.preflight(resolved.config, resolved.dataset)
    return RunResponse.from_record(
        service.enqueue(resolved, result.scenario_count).run
    )


@router.get("/{run_id}/results", response_model=list[RagCaseResultResponse])
def list_rag_run_results(
    run_id: int, registry: RegistryRepositories = Depends(get_registry)
) -> list[RagCaseResultResponse]:
    run = registry.runs.get_by_id(run_id)
    if run.task_type.value != "rag":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run is not a RAG run; use /api/v1/runs/{run_id}/results",
        )
    return [
        RagCaseResultResponse.from_record(record)
        for record in registry.rag_cases.list_by_run_id(run_id)
    ]

"""FastAPI application factory for the registry API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_benchmark.application import RegistrationMismatchError
from llm_benchmark.config import RunConfig
from llm_benchmark.db.errors import (
    InactiveDependencyError,
    RecordNotFoundError,
    RepositoryError,
    UniquenessConflictError,
)
from llm_benchmark.db.registry import RegistryRepositories, create_registry_repositories
from llm_benchmark.run_preflight import (
    DatasetPreflightError,
    RunApiGuardrailPolicy,
    RunGuardrailViolationError,
    RunSelectionValidationError,
)
from llm_benchmark.run_resolution import RunConfigResolutionError
from llm_benchmark.runner import PipelineExecution, execute_benchmark


def create_app(
    registry: RegistryRepositories | None = None,
    *,
    registry_factory: Callable[[], RegistryRepositories] = create_registry_repositories,
    benchmark_executor: Callable[[RunConfig], PipelineExecution] = execute_benchmark,
    run_output_root: Path = Path("outputs/api"),
    run_guardrail_policy: RunApiGuardrailPolicy | None = None,
) -> FastAPI:
    app = FastAPI(title="LLM Benchmark Registry API", version="1.0.0")
    app.state.registry = registry
    app.state.registry_factory = registry_factory
    app.state.registry_lock = Lock()
    app.state.benchmark_executor = benchmark_executor
    app.state.run_output_root = run_output_root
    app.state.run_guardrail_policy = run_guardrail_policy or RunApiGuardrailPolicy()
    _install_exception_handlers(app)

    from .routes import router

    app.include_router(router)
    return app


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation(_: Request, error: RequestValidationError) -> JSONResponse:
        safe_errors = [
            {
                "loc": list(item.get("loc", ())),
                "msg": item.get("msg", "Invalid request"),
                "type": item.get("type", "validation_error"),
            }
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.exception_handler(RecordNotFoundError)
    async def record_not_found(_: Request, error: RecordNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": f"{error.entity} record was not found", "record_id": error.record_id},
        )

    @app.exception_handler(UniquenessConflictError)
    async def uniqueness_conflict(_: Request, __: UniquenessConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Resource already exists"})

    @app.exception_handler(InactiveDependencyError)
    async def inactive_dependency(_: Request, error: InactiveDependencyError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": f"{error.entity} is inactive", "record_id": error.record_id},
        )

    @app.exception_handler(RunConfigResolutionError)
    async def run_config_resolution(_: Request, __: RunConfigResolutionError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Registered run configuration is invalid"})

    @app.exception_handler(RegistrationMismatchError)
    async def registration_mismatch(_: Request, __: RegistrationMismatchError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Run configuration conflicts with registrations"})

    @app.exception_handler(RunGuardrailViolationError)
    async def run_guardrail_violation(_: Request, __: RunGuardrailViolationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Run request exceeds the synchronous API limits",
                "code": "run_guardrail_violation",
            },
        )

    @app.exception_handler(RunSelectionValidationError)
    async def run_selection_invalid(_: Request, __: RunSelectionValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Requested dataset selection is invalid",
                "code": "invalid_dataset_selection",
            },
        )

    @app.exception_handler(DatasetPreflightError)
    async def dataset_preflight_failed(_: Request, __: DatasetPreflightError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Registered dataset is unavailable for preflight",
                "code": "dataset_preflight_failed",
            },
        )

    @app.exception_handler(RepositoryError)
    async def repository_failure(_: Request, __: RepositoryError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Registry persistence failed"})

    @app.exception_handler(Exception)
    async def unexpected_failure(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

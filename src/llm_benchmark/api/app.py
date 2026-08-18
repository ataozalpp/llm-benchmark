"""FastAPI application factory for the registry API."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_benchmark.db.errors import (
    InactiveDependencyError,
    RecordNotFoundError,
    RepositoryError,
    UniquenessConflictError,
)
from llm_benchmark.db.registry import RegistryRepositories, create_registry_repositories


def create_app(
    registry: RegistryRepositories | None = None,
    *,
    registry_factory: Callable[[], RegistryRepositories] = create_registry_repositories,
) -> FastAPI:
    app = FastAPI(title="LLM Benchmark Registry API", version="1.0.0")
    app.state.registry = registry
    app.state.registry_factory = registry_factory
    app.state.registry_lock = Lock()
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

    @app.exception_handler(RepositoryError)
    async def repository_failure(_: Request, __: RepositoryError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Registry persistence failed"})

    @app.exception_handler(Exception)
    async def unexpected_failure(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

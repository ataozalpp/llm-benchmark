"""FastAPI dependency adapters with lazy default repository construction."""

from __future__ import annotations

from fastapi import Request

from llm_benchmark.db.registry import RegistryRepositories


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

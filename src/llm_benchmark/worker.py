"""Single-process polling worker for queued benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Protocol

from .application import BenchmarkApplicationService
from .config import RunConfig
from .dataset_storage import LocalDatasetStorage
from .datasets import DatasetLoader
from .db.records import BenchmarkRunRecord
from .db.registry import RegistryRepositories, create_registry_repositories
from .rag.application import RagApplicationService
from .rag.run_config import LoadedRagRunConfig
from .rag.run_service import RagPipelineExecution, execute_rag_benchmark
from .runner import PipelineExecution, execute_benchmark


class QueuedRunClaimer(Protocol):
    def claim_next_queued(self) -> BenchmarkRunRecord | None: ...


class ClaimedRunExecutor(Protocol):
    def execute_claimed(self, run: BenchmarkRunRecord) -> object: ...


class TaskTypeDispatcher:
    """Route a claimed run to the service for its task type."""

    def __init__(
        self,
        services: dict[str, ClaimedRunExecutor],
        *,
        runs: object | None = None,
    ) -> None:
        self._services = services
        self._runs = runs

    def execute_claimed(self, run: BenchmarkRunRecord) -> object:
        service = self._services.get(run.task_type.value)
        if service is None:
            if self._runs is None:
                raise ValueError(f"Unsupported task type {run.task_type.value!r}")
            # Fail the claimed run instead of leaving it running forever.
            return self._runs.transition_status(
                run.id,
                "failed",
                error_type="UnsupportedTaskType",
                error_message=f"No executor for task type {run.task_type.value!r}",
            )
        return service.execute_claimed(run)


class BenchmarkWorker:
    """Claim and execute at most one queued run per call."""

    def __init__(self, *, runs: QueuedRunClaimer, service: ClaimedRunExecutor) -> None:
        self._runs = runs
        self._service = service

    def run_once(self) -> bool:
        run = self._runs.claim_next_queued()
        if run is None:
            return False
        self._service.execute_claimed(run)
        return True


def create_worker(
    *,
    registry_factory: Callable[[], RegistryRepositories] = create_registry_repositories,
    dataset_storage_root: Path = Path("runtime/datasets"),
    executor: Callable[[RunConfig], PipelineExecution] | None = None,
    rag_executor: Callable[[LoadedRagRunConfig], RagPipelineExecution] = (
        execute_rag_benchmark
    ),
) -> BenchmarkWorker:
    """Construct the default worker lazily when the entry point is invoked."""

    registry = registry_factory()
    storage = LocalDatasetStorage(dataset_storage_root)
    dataset_loader = DatasetLoader(storage).load
    benchmark_executor = executor or partial(
        execute_benchmark,
        dataset_loader=dataset_loader,
    )
    service = BenchmarkApplicationService(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        runs=registry.runs,
        samples=registry.samples,
        executor=benchmark_executor,
    )
    rag_service = RagApplicationService(
        endpoints=registry.endpoints,
        models=registry.models,
        datasets=registry.datasets,
        runs=registry.runs,
        rag_cases=registry.rag_cases,
        storage=storage,
        executor=rag_executor,
    )
    dispatcher = TaskTypeDispatcher(
        {"multiple_choice": service, "rag": rag_service}, runs=registry.runs
    )
    return BenchmarkWorker(runs=registry.runs, service=dispatcher)


def run_polling_loop(
    worker: BenchmarkWorker,
    *,
    idle_sleep_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Process queued runs serially until interrupted."""

    if idle_sleep_seconds <= 0:
        raise ValueError("idle_sleep_seconds must be positive")
    try:
        while True:
            if not worker.run_once():
                sleep(idle_sleep_seconds)
    except KeyboardInterrupt:
        return


def main() -> None:
    run_polling_loop(create_worker())


if __name__ == "__main__":
    main()

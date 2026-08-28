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
from .runner import PipelineExecution, execute_benchmark


class QueuedRunClaimer(Protocol):
    def claim_next_queued(self) -> BenchmarkRunRecord | None: ...


class ClaimedRunExecutor(Protocol):
    def execute_claimed(self, run: BenchmarkRunRecord) -> object: ...


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
) -> BenchmarkWorker:
    """Construct the default worker lazily when the entry point is invoked."""

    registry = registry_factory()
    dataset_loader = DatasetLoader(LocalDatasetStorage(dataset_storage_root)).load
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
    return BenchmarkWorker(runs=registry.runs, service=service)


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

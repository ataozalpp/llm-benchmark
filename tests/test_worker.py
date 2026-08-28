from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_benchmark.worker import BenchmarkWorker, run_polling_loop


@dataclass(frozen=True)
class StubRun:
    id: int


class StubRuns:
    def __init__(self, claims: list[StubRun | None]) -> None:
        self.claims = claims
        self.calls = 0

    def claim_next_queued(self) -> Any:
        self.calls += 1
        return self.claims.pop(0) if self.claims else None


class StubService:
    def __init__(self, error: Exception | None = None) -> None:
        self.runs: list[StubRun] = []
        self.error = error

    def execute_claimed(self, run: Any) -> object:
        self.runs.append(run)
        if self.error is not None:
            raise self.error
        return object()


def test_empty_queue_returns_false() -> None:
    assert BenchmarkWorker(runs=StubRuns([None]), service=StubService()).run_once() is False


def test_claimed_run_is_executed_once_and_then_not_repeated() -> None:
    claimed = StubRun(7)
    runs = StubRuns([claimed, None])
    service = StubService()
    worker = BenchmarkWorker(runs=runs, service=service)

    assert worker.run_once() is True
    assert worker.run_once() is False
    assert service.runs == [claimed]


def test_running_and_terminal_runs_are_not_processed_when_repository_skips_them() -> None:
    service = StubService()
    worker = BenchmarkWorker(runs=StubRuns([None]), service=service)

    assert worker.run_once() is False
    assert service.runs == []


def test_execution_failure_is_not_retried() -> None:
    claimed = StubRun(9)
    runs = StubRuns([claimed])
    service = StubService(RuntimeError("failed"))
    worker = BenchmarkWorker(runs=runs, service=service)

    with pytest.raises(RuntimeError, match="failed"):
        worker.run_once()
    assert runs.calls == 1
    assert service.runs == [claimed]


def test_import_has_no_runtime_filesystem_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("llm_benchmark.worker", None)

    importlib.import_module("llm_benchmark.worker")

    assert not (tmp_path / "runtime").exists()
    assert not list(tmp_path.glob("*.db"))


def test_polling_sleeps_when_idle_and_stops_cleanly() -> None:
    runs = StubRuns([None, None])
    worker = BenchmarkWorker(runs=runs, service=StubService())
    sleeps: list[float] = []

    def stop_after_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    run_polling_loop(worker, idle_sleep_seconds=0.25, sleep=stop_after_sleep)

    assert runs.calls == 1
    assert sleeps == [0.25]

"""Bounded pre-submission checks for API RAG runs.

A rejected preflight creates no run row, result row or artifact directory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..dataset_storage import DatasetStorageError, LocalDatasetStorage
from ..db.records import DatasetRecord
from ..run_preflight import DatasetPreflightError, RunGuardrailViolationError
from .dataset import RagDataset, load_rag_dataset
from .embedding_profiles import embeddings_available
from .run_config import RagRunConfig


class EmbeddingsUnavailableError(Exception):
    """Raised when dense or hybrid retrieval is requested without the extra."""


@dataclass(frozen=True)
class RagRunGuardrailPolicy:
    max_scenarios: int = 100
    max_k: int = 20
    max_context_chars: int = 20_000
    max_candidate_k: int = 100

    def __post_init__(self) -> None:
        for name in ("max_scenarios", "max_k", "max_context_chars", "max_candidate_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RagPreflightResult:
    suite_path: Path
    scenario_count: int


class RagRunPreflightService:
    def __init__(
        self,
        *,
        policy: RagRunGuardrailPolicy,
        storage: LocalDatasetStorage,
        embeddings_check: Callable[[], bool] = embeddings_available,
    ) -> None:
        self._policy = policy
        self._storage = storage
        self._embeddings_check = embeddings_check

    def preflight(
        self, config: RagRunConfig, dataset: DatasetRecord
    ) -> RagPreflightResult:
        policy = self._policy
        if config.k > policy.max_k or config.max_context_chars > policy.max_context_chars:
            raise RunGuardrailViolationError("RAG run exceeds API limits")
        if config.hybrid is not None and config.hybrid.candidate_k > policy.max_candidate_k:
            raise RunGuardrailViolationError("Hybrid candidate_k exceeds API limits")
        if config.retriever in {"dense", "hybrid"} and not self._embeddings_check():
            raise EmbeddingsUnavailableError("Embedding dependency is not installed")

        suite_path, suite = self.load_suite(dataset)
        if len(suite.scenarios) > policy.max_scenarios:
            raise RunGuardrailViolationError("RAG suite exceeds API limits")
        return RagPreflightResult(
            suite_path=suite_path, scenario_count=len(suite.scenarios)
        )

    def load_suite(self, dataset: DatasetRecord) -> tuple[Path, RagDataset]:
        try:
            path = self._storage.verify(dataset.source_uri, dataset.checksum)
            return path, load_rag_dataset(path)
        except (DatasetStorageError, ValueError) as error:
            raise DatasetPreflightError("RAG suite is unavailable") from error

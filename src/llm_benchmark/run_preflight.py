"""Framework-independent guardrails for synchronous Run API execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from .config import DatasetConfig, RunConfig
from .datasets import load_examples, sample_examples
from .models import DatasetExample

_EXPECTED_DATASET_LOAD_ERRORS = (
    RuntimeError,
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    KeyError,
    ValueError,
    ValidationError,
)


class RunPreflightError(Exception):
    """Base class for safe, typed preflight failures."""


class RunGuardrailViolationError(RunPreflightError):
    """Raised when a request exceeds the synchronous API policy."""


class RunSelectionValidationError(RunPreflightError):
    """Raised when a requested dataset selection is invalid or empty."""


class DatasetPreflightError(RunPreflightError):
    """Raised when registered dataset content cannot be loaded for preflight."""


@dataclass(frozen=True)
class RunApiGuardrailPolicy:
    allowed_profiles: frozenset[str] = field(default_factory=lambda: frozenset({"smoke", "poc"}))
    max_selected_samples: int = 100
    max_sample_ids: int = 100

    def __post_init__(self) -> None:
        if not self.allowed_profiles:
            raise ValueError("allowed_profiles must not be empty")
        if self.max_selected_samples <= 0:
            raise ValueError("max_selected_samples must be positive")
        if self.max_sample_ids <= 0:
            raise ValueError("max_sample_ids must be positive")


@dataclass(frozen=True)
class RunPreflightResult:
    selected_sample_count: int
    selected_sample_ids: tuple[str, ...]
    selected_categories: tuple[str, ...]


class RunPreflightService:
    def __init__(
        self,
        *,
        policy: RunApiGuardrailPolicy | None = None,
        loader: Callable[[DatasetConfig], list[DatasetExample]] = load_examples,
    ) -> None:
        self._policy = policy or RunApiGuardrailPolicy()
        self._loader = loader

    def preflight(self, config: RunConfig) -> RunPreflightResult:
        dataset = config.dataset
        self._validate_before_loading(dataset)
        try:
            loaded = self._loader(dataset)
        except _EXPECTED_DATASET_LOAD_ERRORS as error:
            raise DatasetPreflightError("Dataset loading failed during preflight") from error

        available_categories = {example.category for example in loaded}
        if any(category not in available_categories for category in dataset.category_filter):
            raise RunSelectionValidationError("Requested category does not exist")

        try:
            selected = sample_examples(loaded, dataset, config.seed)
        except ValueError as error:
            raise RunSelectionValidationError("Requested sample selection is invalid") from error
        if not selected:
            raise RunSelectionValidationError("Requested sample selection is empty")
        if len(selected) > self._policy.max_selected_samples:
            raise RunGuardrailViolationError("Selected sample count exceeds the synchronous API limit")

        return RunPreflightResult(
            selected_sample_count=len(selected),
            selected_sample_ids=tuple(example.sample_id for example in selected),
            selected_categories=tuple(sorted({example.category for example in selected})),
        )

    def _validate_before_loading(self, dataset: DatasetConfig) -> None:
        if dataset.profile not in self._policy.allowed_profiles:
            raise RunGuardrailViolationError("Dataset profile is not allowed by the synchronous API")
        if dataset.sample_size is not None and dataset.sample_size > self._policy.max_selected_samples:
            raise RunGuardrailViolationError("Requested sample size exceeds the synchronous API limit")
        if len(dataset.sample_ids) > self._policy.max_sample_ids:
            raise RunGuardrailViolationError("Requested sample ID count exceeds the synchronous API limit")

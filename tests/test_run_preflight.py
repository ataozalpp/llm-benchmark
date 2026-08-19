from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from llm_benchmark.config import RunConfig, load_config
from llm_benchmark.reproducibility import canonical_hash
from llm_benchmark.run_preflight import (
    DatasetPreflightError,
    RunApiGuardrailPolicy,
    RunGuardrailViolationError,
    RunPreflightService,
    RunSelectionValidationError,
)


def fixture_config(**dataset_updates: object) -> RunConfig:
    source = load_config(Path("configs/mock_smoke.yaml"))
    dataset = source.dataset.model_copy(
        update={"profile": "smoke", "sample_size": None, "sample_ids": [], **dataset_updates}
    )
    return source.model_copy(update={"dataset": dataset, "models": [source.models[0]]})


def test_default_policy_is_immutable_and_allows_fixture_smoke() -> None:
    policy = RunApiGuardrailPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.max_selected_samples = 1  # type: ignore[misc]

    result = RunPreflightService(policy=policy).preflight(fixture_config())
    assert result.selected_sample_count == 8
    assert result.selected_sample_ids == tuple(f"q{index:02d}" for index in range(1, 9))
    assert result.selected_categories == ("geography", "language", "math", "science")


def test_injected_exact_selection_limit_is_enforced() -> None:
    service = RunPreflightService(policy=RunApiGuardrailPolicy(max_selected_samples=2))
    with pytest.raises(RunGuardrailViolationError):
        service.preflight(fixture_config())


@pytest.mark.parametrize(
    "updates",
    [
        {"profile": "full"},
        {"sample_size": 101},
        {"sample_size": 101, "sample_ids": [f"id-{index}" for index in range(101)]},
    ],
)
def test_policy_rejects_before_dataset_loading(updates: dict[str, object]) -> None:
    loader_called = False

    def forbidden_loader(_: object) -> list[object]:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("dataset loader must not run")

    service = RunPreflightService(loader=forbidden_loader)  # type: ignore[arg-type]
    with pytest.raises(RunGuardrailViolationError):
        service.preflight(fixture_config(**updates))
    assert loader_called is False


def test_unknown_sample_id_is_rejected() -> None:
    with pytest.raises(RunSelectionValidationError):
        RunPreflightService().preflight(
            fixture_config(sample_size=1, sample_ids=["unknown-private-id"])
        )


@pytest.mark.parametrize("categories", [["unknown"], ["math", "unknown"]])
def test_unknown_and_mixed_categories_are_rejected(categories: list[str]) -> None:
    with pytest.raises(RunSelectionValidationError):
        RunPreflightService().preflight(fixture_config(category_filter=categories))


def test_empty_selection_is_rejected() -> None:
    service = RunPreflightService(loader=lambda _: [])
    with pytest.raises(RunSelectionValidationError):
        service.preflight(fixture_config())


def test_dataset_loading_failure_is_typed_and_does_not_expose_cause() -> None:
    def fail(_: object) -> list[object]:
        raise RuntimeError(r"C:\Users\private\dataset.jsonl api_key=secret")

    with pytest.raises(DatasetPreflightError) as captured:
        RunPreflightService(loader=fail).preflight(fixture_config())  # type: ignore[arg-type]
    assert str(captured.value) == "Dataset loading failed during preflight"


@pytest.mark.parametrize(
    "error",
    [
        TypeError("private type failure"),
        AttributeError("private attribute failure"),
        AssertionError("private assertion failure"),
    ],
)
def test_unexpected_loader_programming_errors_propagate(error: Exception) -> None:
    def fail(_: object) -> list[object]:
        raise error

    with pytest.raises(type(error), match="private"):
        RunPreflightService(loader=fail).preflight(fixture_config())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "error",
    [
        RunGuardrailViolationError("guardrail"),
        RunSelectionValidationError("selection"),
        DatasetPreflightError("dataset"),
    ],
)
def test_preflight_errors_raised_by_loader_are_not_reclassified(error: Exception) -> None:
    def fail(_: object) -> list[object]:
        raise error

    with pytest.raises(type(error)) as captured:
        RunPreflightService(loader=fail).preflight(fixture_config())  # type: ignore[arg-type]
    assert captured.value is error


def test_preflight_does_not_change_canonical_config_hash() -> None:
    config = fixture_config(sample_size=1, sample_ids=["q01"])
    before = canonical_hash(config.model_dump(mode="json"))
    RunPreflightService().preflight(config)
    after = canonical_hash(config.model_dump(mode="json"))
    assert after == before

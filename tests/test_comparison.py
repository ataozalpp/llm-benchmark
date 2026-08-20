from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_benchmark.comparison import (
    BenchmarkComparisonService,
    ComparisonRunNotFoundError,
    DuplicateSampleResultError,
    IncompatibleRunsError,
    MissingComparisonDataError,
    PersistedComparisonDataError,
    ReferenceDataMismatchError,
    RunNotCompletedError,
    SameRunComparisonError,
    SamplePopulationMismatchError,
)
from llm_benchmark.db.errors import RecordNotFoundError
from llm_benchmark.db.models import ProviderType, ReasoningPolicy, RunStatus
from llm_benchmark.db.records import (
    BenchmarkRunRecord,
    DatasetRecord,
    ModelRecord,
    ProviderEndpointRecord,
    SampleResultRecord,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class Readers:
    runs: dict[int, BenchmarkRunRecord]
    samples: dict[int, list[SampleResultRecord]]
    models: dict[int, ModelRecord]
    datasets: dict[int, DatasetRecord]
    endpoints: dict[int, ProviderEndpointRecord]

    def get_by_id(self, record_id: int):
        for mapping in (self.runs, self.models, self.datasets, self.endpoints):
            if record_id in mapping:
                return mapping[record_id]
        raise RecordNotFoundError("record", record_id)

    def list_by_run_id(self, run_id: int) -> list[SampleResultRecord]:
        return list(self.samples.get(run_id, []))


class MappingReader:
    def __init__(self, values, entity="record"):
        self.values = values
        self.entity = entity

    def get_by_id(self, record_id):
        try:
            return self.values[record_id]
        except KeyError as error:
            raise RecordNotFoundError(self.entity, record_id) from error


class SampleReader:
    def __init__(self, values):
        self.values = values

    def list_by_run_id(self, run_id):
        return list(self.values.get(run_id, []))


def endpoint(record_id: int, provider=ProviderType.MOCK) -> ProviderEndpointRecord:
    return ProviderEndpointRecord(
        id=record_id, name=f"endpoint-{record_id}", provider_type=provider,
        base_url="http://sensitive.invalid", credential_env_var="SECRET_ENV",
        is_active=True, created_at=NOW, updated_at=NOW,
    )


def model(record_id: int, endpoint_id: int, *, reasoning=ReasoningPolicy.UNSUPPORTED) -> ModelRecord:
    return ModelRecord(
        id=record_id, name=f"model-{record_id}", model_identifier=f"identifier-{record_id}",
        endpoint_id=endpoint_id, reasoning_policy=reasoning, capabilities_json={},
        default_generation_config_json={}, metadata_json={"secret": "do-not-expose"},
        is_active=True, created_at=NOW, updated_at=NOW,
    )


def dataset(record_id: int, **changes) -> DatasetRecord:
    data = dict(
        id=record_id, name="fixture", source_type="local", source_uri="C:/private/dataset.jsonl",
        revision="rev-1", split="test", task_type="multiple_choice", adapter_type="fixture_jsonl",
        license="project fixture", checksum="abc", metadata_json={"private": "value"},
        is_active=True, created_at=NOW, updated_at=NOW,
    )
    data.update(changes)
    return DatasetRecord(**data)


def resolved(*, model_id="model", **model_changes):
    model_data = {
        "provider": "mock", "endpoint_alias": "mock", "model_id": model_id,
        "reasoning": None, "temperature": 0, "top_p": None, "top_k": None,
        "min_p": None, "repeat_penalty": None, "max_output_tokens": 64,
        "timeout_seconds": 120, "scenario_cycle": ["correct"],
        "scenario_overrides": {}, "mock_latency_ms": 12.5,
    }
    model_data.update(model_changes)
    return {
        "schema_version": 1, "experiment_name": "comparison", "seed": 42,
        "output_dir": "outputs/api", "dataset": {"source": "local", "name": "fixture"},
        "models": [model_data],
        "evaluation": {"prompt_version": "v1", "parser_version": "p1", "evaluator_version": "e1", "scoring_mode": "generated_answer"},
    }


def summary(*, total=2, correct=1, prompt_hash="prompt-hash"):
    return {
        "prompt_template_hash": prompt_hash, "parser_version": "p1", "evaluator_version": "e1",
        "overall": {"total_samples": total, "correct_count": correct, "incorrect_count": total-correct,
                    "unparseable_count": 0, "failed_request_count": 0, "run_wall_time_ms": 25.0},
    }


def run(record_id: int, model_id: int, dataset_id: int, **changes) -> BenchmarkRunRecord:
    data = dict(
        id=record_id, experiment_name=f"run-{record_id}", model_id=model_id, dataset_id=dataset_id,
        status=RunStatus.COMPLETED, resolved_config_json=resolved(model_id=str(model_id)),
        config_hash=str(record_id) * 64, seed=42, sample_count=2, summary_json=summary(),
        artifact_directory="C:/private/output", created_at=NOW, started_at=NOW, completed_at=NOW,
        error_type=None, error_message="password=hidden SQL SELECT secret",
    )
    data.update(changes)
    return BenchmarkRunRecord(**data)


def sample(run_id: int, record_id: int, sample_id: str, **changes) -> SampleResultRecord:
    data = dict(
        id=record_id, run_id=run_id, sample_id=sample_id, category="science", correct_answer="B",
        parsed_answer="B", raw_response="private raw model output", request_status="succeeded",
        parse_status="normalized_label", evaluation_status="correct", is_correct=True,
        input_tokens=10, output_tokens=4, reasoning_tokens=None, total_tokens=14,
        latency_ms=20.0, ttft_ms=None, throughput_tokens_per_second=None,
        error_type=None, provider_error_message="Bearer sensitive-token",
    )
    data.update(changes)
    return SampleResultRecord(**data)


def fixture() -> tuple[BenchmarkComparisonService, dict[str, object]]:
    endpoints = {1: endpoint(1), 2: endpoint(2, ProviderType.OPENAI_COMPATIBLE)}
    models = {1: model(1, 1), 2: model(2, 2)}
    datasets = {1: dataset(1), 2: dataset(2)}
    runs = {1: run(1, 1, 1), 2: run(2, 2, 2)}
    samples = {
        1: [sample(1, 1, "a"), sample(1, 2, "b", parsed_answer="A", evaluation_status="incorrect", is_correct=False, latency_ms=30.0)],
        2: [sample(2, 4, "b", parsed_answer=None, parse_status="no_answer_found", evaluation_status="unparseable", is_correct=False, input_tokens=None, total_tokens=None, latency_ms=40.0), sample(2, 3, "a", output_tokens=8, total_tokens=18, latency_ms=25.0)],
    }
    service = BenchmarkComparisonService(
        runs=MappingReader(runs, "run"), samples=SampleReader(samples), models=MappingReader(models),
        datasets=MappingReader(datasets), endpoints=MappingReader(endpoints),
    )
    return service, {"runs": runs, "samples": samples, "models": models, "datasets": datasets, "endpoints": endpoints}


def metric(result, dimension: str, name: str):
    group = getattr(result.aggregate, dimension)
    return next(item for item in group.metrics if item.name == name)


def test_comparable_runs_align_rows_and_report_context_without_sensitive_data():
    service, _ = fixture()
    result = service.compare(baseline_run_id=1, candidate_run_id=2)

    assert [item.sample_id for item in result.samples] == ["a", "b"]
    assert result.comparability.directly_comparable is True
    assert result.comparability.contextual_differences == ("model_identity", "endpoint_identity", "provider_identity")
    assert metric(result, "quality", "correct_count").candidate.value == 1
    assert metric(result, "format_compliance", "unparseable_count").candidate.value == 1
    assert metric(result, "format_compliance", "parse_success_rate").candidate.value == pytest.approx(0.5)
    assert metric(result, "latency", "run_wall_time_ms").baseline.value == 25.0
    assert result.categories[0].category == "science"
    dumped = result.model_dump_json()
    for forbidden in ("private raw", "sensitive-token", "password=", "dataset.jsonl", "private/output", "SECRET_ENV", "http://"):
        assert forbidden not in dumped


def test_result_records_are_immutable_and_no_composite_score_exists():
    result = fixture()[0].compare(baseline_run_id=1, candidate_run_id=2)
    with pytest.raises(ValidationError):
        result.baseline.run_id = 99
    assert "score" not in type(result).model_fields


def test_same_missing_noncompleted_and_empty_runs_are_rejected():
    service, state = fixture()
    with pytest.raises(SameRunComparisonError):
        service.compare(baseline_run_id=1, candidate_run_id=1)
    with pytest.raises(ComparisonRunNotFoundError):
        service.compare(baseline_run_id=1, candidate_run_id=999)
    for status in (RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED):
        state["runs"][2] = run(2, 2, 2, status=status)
        with pytest.raises(RunNotCompletedError):
            service.compare(baseline_run_id=1, candidate_run_id=2)
    state["runs"][2] = run(2, 2, 2)
    state["samples"][2] = []
    with pytest.raises(MissingComparisonDataError):
        service.compare(baseline_run_id=1, candidate_run_id=2)


def test_population_duplicate_and_reference_mismatches_are_rejected():
    service, state = fixture()
    state["samples"][2] = [sample(2, 3, "a")]
    with pytest.raises(SamplePopulationMismatchError):
        service.compare(baseline_run_id=1, candidate_run_id=2)
    state["samples"][2] = [sample(2, 3, "a"), sample(2, 4, "a")]
    with pytest.raises(DuplicateSampleResultError):
        service.compare(baseline_run_id=1, candidate_run_id=2)
    for change in ({"correct_answer": "C"}, {"category": "other"}):
        state["samples"][2] = [sample(2, 3, "a", **change), sample(2, 4, "b")]
        with pytest.raises(ReferenceDataMismatchError):
            service.compare(baseline_run_id=1, candidate_run_id=2)


@pytest.mark.parametrize("field,value", [
    ("name", "other"), ("source_type", "huggingface"), ("revision", "rev-2"),
    ("split", "validation"), ("task_type", "other"), ("adapter_type", "other"),
])
def test_dataset_incompatibilities_block(field, value):
    service, state = fixture()
    state["datasets"][2] = dataset(2, **{field: value})
    with pytest.raises(IncompatibleRunsError, match="dataset"):
        service.compare(baseline_run_id=1, candidate_run_id=2)


@pytest.mark.parametrize("kind", ["prompt", "parser", "evaluator", "scoring"])
def test_evaluation_provenance_mismatch_blocks(kind):
    service, state = fixture()
    candidate = state["runs"][2]
    config = candidate.resolved_config_json.copy()
    config["evaluation"] = dict(config["evaluation"])
    report = dict(candidate.summary_json)
    if kind == "prompt":
        report["prompt_template_hash"] = "different"
    else:
        config["evaluation"][{"parser": "parser_version", "evaluator": "evaluator_version", "scoring": "scoring_mode"}[kind]] = "different"
    state["runs"][2] = candidate.model_copy(update={"resolved_config_json": config, "summary_json": report})
    with pytest.raises(IncompatibleRunsError):
        service.compare(baseline_run_id=1, candidate_run_id=2)


def test_missing_provenance_is_a_typed_persisted_data_error():
    service, state = fixture()
    state["runs"][2] = state["runs"][2].model_copy(update={"summary_json": {}})
    with pytest.raises(PersistedComparisonDataError):
        service.compare(baseline_run_id=1, candidate_run_id=2)


def test_generation_and_reasoning_differences_are_conditional_not_blocking():
    service, state = fixture()
    candidate = state["runs"][2]
    config = candidate.resolved_config_json.copy()
    config["models"] = [dict(config["models"][0], reasoning="on", temperature=1.0, top_p=.95,
                              top_k=20, min_p=.1, repeat_penalty=1.1, max_output_tokens=None,
                              timeout_seconds=300)]
    state["runs"][2] = candidate.model_copy(update={"resolved_config_json": config, "seed": 7})
    state["models"][2] = model(2, 2, reasoning=ReasoningPolicy.TOGGLE)
    result = service.compare(baseline_run_id=1, candidate_run_id=2)
    differences = set(result.comparability.conditional_differences)
    assert {"reasoning_policy", "reasoning", "temperature", "top_p", "top_k", "min_p",
            "repeat_penalty", "max_output_tokens", "output_budget_policy", "timeout_seconds",
            "seed"} <= differences
    assert result.comparability.directly_comparable is True
    assert result.comparability.generation_policy_equal is False


def test_null_metrics_preserve_missing_coverage_and_delta_is_null():
    result = fixture()[0].compare(baseline_run_id=1, candidate_run_id=2)
    tokens = metric(result, "tokens", "total_tokens")
    assert tokens.baseline == tokens.baseline.model_copy(update={"value": 28, "available_count": 2, "missing_count": 0, "complete": True})
    assert tokens.candidate.value == 18
    assert tokens.candidate.available_count == 1
    assert tokens.candidate.missing_count == 1
    assert tokens.candidate.complete is False
    assert tokens.absolute_delta is None
    ttft = metric(result, "latency", "ttft_mean_ms")
    assert ttft.baseline.value is None and ttft.candidate.value is None
    assert ttft.absolute_delta is None


def test_zero_correct_tokens_per_correct_is_null_and_latency_populations_are_separate():
    service, state = fixture()
    state["samples"][2] = [
        sample(2, 3, "a", request_status="failed", evaluation_status="request_failed", is_correct=False, parsed_answer=None, latency_ms=100),
        sample(2, 4, "b", evaluation_status="incorrect", is_correct=False, parsed_answer="A", latency_ms=20),
    ]
    result = service.compare(baseline_run_id=1, candidate_run_id=2)
    assert metric(result, "tokens", "tokens_per_correct_answer").candidate.value is None
    assert metric(result, "latency", "successful_latency_mean_ms").candidate.value == 20
    assert metric(result, "latency", "failed_latency_mean_ms").candidate.value == 100
    assert metric(result, "reliability", "request_success_rate").candidate.value == 0.5


def test_successful_latency_percentiles_report_partial_coverage_and_suppress_delta():
    service, state = fixture()
    state["samples"][2][0] = state["samples"][2][0].model_copy(update={"latency_ms": None})

    result = service.compare(baseline_run_id=1, candidate_run_id=2)

    for name, baseline_value in (("successful_latency_p50_ms", 25.0), ("successful_latency_p95_ms", 29.5)):
        comparison = metric(result, "latency", name)
        assert comparison.baseline.value == baseline_value
        assert comparison.baseline.available_count == 2
        assert comparison.baseline.missing_count == 0
        assert comparison.baseline.complete is True
        assert comparison.candidate.value == 25.0
        assert comparison.candidate.available_count == 1
        assert comparison.candidate.missing_count == 1
        assert comparison.candidate.complete is False
        assert comparison.absolute_delta is None


def test_summary_disagreement_is_warning_and_samples_remain_authoritative():
    service, state = fixture()
    state["runs"][2] = state["runs"][2].model_copy(update={"summary_json": summary(total=99, correct=99)})
    result = service.compare(baseline_run_id=1, candidate_run_id=2)
    assert result.comparability.warnings == ("candidate_summary_sample_metrics_mismatch",)
    assert metric(result, "quality", "correct_count").candidate.value == 1


def test_module_has_no_sqlalchemy_orm_artifact_or_network_dependency():
    source = Path("src/llm_benchmark/comparison.py").read_text(encoding="utf-8")
    assert "sqlalchemy" not in source.lower()
    assert "requests" not in source and "httpx" not in source
    assert "open(" not in source and "write_" not in source

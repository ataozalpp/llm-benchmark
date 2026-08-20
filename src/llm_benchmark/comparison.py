"""Framework-independent, read-only comparison of persisted benchmark runs."""

from __future__ import annotations

from collections import Counter
from math import ceil, floor
from statistics import mean
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from .db.errors import RecordNotFoundError
from .db.records import (
    BenchmarkRunRecord,
    DatasetRecord,
    ModelRecord,
    ProviderEndpointRecord,
    SampleResultRecord,
)


class ComparisonError(Exception):
    """Base class for safe comparison failures."""


class ComparisonRunNotFoundError(ComparisonError):
    pass


class RunNotCompletedError(ComparisonError):
    pass


class SameRunComparisonError(ComparisonError):
    pass


class MissingComparisonDataError(ComparisonError):
    pass


class DuplicateSampleResultError(ComparisonError):
    pass


class SamplePopulationMismatchError(ComparisonError):
    pass


class ReferenceDataMismatchError(ComparisonError):
    pass


class IncompatibleRunsError(ComparisonError):
    pass


class PersistedComparisonDataError(ComparisonError):
    pass


class RunReader(Protocol):
    def get_by_id(self, run_id: int) -> BenchmarkRunRecord: ...


class SampleReader(Protocol):
    def list_by_run_id(self, run_id: int) -> list[SampleResultRecord]: ...


class ModelReader(Protocol):
    def get_by_id(self, model_id: int) -> ModelRecord: ...


class DatasetReader(Protocol):
    def get_by_id(self, dataset_id: int) -> DatasetRecord: ...


class EndpointReader(Protocol):
    def get_by_id(self, endpoint_id: int) -> ProviderEndpointRecord: ...


class ImmutableRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SafeRunContext(ImmutableRecord):
    run_id: int
    experiment_name: str
    model_registration_id: int
    model_name: str
    model_identifier: str
    provider_type: str
    endpoint_name: str
    dataset_registration_id: int
    dataset_name: str
    dataset_revision: str | None
    dataset_split: str
    config_hash: str
    seed: int
    sample_count: int


class ComparabilityAssessment(ImmutableRecord):
    directly_comparable: bool
    sample_population_equal: bool
    evaluation_protocol_equal: bool
    generation_policy_equal: bool
    serving_context_equal: bool
    blocking_differences: tuple[str, ...]
    conditional_differences: tuple[str, ...]
    contextual_differences: tuple[str, ...]
    warnings: tuple[str, ...]


class MetricValue(ImmutableRecord):
    value: int | float | None
    available_count: int
    missing_count: int
    complete: bool


class MetricComparison(ImmutableRecord):
    name: str
    baseline: MetricValue
    candidate: MetricValue
    absolute_delta: int | float | None


class DimensionComparison(ImmutableRecord):
    name: str
    metrics: tuple[MetricComparison, ...]


class AggregateComparison(ImmutableRecord):
    sample_count: int
    quality: DimensionComparison
    format_compliance: DimensionComparison
    reliability: DimensionComparison
    tokens: DimensionComparison
    latency: DimensionComparison


class CategoryComparison(ImmutableRecord):
    category: str
    aggregate: AggregateComparison


class SampleComparison(ImmutableRecord):
    sample_id: str
    category: str | None
    correct_answer: str | None
    baseline_parsed_answer: str | None
    candidate_parsed_answer: str | None
    baseline_evaluation_status: str
    candidate_evaluation_status: str
    evaluation_transition: str
    baseline_request_status: str
    candidate_request_status: str
    request_transition: str
    baseline_parse_status: str
    candidate_parse_status: str
    parse_transition: str
    baseline_is_correct: bool
    candidate_is_correct: bool
    input_tokens: MetricComparison
    output_tokens: MetricComparison
    reasoning_tokens: MetricComparison
    total_tokens: MetricComparison
    latency_ms: MetricComparison
    ttft_ms: MetricComparison
    throughput_tokens_per_second: MetricComparison
    baseline_error_type: str | None
    candidate_error_type: str | None


class BenchmarkComparisonResult(ImmutableRecord):
    baseline: SafeRunContext
    candidate: SafeRunContext
    comparability: ComparabilityAssessment
    aggregate: AggregateComparison
    categories: tuple[CategoryComparison, ...]
    samples: tuple[SampleComparison, ...]


class BenchmarkComparisonService:
    """Compare two completed runs without database or artifact writes."""

    def __init__(
        self,
        *,
        runs: RunReader,
        samples: SampleReader,
        models: ModelReader,
        datasets: DatasetReader,
        endpoints: EndpointReader,
    ) -> None:
        self._runs = runs
        self._samples = samples
        self._models = models
        self._datasets = datasets
        self._endpoints = endpoints

    def compare(self, *, baseline_run_id: int, candidate_run_id: int) -> BenchmarkComparisonResult:
        if baseline_run_id == candidate_run_id:
            raise SameRunComparisonError("Two distinct completed runs are required")
        baseline_run = self._get_run(baseline_run_id)
        candidate_run = self._get_run(candidate_run_id)
        self._require_completed(baseline_run)
        self._require_completed(candidate_run)

        baseline_samples = self._samples.list_by_run_id(baseline_run.id)
        candidate_samples = self._samples.list_by_run_id(candidate_run.id)
        if not baseline_samples or not candidate_samples:
            raise MissingComparisonDataError("Both completed runs must contain sample results")
        baseline_by_id = _index_samples(baseline_samples)
        candidate_by_id = _index_samples(candidate_samples)
        if set(baseline_by_id) != set(candidate_by_id):
            raise SamplePopulationMismatchError("Runs contain different sample populations")

        ordered_ids = sorted(baseline_by_id)
        for sample_id in ordered_ids:
            left, right = baseline_by_id[sample_id], candidate_by_id[sample_id]
            if left.correct_answer != right.correct_answer or left.category != right.category:
                raise ReferenceDataMismatchError("Aligned samples contain different reference data")

        baseline_model = self._models.get_by_id(baseline_run.model_id)
        candidate_model = self._models.get_by_id(candidate_run.model_id)
        baseline_dataset = self._datasets.get_by_id(baseline_run.dataset_id)
        candidate_dataset = self._datasets.get_by_id(candidate_run.dataset_id)
        baseline_endpoint = self._endpoints.get_by_id(baseline_model.endpoint_id)
        candidate_endpoint = self._endpoints.get_by_id(candidate_model.endpoint_id)

        blocking = _blocking_differences(
            baseline_run, candidate_run, baseline_dataset, candidate_dataset
        )
        if blocking:
            raise IncompatibleRunsError("Runs use incompatible evaluation provenance: " + ", ".join(blocking))
        conditional = _conditional_differences(baseline_run, candidate_run, baseline_model, candidate_model)
        contextual = _contextual_differences(
            baseline_model, candidate_model, baseline_endpoint, candidate_endpoint
        )
        warnings = _summary_consistency_warnings(
            baseline_run, candidate_run, baseline_samples, candidate_samples
        )
        assessment = ComparabilityAssessment(
            directly_comparable=True,
            sample_population_equal=True,
            evaluation_protocol_equal=True,
            generation_policy_equal=not conditional,
            serving_context_equal=not contextual,
            blocking_differences=(),
            conditional_differences=tuple(conditional),
            contextual_differences=tuple(contextual),
            warnings=tuple(warnings),
        )

        pairs = [(baseline_by_id[item], candidate_by_id[item]) for item in ordered_ids]
        categories = tuple(
            CategoryComparison(
                category=category,
                aggregate=_aggregate(
                    [(left, right) for left, right in pairs if (left.category or "uncategorized") == category]
                ),
            )
            for category in sorted({item.category or "uncategorized" for item in baseline_samples})
        )
        return BenchmarkComparisonResult(
            baseline=_context(baseline_run, baseline_model, baseline_dataset, baseline_endpoint),
            candidate=_context(candidate_run, candidate_model, candidate_dataset, candidate_endpoint),
            comparability=assessment,
            aggregate=_aggregate(pairs, baseline_run=baseline_run, candidate_run=candidate_run),
            categories=categories,
            samples=tuple(_sample_comparison(left, right) for left, right in pairs),
        )

    def _get_run(self, run_id: int) -> BenchmarkRunRecord:
        try:
            return self._runs.get_by_id(run_id)
        except RecordNotFoundError as error:
            raise ComparisonRunNotFoundError("Benchmark run was not found") from error

    @staticmethod
    def _require_completed(run: BenchmarkRunRecord) -> None:
        if run.status.value != "completed":
            raise RunNotCompletedError("Only completed benchmark runs can be compared")


def _index_samples(samples: list[SampleResultRecord]) -> dict[str, SampleResultRecord]:
    indexed: dict[str, SampleResultRecord] = {}
    for sample in samples:
        if sample.sample_id in indexed:
            raise DuplicateSampleResultError("A run contains duplicate sample IDs")
        indexed[sample.sample_id] = sample
    return indexed


def _required_provenance(run: BenchmarkRunRecord) -> tuple[str, str, str, str]:
    summary = run.summary_json
    evaluation = run.resolved_config_json.get("evaluation")
    if not isinstance(summary, dict) or not isinstance(evaluation, dict):
        raise PersistedComparisonDataError("Required evaluation provenance is missing")
    values = (
        summary.get("prompt_template_hash"),
        evaluation.get("parser_version") or summary.get("parser_version"),
        evaluation.get("evaluator_version") or summary.get("evaluator_version"),
        evaluation.get("scoring_mode"),
    )
    if not all(isinstance(value, str) and value for value in values):
        raise PersistedComparisonDataError("Required evaluation provenance is missing")
    return values  # type: ignore[return-value]


def _blocking_differences(
    left: BenchmarkRunRecord,
    right: BenchmarkRunRecord,
    left_dataset: DatasetRecord,
    right_dataset: DatasetRecord,
) -> list[str]:
    fields = {
        "dataset_identity": (left_dataset.name, right_dataset.name),
        "dataset_source": (left_dataset.source_type, right_dataset.source_type),
        "dataset_source_identity": (left_dataset.source_uri, right_dataset.source_uri),
        "dataset_revision": (left_dataset.revision, right_dataset.revision),
        "dataset_split": (left_dataset.split, right_dataset.split),
        "dataset_task_type": (left_dataset.task_type, right_dataset.task_type),
        "dataset_adapter": (left_dataset.adapter_type, right_dataset.adapter_type),
    }
    differences = [name for name, values in fields.items() if values[0] != values[1]]
    provenance_names = ("prompt_template_hash", "parser_version", "evaluator_version", "scoring_mode")
    left_provenance, right_provenance = _required_provenance(left), _required_provenance(right)
    differences.extend(
        name for name, left_value, right_value in zip(provenance_names, left_provenance, right_provenance)
        if left_value != right_value
    )
    return differences


def _model_config(run: BenchmarkRunRecord) -> dict[str, Any]:
    models = run.resolved_config_json.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise PersistedComparisonDataError("Required generation provenance is missing")
    return models[0]


def _conditional_differences(
    left: BenchmarkRunRecord,
    right: BenchmarkRunRecord,
    left_model: ModelRecord,
    right_model: ModelRecord,
) -> list[str]:
    left_config, right_config = _model_config(left), _model_config(right)
    differences: list[str] = []
    if left_model.reasoning_policy != right_model.reasoning_policy:
        differences.append("reasoning_policy")
    fields = (
        "reasoning", "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
        "max_output_tokens", "timeout_seconds", "scenario_cycle", "scenario_overrides",
        "mock_latency_ms",
    )
    differences.extend(name for name in fields if left_config.get(name) != right_config.get(name))
    if (left_config.get("max_output_tokens") is None) != (right_config.get("max_output_tokens") is None):
        differences.append("output_budget_policy")
    if left.seed != right.seed:
        differences.append("seed")
    return list(dict.fromkeys(differences))


def _contextual_differences(
    left_model: ModelRecord,
    right_model: ModelRecord,
    left_endpoint: ProviderEndpointRecord,
    right_endpoint: ProviderEndpointRecord,
) -> list[str]:
    differences: list[str] = []
    if left_model.id != right_model.id or left_model.model_identifier != right_model.model_identifier:
        differences.append("model_identity")
    if left_endpoint.id != right_endpoint.id:
        differences.append("endpoint_identity")
    if left_endpoint.provider_type != right_endpoint.provider_type:
        differences.append("provider_identity")
    return differences


def _context(
    run: BenchmarkRunRecord,
    model: ModelRecord,
    dataset: DatasetRecord,
    endpoint: ProviderEndpointRecord,
) -> SafeRunContext:
    return SafeRunContext(
        run_id=run.id,
        experiment_name=run.experiment_name,
        model_registration_id=model.id,
        model_name=model.name,
        model_identifier=model.model_identifier,
        provider_type=endpoint.provider_type.value,
        endpoint_name=endpoint.name,
        dataset_registration_id=dataset.id,
        dataset_name=dataset.name,
        dataset_revision=dataset.revision,
        dataset_split=dataset.split,
        config_hash=run.config_hash,
        seed=run.seed,
        sample_count=run.sample_count,
    )


def _metric_value(
    values: list[int | float | None], *, total: bool = False, expected_count: int | None = None
) -> MetricValue:
    known = [value for value in values if value is not None]
    population = len(values) if expected_count is None else expected_count
    value: int | float | None
    if not known:
        value = None
    elif total:
        value = sum(known)
    else:
        value = mean(known)
    return MetricValue(
        value=value,
        available_count=len(known),
        missing_count=population - len(known),
        complete=len(known) == population,
    )


def _scalar(value: int | float | None, count: int) -> MetricValue:
    return MetricValue(value=value, available_count=count if value is not None else 0, missing_count=0 if value is not None else count, complete=value is not None)


def _compare_metric(name: str, left: MetricValue, right: MetricValue) -> MetricComparison:
    delta = None
    if left.value is not None and right.value is not None and left.complete and right.complete:
        delta = right.value - left.value
    return MetricComparison(name=name, baseline=left, candidate=right, absolute_delta=delta)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = floor(position), ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _percentile_metric(values: list[float], q: float, expected_count: int) -> MetricValue:
    return MetricValue(
        value=_percentile(values, q),
        available_count=len(values),
        missing_count=expected_count - len(values),
        complete=len(values) == expected_count,
    )


def _aggregate_one(samples: list[SampleResultRecord]) -> dict[str, MetricValue]:
    count = len(samples)
    correct = sum(item.evaluation_status == "correct" for item in samples)
    incorrect = sum(item.evaluation_status == "incorrect" for item in samples)
    unparseable = sum(item.evaluation_status == "unparseable" for item in samples)
    failed = sum(item.evaluation_status == "request_failed" for item in samples)
    request_success = count - failed
    parseable = correct + incorrect
    success_latency = [item.latency_ms for item in samples if item.request_status == "succeeded" and item.latency_ms is not None]
    failure_latency = [item.latency_ms for item in samples if item.request_status != "succeeded" and item.latency_ms is not None]
    success_count = sum(item.request_status == "succeeded" for item in samples)
    failure_count = count - success_count
    metrics = {
        "correct_count": _scalar(correct, count), "incorrect_count": _scalar(incorrect, count),
        "accuracy": _scalar(correct / count if count else None, count),
        "answered_accuracy": _scalar(correct / parseable if parseable else None, count),
        "unparseable_count": _scalar(unparseable, count),
        "parse_success_rate": _scalar(parseable / request_success if request_success else None, count),
        "format_failure_rate": _scalar(unparseable / request_success if request_success else None, count),
        "failed_request_count": _scalar(failed, count),
        "request_success_rate": _scalar(request_success / count if count else None, count),
        "input_tokens": _metric_value([item.input_tokens for item in samples], total=True),
        "output_tokens": _metric_value([item.output_tokens for item in samples], total=True),
        "reasoning_tokens": _metric_value([item.reasoning_tokens for item in samples], total=True),
        "total_tokens": _metric_value([item.total_tokens for item in samples], total=True),
        "tokens_per_correct_answer": _scalar(
            sum(item.total_tokens for item in samples if item.total_tokens is not None) / correct
            if correct and all(item.total_tokens is not None for item in samples) else None,
            count,
        ),
        "successful_latency_mean_ms": _metric_value(success_latency, expected_count=success_count),
        "successful_latency_p50_ms": _percentile_metric(success_latency, 0.50, success_count),
        "successful_latency_p95_ms": _percentile_metric(success_latency, 0.95, success_count),
        "failed_latency_mean_ms": _metric_value(failure_latency, expected_count=failure_count),
        "ttft_mean_ms": _metric_value([item.ttft_ms for item in samples]),
        "throughput_mean_tokens_per_second": _metric_value([item.throughput_tokens_per_second for item in samples]),
    }
    return metrics


def _aggregate(
    pairs: list[tuple[SampleResultRecord, SampleResultRecord]],
    *,
    baseline_run: BenchmarkRunRecord | None = None,
    candidate_run: BenchmarkRunRecord | None = None,
) -> AggregateComparison:
    left, right = [item[0] for item in pairs], [item[1] for item in pairs]
    left_metrics, right_metrics = _aggregate_one(left), _aggregate_one(right)
    left_metrics["run_wall_time_ms"] = _scalar(_run_wall_time(baseline_run), 1)
    right_metrics["run_wall_time_ms"] = _scalar(_run_wall_time(candidate_run), 1)
    def dimension(name: str, metric_names: tuple[str, ...]) -> DimensionComparison:
        return DimensionComparison(name=name, metrics=tuple(
            _compare_metric(metric, left_metrics[metric], right_metrics[metric]) for metric in metric_names
        ))
    return AggregateComparison(
        sample_count=len(pairs),
        quality=dimension("quality", ("correct_count", "incorrect_count", "accuracy", "answered_accuracy")),
        format_compliance=dimension("format_compliance", ("unparseable_count", "parse_success_rate", "format_failure_rate")),
        reliability=dimension("reliability", ("failed_request_count", "request_success_rate")),
        tokens=dimension("tokens", ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "tokens_per_correct_answer")),
        latency=dimension("latency", ("successful_latency_mean_ms", "successful_latency_p50_ms", "successful_latency_p95_ms", "failed_latency_mean_ms", "ttft_mean_ms", "throughput_mean_tokens_per_second", "run_wall_time_ms")),
    )


def _run_wall_time(run: BenchmarkRunRecord | None) -> float | None:
    if run is None or not isinstance(run.summary_json, dict):
        return None
    overall = run.summary_json.get("overall")
    if not isinstance(overall, dict):
        return None
    value = overall.get("run_wall_time_ms")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _single_metric(name: str, left: int | float | None, right: int | float | None) -> MetricComparison:
    return _compare_metric(name, _metric_value([left]), _metric_value([right]))


def _sample_comparison(left: SampleResultRecord, right: SampleResultRecord) -> SampleComparison:
    return SampleComparison(
        sample_id=left.sample_id, category=left.category, correct_answer=left.correct_answer,
        baseline_parsed_answer=left.parsed_answer, candidate_parsed_answer=right.parsed_answer,
        baseline_evaluation_status=left.evaluation_status, candidate_evaluation_status=right.evaluation_status,
        evaluation_transition=f"{left.evaluation_status}->{right.evaluation_status}",
        baseline_request_status=left.request_status, candidate_request_status=right.request_status,
        request_transition=f"{left.request_status}->{right.request_status}",
        baseline_parse_status=left.parse_status, candidate_parse_status=right.parse_status,
        parse_transition=f"{left.parse_status}->{right.parse_status}",
        baseline_is_correct=left.is_correct, candidate_is_correct=right.is_correct,
        input_tokens=_single_metric("input_tokens", left.input_tokens, right.input_tokens),
        output_tokens=_single_metric("output_tokens", left.output_tokens, right.output_tokens),
        reasoning_tokens=_single_metric("reasoning_tokens", left.reasoning_tokens, right.reasoning_tokens),
        total_tokens=_single_metric("total_tokens", left.total_tokens, right.total_tokens),
        latency_ms=_single_metric("latency_ms", left.latency_ms, right.latency_ms),
        ttft_ms=_single_metric("ttft_ms", left.ttft_ms, right.ttft_ms),
        throughput_tokens_per_second=_single_metric("throughput_tokens_per_second", left.throughput_tokens_per_second, right.throughput_tokens_per_second),
        baseline_error_type=left.error_type, candidate_error_type=right.error_type,
    )


def _summary_consistency_warnings(
    left_run: BenchmarkRunRecord,
    right_run: BenchmarkRunRecord,
    left_samples: list[SampleResultRecord],
    right_samples: list[SampleResultRecord],
) -> list[str]:
    warnings: list[str] = []
    for label, run, samples in (("baseline", left_run, left_samples), ("candidate", right_run, right_samples)):
        overall = run.summary_json.get("overall") if isinstance(run.summary_json, dict) else None
        if isinstance(overall, dict):
            recomputed = Counter(item.evaluation_status for item in samples)
            checks = {
                "total_samples": len(samples), "correct_count": recomputed["correct"],
                "incorrect_count": recomputed["incorrect"], "unparseable_count": recomputed["unparseable"],
                "failed_request_count": recomputed["request_failed"],
            }
            if any(overall.get(name) != value for name, value in checks.items()):
                warnings.append(f"{label}_summary_sample_metrics_mismatch")
    return warnings

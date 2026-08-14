from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetExample:
    sample_id: str
    question: str
    options: list[str]
    correct_answer: str
    category: str

    @property
    def allowed_labels(self) -> list[str]:
        return [chr(65 + index) for index in range(len(self.options))]


@dataclass(frozen=True)
class ProviderResponse:
    request_status: str
    raw_response: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    error_type: str | None = None
    returned_model: str | None = None
    input_tokens: int | None = None
    total_output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    final_output_tokens: int | None = None
    reasoning_observed: bool | None = None
    tokens_per_second: float | None = None
    time_to_first_token_ms: float | None = None
    stop_reason: str | None = None
    http_status_code: int | None = None
    provider_error_type: str | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None


@dataclass(frozen=True)
class ParseResult:
    parsed_answer: str | None
    parse_status: str
    candidate_answers: list[str]


@dataclass
class BenchmarkResult:
    run_id: str
    sample_id: str
    dataset_name: str
    dataset_split: str
    category: str
    provider: str
    endpoint_alias: str
    model: str
    returned_model: str | None
    raw_response: str | None
    parsed_answer: str | None
    correct_answer: str
    is_correct: bool
    evaluation_status: str
    parse_status: str
    request_status: str
    error_type: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: float | None
    attempt_count: int
    attempt_latency_ms: float
    logical_request_latency_ms: float
    started_at: str
    completed_at: str
    parser_version: str
    evaluator_version: str
    reasoning_mode: str | None = None
    input_tokens: int | None = None
    total_output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    final_output_tokens: int | None = None
    reasoning_observed: bool | None = None
    tokens_per_second: float | None = None
    time_to_first_token_ms: float | None = None
    stop_reason: str | None = None
    timeout_seconds: float | None = None
    http_status_code: int | None = None
    provider_error_type: str | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""ORM-independent records returned by registry repositories."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from .models import ProviderType, ReasoningPolicy, RunStatus


class RepositoryRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)


class ProviderEndpointRecord(RepositoryRecord):
    id: int
    name: str
    provider_type: ProviderType
    base_url: str
    credential_env_var: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ModelRecord(RepositoryRecord):
    id: int
    name: str
    model_identifier: str
    endpoint_id: int
    reasoning_policy: ReasoningPolicy
    capabilities_json: dict[str, Any]
    default_generation_config_json: dict[str, Any]
    metadata_json: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class DatasetRecord(RepositoryRecord):
    id: int
    name: str
    source_type: str
    source_uri: str
    revision: str | None
    split: str
    task_type: str
    adapter_type: str
    license: str | None
    checksum: str | None
    metadata_json: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class BenchmarkRunRecord(RepositoryRecord):
    id: int
    experiment_name: str
    model_id: int
    dataset_id: int
    status: RunStatus
    resolved_config_json: dict[str, Any]
    config_hash: str
    seed: int
    sample_count: int
    summary_json: dict[str, Any] | None
    artifact_directory: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_type: str | None
    error_message: str | None


class SampleResultCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    category: str | None = None
    correct_answer: str | None = None
    parsed_answer: str | None = None
    raw_response: str | None = None
    request_status: str
    parse_status: str
    evaluation_status: str
    is_correct: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    throughput_tokens_per_second: float | None = None
    error_type: str | None = None
    provider_error_message: str | None = None


class SampleResultRecord(SampleResultCreate):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: int
    run_id: int

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum as SqlEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, utc_now


class ProviderType(str, Enum):
    MOCK = "mock"
    OPENAI_COMPATIBLE = "openai_compatible"
    LM_STUDIO_NATIVE = "lm_studio_native"


class ReasoningPolicy(str, Enum):
    UNSUPPORTED = "unsupported"
    TOGGLE = "toggle"
    ALWAYS_ON = "always_on"
    PROVIDER_MANAGED = "provider_managed"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def portable_enum(enum_type: type[Enum], name: str) -> SqlEnum:
    return SqlEnum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, server_default=func.now(), nullable=False
    )


class ProviderEndpoint(TimestampMixin, Base):
    __tablename__ = "provider_endpoints"
    __table_args__ = (
        UniqueConstraint("name", name="uq_provider_endpoints_name"),
        Index("ix_provider_endpoints_provider_type_active", "provider_type", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_type: Mapped[ProviderType] = mapped_column(portable_enum(ProviderType, "provider_type"), nullable=False)
    base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    credential_env_var: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())

    models: Mapped[list[RegisteredModel]] = relationship(back_populates="endpoint")


class RegisteredModel(TimestampMixin, Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("endpoint_id", "model_identifier", name="uq_models_endpoint_identifier"),
        Index("ix_models_endpoint_active", "endpoint_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String(512), nullable=False)
    endpoint_id: Mapped[int] = mapped_column(ForeignKey("provider_endpoints.id"), nullable=False)
    reasoning_policy: Mapped[ReasoningPolicy] = mapped_column(
        portable_enum(ReasoningPolicy, "reasoning_policy"), nullable=False
    )
    capabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    default_generation_config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())

    endpoint: Mapped[ProviderEndpoint] = relationship(back_populates="models")
    benchmark_runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="model")


class RegisteredDataset(TimestampMixin, Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_uri",
            "revision",
            "split",
            "task_type",
            "adapter_type",
            name="uq_datasets_source_revision_task",
        ),
        Index("ix_datasets_task_active", "task_type", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    revision: Mapped[str | None] = mapped_column(String(255))
    split: Mapped[str] = mapped_column(String(128), nullable=False)
    task_type: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_type: Mapped[str] = mapped_column(String(255), nullable=False)
    license: Mapped[str | None] = mapped_column(String(255))
    checksum: Mapped[str | None] = mapped_column(String(255))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())

    benchmark_runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="dataset")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    __table_args__ = (
        Index("ix_benchmark_runs_status_created", "status", "created_at"),
        Index("ix_benchmark_runs_config_hash", "config_hash"),
        Index("ix_benchmark_runs_model_dataset", "model_id", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        portable_enum(RunStatus, "run_status"), nullable=False, default=RunStatus.QUEUED, server_default="queued"
    )
    resolved_config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    artifact_directory: Mapped[str] = mapped_column(String(2048), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)

    model: Mapped[RegisteredModel] = relationship(back_populates="benchmark_runs")
    dataset: Mapped[RegisteredDataset] = relationship(back_populates="benchmark_runs")
    sample_results: Mapped[list[SampleResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class SampleResult(Base):
    __tablename__ = "sample_results"
    __table_args__ = (
        UniqueConstraint("run_id", "sample_id", name="uq_sample_results_run_sample"),
        Index("ix_sample_results_run_evaluation", "run_id", "evaluation_status"),
        Index("ix_sample_results_run_category", "run_id", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False)
    sample_id: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str | None] = mapped_column(String(255))
    correct_answer: Mapped[str | None] = mapped_column(String(128))
    parsed_answer: Mapped[str | None] = mapped_column(String(128))
    raw_response: Mapped[str | None] = mapped_column(Text)
    request_status: Mapped[str] = mapped_column(String(64), nullable=False)
    parse_status: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluation_status: Mapped[str] = mapped_column(String(64), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    ttft_ms: Mapped[float | None] = mapped_column(Float)
    throughput_tokens_per_second: Mapped[float | None] = mapped_column(Float)
    error_type: Mapped[str | None] = mapped_column(String(128))
    provider_error_message: Mapped[str | None] = mapped_column(Text)

    run: Mapped[BenchmarkRun] = relationship(back_populates="sample_results")

"""Create the initial benchmark registry tables.

Revision ID: 20260817_0001
Revises: None
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260817_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


provider_type = sa.Enum(
    "mock",
    "openai_compatible",
    "lm_studio_native",
    name="provider_type",
    native_enum=False,
    create_constraint=True,
)
reasoning_policy = sa.Enum(
    "unsupported",
    "toggle",
    "always_on",
    "provider_managed",
    name="reasoning_policy",
    native_enum=False,
    create_constraint=True,
)
run_status = sa.Enum(
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    name="run_status",
    native_enum=False,
    create_constraint=True,
)


def upgrade() -> None:
    op.create_table(
        "provider_endpoints",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider_type", provider_type, nullable=False),
        sa.Column("base_url", sa.String(length=2048), nullable=False),
        sa.Column("credential_env_var", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_provider_endpoints")),
        sa.UniqueConstraint("name", name="uq_provider_endpoints_name"),
    )
    op.create_index(
        "ix_provider_endpoints_provider_type_active", "provider_endpoints", ["provider_type", "is_active"]
    )

    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_uri", sa.String(length=2048), nullable=False),
        sa.Column("revision", sa.String(length=255), nullable=True),
        sa.Column("split", sa.String(length=128), nullable=False),
        sa.Column("task_type", sa.String(length=128), nullable=False),
        sa.Column("adapter_type", sa.String(length=255), nullable=False),
        sa.Column("license", sa.String(length=255), nullable=True),
        sa.Column("checksum", sa.String(length=255), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_datasets")),
        sa.UniqueConstraint(
            "source_type",
            "source_uri",
            "revision",
            "split",
            "task_type",
            "adapter_type",
            name="uq_datasets_source_revision_task",
        ),
    )
    op.create_index("ix_datasets_task_active", "datasets", ["task_type", "is_active"])

    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model_identifier", sa.String(length=512), nullable=False),
        sa.Column("endpoint_id", sa.Integer(), nullable=False),
        sa.Column("reasoning_policy", reasoning_policy, nullable=False),
        sa.Column("capabilities_json", sa.JSON(), nullable=False),
        sa.Column("default_generation_config_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["endpoint_id"], ["provider_endpoints.id"], name=op.f("fk_models_endpoint_id_provider_endpoints")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_models")),
        sa.UniqueConstraint("endpoint_id", "model_identifier", name="uq_models_endpoint_identifier"),
    )
    op.create_index("ix_models_endpoint_active", "models", ["endpoint_id", "is_active"])

    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_name", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("status", run_status, server_default="queued", nullable=False),
        sa.Column("resolved_config_json", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("artifact_directory", sa.String(length=2048), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name=op.f("fk_benchmark_runs_dataset_id_datasets")),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], name=op.f("fk_benchmark_runs_model_id_models")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_runs")),
    )
    op.create_index("ix_benchmark_runs_config_hash", "benchmark_runs", ["config_hash"])
    op.create_index("ix_benchmark_runs_model_dataset", "benchmark_runs", ["model_id", "dataset_id"])
    op.create_index("ix_benchmark_runs_status_created", "benchmark_runs", ["status", "created_at"])

    op.create_table(
        "sample_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("sample_id", sa.String(length=512), nullable=False),
        sa.Column("category", sa.String(length=255), nullable=True),
        sa.Column("correct_answer", sa.String(length=128), nullable=True),
        sa.Column("parsed_answer", sa.String(length=128), nullable=True),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("request_status", sa.String(length=64), nullable=False),
        sa.Column("parse_status", sa.String(length=128), nullable=False),
        sa.Column("evaluation_status", sa.String(length=64), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("throughput_tokens_per_second", sa.Float(), nullable=True),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("provider_error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["benchmark_runs.id"], name=op.f("fk_sample_results_run_id_benchmark_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sample_results")),
        sa.UniqueConstraint("run_id", "sample_id", name="uq_sample_results_run_sample"),
    )
    op.create_index("ix_sample_results_run_category", "sample_results", ["run_id", "category"])
    op.create_index("ix_sample_results_run_evaluation", "sample_results", ["run_id", "evaluation_status"])


def downgrade() -> None:
    op.drop_index("ix_sample_results_run_evaluation", table_name="sample_results")
    op.drop_index("ix_sample_results_run_category", table_name="sample_results")
    op.drop_table("sample_results")
    op.drop_index("ix_benchmark_runs_status_created", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_model_dataset", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_config_hash", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
    op.drop_index("ix_models_endpoint_active", table_name="models")
    op.drop_table("models")
    op.drop_index("ix_datasets_task_active", table_name="datasets")
    op.drop_table("datasets")
    op.drop_index("ix_provider_endpoints_provider_type_active", table_name="provider_endpoints")
    op.drop_table("provider_endpoints")

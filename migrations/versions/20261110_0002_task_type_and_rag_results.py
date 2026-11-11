"""Add run task types and RAG case results.

Revision ID: 20261110_0002
Revises: 20260817_0001

Existing runs become ``multiple_choice``. ``seed`` becomes nullable because
RAG runs have no sampling seed. Downgrading deletes non-multiple-choice runs
(and, through the cascade, their RAG case results) before restoring the
NOT NULL seed column.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261110_0002"
down_revision: str | None = "20260817_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TASK_TYPE_CHECK = "ck_benchmark_runs_task_type"
TASK_TYPE_VALUES = "task_type IN ('multiple_choice', 'rag')"


def upgrade() -> None:
    # On SQLite the batch recreates the table; reflected CHECK constraints
    # (including ck_benchmark_runs_run_status) are carried over.
    with op.batch_alter_table("benchmark_runs") as batch:
        batch.add_column(
            sa.Column(
                "task_type",
                sa.String(length=15),
                server_default="multiple_choice",
                nullable=False,
            )
        )
        batch.alter_column("seed", existing_type=sa.Integer(), nullable=True)
        batch.create_check_constraint(op.f(TASK_TYPE_CHECK), sa.text(TASK_TYPE_VALUES))
        batch.create_index("ix_benchmark_runs_task_type", ["task_type"])

    op.create_table(
        "rag_case_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("query_id", sa.String(length=512), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("retrieved_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("context_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("parsed_answer", sa.JSON(), nullable=True),
        sa.Column("evaluation", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("retrieval_latency_ms", sa.Float(), nullable=True),
        sa.Column("generation_latency_ms", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_runs.id"],
            name=op.f("fk_rag_case_results_run_id_benchmark_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_case_results")),
        sa.UniqueConstraint("run_id", "query_id", name="uq_rag_case_results_run_query"),
    )
    op.create_index(
        "ix_rag_case_results_run_status", "rag_case_results", ["run_id", "status"]
    )


def downgrade() -> None:
    op.drop_index("ix_rag_case_results_run_status", table_name="rag_case_results")
    op.drop_table("rag_case_results")
    op.execute("DELETE FROM benchmark_runs WHERE task_type <> 'multiple_choice'")
    with op.batch_alter_table("benchmark_runs") as batch:
        batch.drop_index("ix_benchmark_runs_task_type")
        batch.drop_constraint(op.f(TASK_TYPE_CHECK), type_="check")
        batch.alter_column("seed", existing_type=sa.Integer(), nullable=False)
        batch.drop_column("task_type")

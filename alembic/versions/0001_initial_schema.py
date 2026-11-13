"""Initial Phase 8 schema: users, workspaces, API keys, async jobs, and workspace-scoped tables.

Revision ID: 0001_initial_phase8
Revises:
Create Date: 2026-11-12
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_phase8"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=512), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.String(length=128), nullable=True),
        sa.Column("settings_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workspaces_id", "workspaces", ["id"], unique=False)
    op.create_index("ix_workspaces_slug", "workspaces", ["slug"], unique=True)

    op.create_table(
        "workspace_members",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member"),
    )
    op.create_index("ix_workspace_members_id", "workspace_members", ["id"], unique=False)
    op.create_index("ix_workspace_members_user_id", "workspace_members", ["user_id"], unique=False)
    op.create_index("ix_workspace_members_workspace_id", "workspace_members", ["workspace_id"], unique=False)

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("key_prefix", sa.String(length=32), nullable=False),
        sa.Column("key_hash", sa.String(length=128), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_keys_id", "api_keys", ["id"], unique=False)
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"], unique=False)
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"], unique=False)
    op.create_index("ix_api_keys_workspace_id", "api_keys", ["workspace_id"], unique=False)

    op.create_table(
        "async_jobs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("backend", sa.String(length=32), nullable=False),
        sa.Column("celery_task_id", sa.String(length=128), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("progress_pct", sa.Float(), nullable=True),
        sa.Column("current_step", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_async_jobs_id", "async_jobs", ["id"], unique=False)
    op.create_index("ix_async_jobs_job_type", "async_jobs", ["job_type"], unique=False)
    op.create_index("ix_async_jobs_status", "async_jobs", ["status"], unique=False)
    op.create_index("ix_async_jobs_workspace_id", "async_jobs", ["workspace_id"], unique=False)

    op.create_table(
        "models",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("architecture", sa.String(length=128), nullable=True),
        sa.Column("parameter_count_b", sa.Float(), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("precision", sa.String(length=32), nullable=True),
        sa.Column("is_commercial_api", sa.Boolean(), nullable=True),
        sa.Column("is_fine_tuned", sa.Boolean(), nullable=True),
        sa.Column("is_quantized", sa.Boolean(), nullable=True),
        sa.Column("base_model_id", sa.String(length=128), nullable=True),
        sa.Column("adapter_path", sa.String(length=512), nullable=True),
        sa.Column("quantization_method", sa.String(length=64), nullable=True),
        sa.Column("hardware_compatibility", sa.JSON(), nullable=True),
        sa.Column("license", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_models_id", "models", ["id"], unique=False)
    op.create_index("ix_models_workspace_id", "models", ["workspace_id"], unique=False)

    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("train_count", sa.Integer(), nullable=True),
        sa.Column("val_count", sa.Integer(), nullable=True),
        sa.Column("test_count", sa.Integer(), nullable=True),
        sa.Column("avg_input_tokens", sa.Float(), nullable=True),
        sa.Column("avg_summary_tokens", sa.Float(), nullable=True),
        sa.Column("max_input_tokens", sa.Integer(), nullable=True),
        sa.Column("max_summary_tokens", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("sample_records", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_id", "datasets", ["id"], unique=False)
    op.create_index("ix_datasets_workspace_id", "datasets", ["workspace_id"], unique=False)

    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("experiment_id", sa.String(length=128), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("dataset_name", sa.String(length=128), nullable=True),
        sa.Column("split", sa.String(length=64), nullable=True),
        sa.Column("num_samples", sa.Integer(), nullable=True),
        sa.Column("rouge1", sa.Float(), nullable=True),
        sa.Column("rouge2", sa.Float(), nullable=True),
        sa.Column("rougeL", sa.Float(), nullable=True),
        sa.Column("bert_score", sa.Float(), nullable=True),
        sa.Column("factual_consistency", sa.Float(), nullable=True),
        sa.Column("avg_input_length", sa.Float(), nullable=True),
        sa.Column("avg_output_length", sa.Float(), nullable=True),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("avg_ttft_ms", sa.Float(), nullable=True),
        sa.Column("tokens_per_second", sa.Float(), nullable=True),
        sa.Column("peak_memory_mb", sa.Float(), nullable=True),
        sa.Column("cost_per_request_usd", sa.Float(), nullable=True),
        sa.Column("hardware_info", sa.JSON(), nullable=True),
        sa.Column("generations_sample", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluations_experiment_id", "evaluations", ["experiment_id"], unique=False)
    op.create_index("ix_evaluations_id", "evaluations", ["id"], unique=False)
    op.create_index("ix_evaluations_model_id", "evaluations", ["model_id"], unique=False)
    op.create_index("ix_evaluations_workspace_id", "evaluations", ["workspace_id"], unique=False)

    op.create_table(
        "training_jobs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("base_model_id", sa.String(length=128), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("trainable_parameters", sa.Integer(), nullable=True),
        sa.Column("total_parameters", sa.Integer(), nullable=True),
        sa.Column("trainable_percentage", sa.Float(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("checkpoint_path", sa.String(length=512), nullable=True),
        sa.Column("checkpoint_size_mb", sa.Float(), nullable=True),
        sa.Column("peak_memory_mb", sa.Float(), nullable=True),
        sa.Column("metrics_history", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_jobs_id", "training_jobs", ["id"], unique=False)
    op.create_index("ix_training_jobs_workspace_id", "training_jobs", ["workspace_id"], unique=False)

    op.create_table(
        "quantizations",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("base_model_id", sa.String(length=128), nullable=False),
        sa.Column("quantized_model_id", sa.String(length=128), nullable=False),
        sa.Column("method", sa.String(length=64), nullable=False),
        sa.Column("target_precision", sa.String(length=32), nullable=False),
        sa.Column("original_memory_mb", sa.Float(), nullable=True),
        sa.Column("quantized_memory_mb", sa.Float(), nullable=True),
        sa.Column("memory_reduction_pct", sa.Float(), nullable=True),
        sa.Column("baseline_rougel", sa.Float(), nullable=True),
        sa.Column("quantized_rougel", sa.Float(), nullable=True),
        sa.Column("rougel_delta", sa.Float(), nullable=True),
        sa.Column("baseline_latency_ms", sa.Float(), nullable=True),
        sa.Column("quantized_latency_ms", sa.Float(), nullable=True),
        sa.Column("speedup_ratio", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_quantizations_id", "quantizations", ["id"], unique=False)
    op.create_index("ix_quantizations_workspace_id", "quantizations", ["workspace_id"], unique=False)

    op.create_table(
        "benchmarks",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("benchmark_type", sa.String(length=32), nullable=True),
        sa.Column("concurrency_level", sa.Integer(), nullable=True),
        sa.Column("num_requests", sa.Integer(), nullable=True),
        sa.Column("successful_requests", sa.Integer(), nullable=True),
        sa.Column("failed_requests", sa.Integer(), nullable=True),
        sa.Column("p50_latency_ms", sa.Float(), nullable=True),
        sa.Column("p95_latency_ms", sa.Float(), nullable=True),
        sa.Column("p99_latency_ms", sa.Float(), nullable=True),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("avg_ttft_ms", sa.Float(), nullable=True),
        sa.Column("requests_per_second", sa.Float(), nullable=True),
        sa.Column("tokens_per_second", sa.Float(), nullable=True),
        sa.Column("peak_memory_mb", sa.Float(), nullable=True),
        sa.Column("engine", sa.String(length=64), nullable=True),
        sa.Column("kv_cache_usage_mb", sa.Float(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmarks_id", "benchmarks", ["id"], unique=False)
    op.create_index("ix_benchmarks_workspace_id", "benchmarks", ["workspace_id"], unique=False)

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=128), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("rouge1", sa.Float(), nullable=True),
        sa.Column("rouge2", sa.Float(), nullable=True),
        sa.Column("rougeL", sa.Float(), nullable=True),
        sa.Column("bert_score", sa.Float(), nullable=True),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("tokens_per_second", sa.Float(), nullable=True),
        sa.Column("peak_memory_mb", sa.Float(), nullable=True),
        sa.Column("cost_per_1k_requests", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiments_id", "experiments", ["id"], unique=False)
    op.create_index("ix_experiments_workspace_id", "experiments", ["workspace_id"], unique=False)


def downgrade() -> None:
    for table in [
        "experiments",
        "benchmarks",
        "quantizations",
        "training_jobs",
        "evaluations",
        "datasets",
        "models",
        "async_jobs",
        "api_keys",
        "workspace_members",
        "workspaces",
        "users",
    ]:
        op.drop_table(table)


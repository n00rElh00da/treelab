"""initial TreeLab schema

Revision ID: 0001_initial
Revises:
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def uuid_col():
    return postgresql.UUID(as_uuid=False), False


def upgrade():
    op.create_table("users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(20), server_default="user", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table("sessions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_token_hash", "sessions", ["token_hash"], unique=True)
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])
    op.create_table("datasets",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("slug", sa.String(120), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("task_type", sa.String(40), server_default="classification", nullable=False),
        sa.Column("target_column", sa.String(200), nullable=True),
        sa.Column("active_version_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_datasets_owner_id", "datasets", ["owner_id"])
    op.create_index("ix_datasets_slug", "datasets", ["slug"])
    op.create_index("ix_datasets_is_demo", "datasets", ["is_demo"])
    op.create_table("dataset_versions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("column_count", sa.Integer(), nullable=False),
        sa.Column("schema_json", postgresql.JSONB(), nullable=False),
        sa.Column("preview_json", postgresql.JSONB(), nullable=False),
        sa.Column("raw_csv", sa.LargeBinary(), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("dataset_id", "version_number", name="uq_dataset_version_number"),
    )
    op.create_index("ix_dataset_versions_dataset_id", "dataset_versions", ["dataset_id"])
    op.create_index("ix_dataset_versions_content_hash", "dataset_versions", ["content_hash"])
    op.create_foreign_key("fk_datasets_active_version", "datasets", "dataset_versions", ["active_version_id"], ["id"], ondelete="SET NULL")
    op.create_table("preprocessing_configs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_version_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target", sa.String(200), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("missing", sa.String(80), nullable=False),
        sa.Column("encoding", sa.String(80), nullable=False),
        sa.Column("scaling", sa.String(80), nullable=False),
        sa.Column("duplicates", sa.String(80), nullable=False),
        sa.Column("imbalance", sa.String(80), nullable=False),
        sa.Column("split", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_preprocessing_configs_owner_id", "preprocessing_configs", ["owner_id"])
    op.create_index("ix_preprocessing_configs_dataset_version_id", "preprocessing_configs", ["dataset_version_id"])
    op.create_table("experiments",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_version_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_type", sa.String(120), nullable=False),
        sa.Column("status", sa.String(30), server_default="completed", nullable=False),
        sa.Column("parameters", postgresql.JSONB(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("explainability", postgresql.JSONB(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    for column in ("owner_id", "dataset_id", "dataset_version_id"):
        op.create_index(f"ix_experiments_{column}", "experiments", [column])
    op.create_table("model_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("serialized_model", sa.LargeBinary(), nullable=False),
        sa.Column("framework", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("experiment_id", name="uq_model_artifact_experiment"),
    )
    op.create_index("ix_model_artifacts_owner_id", "model_artifacts", ["owner_id"])


def downgrade():
    op.drop_table("model_artifacts")
    op.drop_table("experiments")
    op.drop_table("preprocessing_configs")
    op.drop_constraint("fk_datasets_active_version", "datasets", type_="foreignkey")
    op.drop_table("dataset_versions")
    op.drop_table("datasets")
    op.drop_table("sessions")
    op.drop_table("users")

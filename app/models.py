from datetime import datetime
from typing import Any
from uuid import uuid4
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


def uuid_column() -> Mapped[str]:
    return mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = uuid_column()
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20), default="user", server_default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="owner")
    sessions: Mapped[list["SessionRecord"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class SessionRecord(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = uuid_column()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[User] = relationship(back_populates="sessions")


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = uuid_column()
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    slug: Mapped[str] = mapped_column(String(120), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    task_type: Mapped[str] = mapped_column(String(40), default="classification", server_default="classification")
    target_column: Mapped[str | None] = mapped_column(String(200), nullable=True)
    active_version_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    owner: Mapped[User | None] = relationship(back_populates="datasets", foreign_keys=[owner_id])
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset", foreign_keys="DatasetVersion.dataset_id", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = uuid_column()
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    schema_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    preview_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    raw_csv: Mapped[bytes] = mapped_column(LargeBinary)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    dataset: Mapped[Dataset] = relationship(back_populates="versions", foreign_keys=[dataset_id])

    __table_args__ = (UniqueConstraint("dataset_id", "version_number", name="uq_dataset_version_number"),)


class PreprocessingConfig(Base):
    __tablename__ = "preprocessing_configs"
    id: Mapped[str] = uuid_column()
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True)
    target: Mapped[str] = mapped_column(String(200))
    features: Mapped[list[str]] = mapped_column(JSONB)
    missing: Mapped[str] = mapped_column(String(80), default="Median imputation")
    encoding: Mapped[str] = mapped_column(String(80), default="One-hot encoding")
    scaling: Mapped[str] = mapped_column(String(80), default="Standard scaler")
    duplicates: Mapped[str] = mapped_column(String(80), default="Keep first row")
    imbalance: Mapped[str] = mapped_column(String(80), default="Balanced class weights")
    split: Mapped[int] = mapped_column(Integer, default=80)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = uuid_column()
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True)
    model_type: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), default="completed")
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    explainability: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    artifact: Mapped["ModelArtifact | None"] = relationship(back_populates="experiment", cascade="all, delete-orphan", uselist=False)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    id: Mapped[str] = uuid_column()
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), unique=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    serialized_model: Mapped[bytes] = mapped_column(LargeBinary)
    framework: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    experiment: Mapped[Experiment] = relationship(back_populates="artifact")

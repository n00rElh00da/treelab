from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: EmailStr
    name: str
    role: Literal["admin", "user"]


class RegisterIn(BaseModel):
    email: EmailStr
    name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8, max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    target_column: str | None = Field(default=None, max_length=200)


class PreprocessIn(BaseModel):
    target: str
    features: list[str]
    missing: str = "Median imputation"
    encoding: str = "One-hot encoding"
    scaling: str = "Standard scaler"
    duplicates: str = "Keep first row"
    imbalance: str = "Balanced class weights"
    split: int = Field(default=80, ge=60, le=90)


class TrainIn(PreprocessIn):
    dataset_id: str
    model_type: Literal["Decision Tree", "Random Forest"]
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_tree_criterion(cls, value: dict[str, Any]) -> dict[str, Any]:
        criterion = value.get("criterion")
        if criterion and criterion not in {"gini", "entropy"}:
            raise ValueError("Decision Tree criterion must be gini or entropy")
        return value


class AutoMLIn(PreprocessIn):
    dataset_id: str
    mode: Literal["Controlled", "Extended"] = "Controlled"
    time_budget: int = Field(default=30, ge=1, le=90)
    metric: Literal["Accuracy", "F1 score", "Precision", "Recall"] = "F1 score"
    validation: Literal["Stratified 5-fold", "Stratified 3-fold", "Holdout 20%"] = "Stratified 5-fold"


class DatasetOut(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    target_column: str | None
    row_count: int
    column_count: int
    version_id: str | None
    version_number: int | None
    updated_at: str | None
    is_demo: bool
    owner_id: str | None
    columns: list[dict[str, Any]] = Field(default_factory=list)


class VersionOut(BaseModel):
    id: str
    version_number: int
    original_filename: str
    content_hash: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: list[dict[str, Any]]
    preview: list[dict[str, Any]]
    target_candidates: list[str]


class ExperimentOut(BaseModel):
    id: str
    dataset_id: str
    dataset_version_id: str
    model_type: str
    status: str
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    explainability: dict[str, Any]
    duration_ms: int
    error: str | None
    created_at: str | None

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


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
    missing: Literal["Median imputation", "Mean imputation", "Drop rows", "Flag as feature"] = "Median imputation"
    encoding: Literal["One-hot encoding", "Ordinal encoding"] = "One-hot encoding"
    scaling: Literal["Standard scaler", "Robust scaler", "Min-max scaler", "No scaling"] = "Standard scaler"
    duplicates: Literal["Keep first row", "Drop duplicates"] = "Keep first row"
    imbalance: Literal["Balanced class weights", "No adjustment"] = "Balanced class weights"
    split: int = Field(default=80, ge=60, le=90)


def _validate_integer_parameter(value: Any, name: str, minimum: int, allow_none: bool = False) -> None:
    if allow_none and (value is None or value == "None"):
        return
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer of at least {minimum}") from exc
    if integer < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


class TrainIn(PreprocessIn):
    dataset_id: str
    model_type: Literal["Decision Tree", "Random Forest"]
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> "TrainIn":
        criterion = self.parameters.get("criterion")
        if criterion and criterion not in {"gini", "entropy"}:
            raise ValueError("Decision Tree criterion must be gini or entropy")
        if "classWeight" in self.parameters and self.parameters["classWeight"] not in {"Balanced", "None"}:
            raise ValueError("parameters.classWeight must be Balanced or None")
        for key, minimum in (("minSplit", 2), ("minLeaf", 1), ("estimators", 1)):
            if key in self.parameters:
                _validate_integer_parameter(self.parameters[key], f"parameters.{key}", minimum)
        if "maxDepth" in self.parameters:
            _validate_integer_parameter(self.parameters["maxDepth"], "parameters.maxDepth", 1, allow_none=True)
        return self


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

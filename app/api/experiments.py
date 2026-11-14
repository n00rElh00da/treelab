from __future__ import annotations
import io
import pickle
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from ..auth import get_current_user
from ..db import get_db
from ..models import DatasetVersion, Experiment, ModelArtifact, PreprocessingConfig, User
from ..schemas import AutoMLIn, ExperimentOut, TrainIn
from ..services.automl import run_automl
from ..services.training import TrainingError, train_classifier
from .datasets import accessible_dataset, active_version

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def frame_for(version: DatasetVersion) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(version.raw_csv))


def config_from(payload: TrainIn | AutoMLIn) -> dict[str, Any]:
    return {"target": payload.target, "features": payload.features, "missing": payload.missing, "encoding": payload.encoding, "scaling": payload.scaling, "duplicates": payload.duplicates, "imbalance": payload.imbalance, "split": payload.split}


def to_out(experiment: Experiment) -> ExperimentOut:
    return ExperimentOut(id=experiment.id, dataset_id=experiment.dataset_id, dataset_version_id=experiment.dataset_version_id, model_type=experiment.model_type, status=experiment.status, parameters=experiment.parameters or {}, metrics=experiment.metrics or {}, explainability=experiment.explainability or {}, duration_ms=experiment.duration_ms, error=experiment.error, created_at=experiment.created_at.isoformat() if experiment.created_at else None)


def save_result(db: Session, user: User, dataset_id: str, version: DatasetVersion, model_type: str, parameters: dict[str, Any], result: dict[str, Any]) -> Experiment:
    experiment = Experiment(owner_id=user.id, dataset_id=dataset_id, dataset_version_id=version.id, model_type=model_type, status="completed", parameters=clean(parameters), metrics=clean(result["metrics"]), explainability=clean(result["explainability"]), duration_ms=result["duration_ms"])
    db.add(experiment)
    db.flush()
    db.add(ModelArtifact(experiment_id=experiment.id, owner_id=user.id, serialized_model=pickle.dumps(result["pipeline"]), framework="scikit-learn / FLAML" if model_type.startswith("AutoML") else "scikit-learn"))
    db.commit()
    db.refresh(experiment)
    return experiment


@router.get("", response_model=list[ExperimentOut])
def list_experiments(dataset_id: str | None = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = select(Experiment).where(Experiment.owner_id == user.id)
    if dataset_id:
        query = query.where(Experiment.dataset_id == dataset_id)
    return [to_out(item) for item in db.scalars(query.order_by(Experiment.created_at.desc()))]


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment(experiment_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    experiment = db.scalar(select(Experiment).where(Experiment.id == experiment_id, Experiment.owner_id == user.id))
    if not experiment:
        raise HTTPException(404, "Experiment not found")
    return to_out(experiment)


@router.post("/train", response_model=ExperimentOut, status_code=201)
def train(payload: TrainIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, payload.dataset_id, user)
    version = active_version(db, dataset)
    if not version:
        raise HTTPException(400, "Dataset has no active version")
    config = config_from(payload)
    try:
        result = train_classifier(frame_for(version), payload.model_type, config, payload.parameters)
    except TrainingError as exc:
        raise HTTPException(422, str(exc)) from exc
    db.add(PreprocessingConfig(owner_id=user.id, dataset_version_id=version.id, **config))
    return save_result(db, user, dataset.id, version, payload.model_type, {**payload.parameters, "preprocessing": config}, result)


@router.post("/automl", response_model=ExperimentOut, status_code=201)
def automl(payload: AutoMLIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, payload.dataset_id, user)
    version = active_version(db, dataset)
    if not version:
        raise HTTPException(400, "Dataset has no active version")
    config = config_from(payload)
    try:
        result = run_automl(frame_for(version), config, payload.mode, payload.time_budget, payload.metric)
    except TrainingError as exc:
        raise HTTPException(422, str(exc)) from exc
    db.add(PreprocessingConfig(owner_id=user.id, dataset_version_id=version.id, **config))
    return save_result(db, user, dataset.id, version, result["model_type"], {**result["parameters"], "preprocessing": config, "validation": payload.validation}, result)

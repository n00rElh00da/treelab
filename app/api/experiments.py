from __future__ import annotations
import json
import logging
import pickle
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..auth import get_current_user
from ..db import get_db
from ..models import DatasetVersion, Experiment, ModelArtifact, PreprocessingConfig, User
from ..schemas import AutoMLIn, ExperimentOut, TrainIn
from ..services.automl import run_automl
from ..services.profiling import parse_csv
from ..services.training import TrainingError, train_classifier
from .datasets import accessible_dataset, active_version

router = APIRouter(prefix="/api/experiments", tags=["experiments"])
logger = logging.getLogger(__name__)


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


def validate_json_payload(value: Any, label: str) -> Any:
    cleaned = clean(value)
    try:
        json.dumps(cleaned, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"{label} contains a non-serializable or non-finite value") from exc
    return cleaned


def frame_for(raw_csv: bytes, original_filename: str) -> pd.DataFrame:
    return parse_csv(raw_csv, original_filename)["frame"]


def config_from(payload: TrainIn | AutoMLIn) -> dict[str, Any]:
    return {"target": payload.target, "features": payload.features, "missing": payload.missing, "encoding": payload.encoding, "scaling": payload.scaling, "duplicates": payload.duplicates, "imbalance": payload.imbalance, "split": payload.split}


def model_parameters(payload: TrainIn, config: dict[str, Any]) -> dict[str, Any]:
    return {**payload.parameters, "classWeight": "Balanced" if config["imbalance"] == "Balanced class weights" else "None"}


def to_out(experiment: Experiment) -> ExperimentOut:
    return ExperimentOut(id=experiment.id, dataset_id=experiment.dataset_id, dataset_version_id=experiment.dataset_version_id, model_type=experiment.model_type, status=experiment.status, parameters=experiment.parameters or {}, metrics=experiment.metrics or {}, explainability=experiment.explainability or {}, duration_ms=experiment.duration_ms, error=experiment.error, created_at=experiment.created_at.isoformat() if experiment.created_at else None)


def save_result(db: Session, user: User, dataset_id: str, version_id: str, model_type: str, parameters: dict[str, Any], result: dict[str, Any]) -> Experiment:
    try:
        experiment = Experiment(
            owner_id=user.id,
            dataset_id=dataset_id,
            dataset_version_id=version_id,
            model_type=model_type,
            status="completed",
            parameters=validate_json_payload(parameters, "Experiment parameters"),
            metrics=validate_json_payload(result["metrics"], "Experiment metrics"),
            explainability=validate_json_payload(result["explainability"], "Experiment explainability"),
            duration_ms=result["duration_ms"],
        )
        db.add(experiment)
        db.flush()
        db.add(
            ModelArtifact(
                experiment_id=experiment.id,
                owner_id=user.id,
                serialized_model=pickle.dumps(result["pipeline"]),
                framework="scikit-learn / FLAML" if model_type.startswith("AutoML") else "scikit-learn",
            )
        )
        db.commit()
        db.refresh(experiment)
        return experiment
    except Exception:
        db.rollback()
        raise


def save_failure(db: Session, user: User, dataset_id: str, version_id: str, model_type: str, parameters: dict[str, Any], error: str) -> bool:
    try:
        db.add(
            Experiment(
                owner_id=user.id,
                dataset_id=dataset_id,
                dataset_version_id=version_id,
                model_type=model_type,
                status="failed",
                parameters=validate_json_payload(parameters, "Failed experiment parameters"),
                metrics={},
                explainability={},
                duration_ms=0,
                error=error,
            )
        )
        db.commit()
        return True
    except Exception:
        db.rollback()
        logger.exception("AutoML failure record persistence failed")
        return False


def persist_failure(
    db: Session,
    user: User,
    dataset_id: str,
    version_id: str,
    model_type: str,
    parameters: dict[str, Any],
    message: str,
) -> None:
    if not save_failure(db, user, dataset_id, version_id, model_type, parameters, message):
        logger.error("AutoML failed without a persisted failure record")


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


@router.get("/{experiment_id}/model")
def get_model(experiment_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    artifact = db.scalar(select(ModelArtifact).where(ModelArtifact.experiment_id == experiment_id, ModelArtifact.owner_id == user.id))
    if not artifact:
        raise HTTPException(404, "Model artifact not found")
    return Response(content=artifact.serialized_model, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="treelab-{experiment_id}.pkl"'})


@router.post("/train", response_model=ExperimentOut, status_code=201)
def train(payload: TrainIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, payload.dataset_id, user)
    version = active_version(db, dataset)
    if not version:
        raise HTTPException(400, "Dataset has no active version")
    dataset_id = dataset.id
    version_id = version.id
    raw_csv = version.raw_csv
    original_filename = version.original_filename
    db.rollback()
    config = config_from(payload)
    effective_parameters = model_parameters(payload, config)
    parameters = {**effective_parameters, "preprocessing": config}
    try:
        result = train_classifier(frame_for(raw_csv, original_filename), payload.model_type, config, effective_parameters)
    except TrainingError as exc:
        persist_failure(db, user, dataset_id, version_id, payload.model_type, parameters, str(exc))
        raise HTTPException(422, str(exc)) from exc
    db.add(PreprocessingConfig(owner_id=user.id, dataset_version_id=version_id, **config))
    return to_out(save_result(db, user, dataset_id, version_id, payload.model_type, parameters, result))


@router.post("/automl", response_model=ExperimentOut, status_code=201)
def automl(payload: AutoMLIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, payload.dataset_id, user)
    version = active_version(db, dataset)
    if not version:
        raise HTTPException(400, "Dataset has no active version")
    dataset_id = dataset.id
    version_id = version.id
    raw_csv = version.raw_csv
    original_filename = version.original_filename
    db.rollback()
    config = config_from(payload)
    parameters = {"mode": payload.mode, "time_budget": payload.time_budget, "metric": payload.metric, "validation": payload.validation, "preprocessing": config}
    logger.info("AutoML stage=start mode=%s budget=%s", payload.mode, payload.time_budget)
    try:
        logger.info("AutoML stage=csv_parse")
        frame = frame_for(raw_csv, original_filename)
        logger.info("AutoML stage=search")
        result = run_automl(frame, {**config, "validation": payload.validation}, payload.mode, payload.time_budget, payload.metric)
        logger.info("AutoML stage=result_validation")
        parameters = validate_json_payload(
            {**parameters, "result_parameters": result.get("parameters", {})},
            "AutoML parameters",
        )
        validate_json_payload(result["metrics"], "AutoML metrics")
        validate_json_payload(result["explainability"], "AutoML explainability")
        logger.info("AutoML stage=persistence")
    except TrainingError as exc:
        persist_failure(db, user, dataset_id, version_id, f"AutoML · {payload.mode}", parameters, str(exc))
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("AutoML stage=unexpected_failure mode=%s", payload.mode)
        persist_failure(
            db,
            user,
            dataset_id,
            version_id,
            f"AutoML · {payload.mode}",
            parameters,
            "AutoML execution failed",
        )
        raise HTTPException(500, "AutoML execution failed") from exc
    db.add(PreprocessingConfig(owner_id=user.id, dataset_version_id=version_id, **config))
    try:
        experiment = save_result(
            db,
            user,
            dataset_id,
            version_id,
            result["model_type"],
            {**result["parameters"], "preprocessing": config, "validation": payload.validation},
            result,
        )
    except TrainingError as exc:
        persist_failure(
            db,
            user,
            dataset_id,
            version_id,
            f"AutoML · {payload.mode}",
            parameters,
            str(exc),
        )
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("AutoML stage=persistence_failure mode=%s", payload.mode)
        persist_failure(
            db,
            user,
            dataset_id,
            version_id,
            f"AutoML · {payload.mode}",
            parameters,
            "AutoML persistence failed",
        )
        raise HTTPException(500, "AutoML persistence failed") from exc
    logger.info("AutoML stage=complete mode=%s experiment_id=%s", payload.mode, experiment.id)
    return to_out(experiment)

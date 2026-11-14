from __future__ import annotations
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from ..auth import get_current_user
from ..db import get_db
from ..models import Dataset, DatasetVersion, User
from ..schemas import DatasetOut, DatasetUpdate, VersionOut
from ..services.profiling import parse_csv

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def accessible_dataset(db: Session, dataset_id: str, user: User) -> Dataset:
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.deleted_at.is_(None), or_(Dataset.owner_id == user.id, Dataset.is_demo.is_(True))))
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset


def active_version(db: Session, dataset: Dataset) -> DatasetVersion | None:
    if not dataset.active_version_id:
        return None
    return db.scalar(select(DatasetVersion).where(DatasetVersion.id == dataset.active_version_id))


def to_dataset_out(db: Session, dataset: Dataset) -> DatasetOut:
    version = active_version(db, dataset)
    return DatasetOut(id=dataset.id, name=dataset.name, description=dataset.description or "", task_type=dataset.task_type, target_column=dataset.target_column, row_count=version.row_count if version else 0, column_count=version.column_count if version else 0, version_id=version.id if version else None, version_number=version.version_number if version else None, updated_at=dataset.updated_at.isoformat() if dataset.updated_at else None, is_demo=dataset.is_demo, owner_id=dataset.owner_id, columns=version.schema_json if version else [])


def to_version_out(version: DatasetVersion, target_candidates: list[str] | None = None) -> VersionOut:
    candidates = target_candidates or [item["name"] for item in version.schema_json if item.get("is_target_candidate")]
    return VersionOut(id=version.id, version_number=version.version_number, original_filename=version.original_filename, content_hash=version.content_hash, size_bytes=version.size_bytes, row_count=version.row_count, column_count=version.column_count, columns=version.schema_json, preview=version.preview_json, target_candidates=candidates)


@router.get("", response_model=list[DatasetOut])
def list_datasets(search: str = "", db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = select(Dataset).where(Dataset.deleted_at.is_(None), or_(Dataset.owner_id == user.id, Dataset.is_demo.is_(True))).order_by(Dataset.updated_at.desc())
    datasets = list(db.scalars(query))
    if search:
        term = search.lower()
        datasets = [item for item in datasets if term in f"{item.name} {item.description} {item.target_column or ''}".lower()]
    return [to_dataset_out(db, item) for item in datasets]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return to_dataset_out(db, accessible_dataset(db, dataset_id, user))


@router.post("", response_model=DatasetOut, status_code=201)
async def create_dataset(name: str = Form(...), description: str = Form(""), target_column: str | None = Form(None), file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    content = await file.read()
    profile = parse_csv(content, file.filename or "dataset.csv")
    target = target_column or profile["target"]
    if target and target not in [item["name"] for item in profile["schema"]]:
        raise HTTPException(400, "Selected target column does not exist")
    dataset = Dataset(owner_id=user.id, slug=f"upload-{profile['content_hash'][:12]}", name=name.strip() or (file.filename or "Uploaded dataset"), description=description.strip(), task_type="classification", target_column=target, is_demo=False)
    db.add(dataset)
    db.flush()
    version = DatasetVersion(dataset_id=dataset.id, version_number=1, original_filename=file.filename or "dataset.csv", content_hash=profile["content_hash"], size_bytes=profile["size_bytes"], row_count=profile["row_count"], column_count=profile["column_count"], schema_json=profile["schema"], preview_json=profile["preview"], raw_csv=content, created_by=user.id)
    db.add(version)
    db.flush()
    dataset.active_version_id = version.id
    db.commit()
    db.refresh(dataset)
    return to_dataset_out(db, dataset)


@router.post("/{dataset_id}/versions", response_model=DatasetOut, status_code=201)
async def add_version(dataset_id: str, file: UploadFile = File(...), target_column: str | None = Form(None), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, dataset_id, user)
    if dataset.is_demo or dataset.owner_id != user.id:
        raise HTTPException(403, "Demo datasets cannot be modified")
    content = await file.read()
    profile = parse_csv(content, file.filename or "dataset.csv")
    target = target_column or dataset.target_column or profile["target"]
    if target not in [item["name"] for item in profile["schema"]]:
        raise HTTPException(400, "Selected target column does not exist")
    last = db.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id).order_by(DatasetVersion.version_number.desc()))
    version = DatasetVersion(dataset_id=dataset.id, version_number=(last.version_number + 1 if last else 1), original_filename=file.filename or "dataset.csv", content_hash=profile["content_hash"], size_bytes=profile["size_bytes"], row_count=profile["row_count"], column_count=profile["column_count"], schema_json=profile["schema"], preview_json=profile["preview"], raw_csv=content, created_by=user.id)
    db.add(version)
    db.flush()
    dataset.active_version_id = version.id
    dataset.target_column = target
    dataset.updated_at = datetime.now(timezone.utc)
    db.commit()
    return to_dataset_out(db, dataset)


@router.get("/{dataset_id}/versions/{version_id}/preview", response_model=VersionOut)
def get_preview(dataset_id: str, version_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, dataset_id, user)
    version = db.scalar(select(DatasetVersion).where(DatasetVersion.id == version_id, DatasetVersion.dataset_id == dataset.id))
    if not version:
        raise HTTPException(404, "Dataset version not found")
    return to_version_out(version)


@router.patch("/{dataset_id}", response_model=DatasetOut)
def update_dataset(dataset_id: str, payload: DatasetUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, dataset_id, user)
    if dataset.is_demo or dataset.owner_id != user.id:
        raise HTTPException(403, "Demo datasets cannot be modified")
    for field in ("name", "description", "target_column"):
        value = getattr(payload, field)
        if value is not None:
            if field == "target_column":
                version = active_version(db, dataset)
                columns = [item["name"] for item in version.schema_json] if version else []
                if value not in columns:
                    raise HTTPException(400, "Selected target column does not exist")
            setattr(dataset, field, value.strip() if isinstance(value, str) else value)
    dataset.updated_at = datetime.now(timezone.utc)
    db.commit()
    return to_dataset_out(db, dataset)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = accessible_dataset(db, dataset_id, user)
    if dataset.is_demo or dataset.owner_id != user.id:
        raise HTTPException(403, "Demo datasets cannot be deleted")
    dataset.deleted_at = datetime.now(timezone.utc)
    db.commit()

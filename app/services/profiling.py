from __future__ import annotations
import hashlib
import io
from typing import Any
import pandas as pd
from fastapi import HTTPException
from ..config import get_settings

settings = get_settings()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def infer_target(columns: list[str], frame: pd.DataFrame) -> tuple[str | None, list[str]]:
    normalized = {str(c).strip().lower(): c for c in columns}
    preferred = ["target", "label", "class", "y", "churn", "outcome", "diagnosis"]
    candidates = [normalized[name] for name in preferred if name in normalized]
    for column in columns:
        nunique = int(frame[column].nunique(dropna=True))
        if column not in candidates and 2 <= nunique <= min(20, max(2, len(frame) // 10)):
            candidates.append(column)
    return (candidates[0] if candidates else None, candidates)


def parse_csv(content: bytes, filename: str) -> dict[str, Any]:
    if not content:
        raise HTTPException(400, "The uploaded CSV is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"CSV exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit")
    frame = None
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            last_error = exc
    if frame is None:
        raise HTTPException(400, f"Unable to parse CSV: {last_error}")
    if frame.empty:
        raise HTTPException(400, "The uploaded CSV has no data rows")
    if len(frame) > settings.max_upload_rows:
        raise HTTPException(413, f"CSV exceeds the {settings.max_upload_rows:,} row limit")
    if len(frame.columns) > settings.max_upload_columns:
        raise HTTPException(413, f"CSV exceeds the {settings.max_upload_columns} column limit")
    columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in columns) or len(set(columns)) != len(columns):
        raise HTTPException(400, "CSV columns must be non-empty and unique")
    frame.columns = columns
    if any(frame[column].notna().sum() == 0 for column in columns):
        raise HTTPException(400, "CSV contains an entirely empty column")
    target, candidates = infer_target(columns, frame)
    schema = []
    for column in columns:
        series = frame[column]
        kind = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        schema.append({
            "name": column,
            "kind": kind,
            "dtype": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "sample_values": [_json_value(item) for item in series.dropna().head(3).tolist()],
            "is_target_candidate": column in candidates,
        })
    preview = [{column: _json_value(value) for column, value in row.items()} for row in frame.head(10).to_dict(orient="index").values()]
    return {
        "frame": frame,
        "content_hash": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "schema": schema,
        "preview": preview,
        "target": target,
        "target_candidates": candidates,
        "row_count": int(len(frame)),
        "column_count": int(len(columns)),
    }

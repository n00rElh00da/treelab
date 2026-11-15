from __future__ import annotations
import hashlib
import io
import csv
from pathlib import Path
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
        nunique = frame[column].dropna().drop_duplicates().shape[0]
        if column not in candidates and 2 <= nunique <= min(20, max(2, len(frame) // 10)):
            candidates.append(column)
    return (candidates[0] if candidates else None, candidates)


def _decode_csv(content: bytes) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise HTTPException(400, f"Unable to decode CSV: {last_error}")


def _detect_delimiter(text: str) -> str:
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def parse_csv(content: bytes, filename: str) -> dict[str, Any]:
    if not content:
        raise HTTPException(400, "The uploaded CSV is empty")
    if filename and Path(filename).suffix.lower() not in {"", ".csv"}:
        raise HTTPException(400, "Only .csv files are supported")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"CSV exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit")
    text = _decode_csv(content)
    delimiter = _detect_delimiter(text)
    header_reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    raw_header = next(header_reader, None)
    if not raw_header:
        raise HTTPException(400, "The uploaded CSV has no header row")
    columns = [str(column).strip() for column in raw_header]
    if len(columns) < 2:
        raise HTTPException(400, "CSV must contain at least one target column and one feature column")
    if any(not column for column in columns) or len(set(columns)) != len(columns):
        raise HTTPException(400, "CSV columns must be non-empty and unique")
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise HTTPException(400, f"Unable to parse CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(400, "The uploaded CSV has no data rows")
    if len(frame) > settings.max_upload_rows:
        raise HTTPException(413, f"CSV exceeds the {settings.max_upload_rows:,} row limit")
    if len(frame.columns) > settings.max_upload_columns:
        raise HTTPException(413, f"CSV exceeds the {settings.max_upload_columns} column limit")
    if len(frame.columns) != len(columns):
        raise HTTPException(400, "CSV rows do not match the header column count")
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
            "missing": series.isna().to_numpy().sum().item(),
            "unique": series.dropna().drop_duplicates().shape[0],
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

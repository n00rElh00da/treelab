from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from .api import auth, datasets, experiments
from .config import get_settings
from .db import engine

settings = get_settings()
ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="TreeLab API", version="1.0.0")
if settings.cors_origin_list:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(experiments.router)


@app.get("/api/health")
def health():
    if engine is None:
        return {"status": "degraded", "database": "unconfigured", "detail": "DATABASE_URL is not set"}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        return {"status": "degraded", "database": "unavailable", "detail": type(exc).__name__}


@app.get("/manus-routes.json")
def route_manifest():
    return FileResponse(ROOT / "public" / "manus-routes.json", media_type="application/json")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

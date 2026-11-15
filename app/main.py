from __future__ import annotations
from pathlib import Path
from urllib.parse import urlsplit
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from .api import auth, datasets, experiments
from .config import get_settings
from .db import engine

settings = get_settings()
ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="TreeLab API", version="1.0.0")


@app.middleware("http")
async def same_origin_mutations(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        candidate = request.headers.get("origin") or request.headers.get("referer")
        if candidate:
            parsed = urlsplit(candidate)
            forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
            forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
            request_origin = f"{forwarded_proto}://{forwarded_host}" if forwarded_host else ""
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin not in set(settings.cors_origin_list) and origin != request_origin:
                return JSONResponse(status_code=403, content={"detail": "Cross-origin mutation rejected"})
    return await call_next(request)


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

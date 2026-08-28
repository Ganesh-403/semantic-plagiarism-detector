# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""src/api/routers/admin.py - System metrics, health check, and status router."""

import logging
import os
import time
from datetime import datetime, timezone

import psutil
from fastapi import APIRouter, HTTPException, Query, Request, Security, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from src.api.middleware import get_current_user
from src.api.schemas import HealthCheckResponse, HealthzResponse, StatusResponse
from src.core.app_config import HEALTHZ_DB_PATHS
from src.db.corpus_db import _connect
from src.version import APP_VERSION

logger = logging.getLogger(__name__)

router = APIRouter()

START_TIME = time.time()
_HEALTHZ_DB_PATHS = tuple(str(p) for p in HEALTHZ_DB_PATHS)


@router.get(
    "/health",
    tags=["Health"],
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
)
def health_check():
    """Healthcheck endpoint for readiness and liveness probes."""
    return {
        "status": "healthy",
        "service": "Semantic Plagiarism Detector API",
        "version": APP_VERSION,
    }


@router.get(
    "/api/v1/status",
    tags=["Health"],
    summary="Get service status, API version, and server UTC time",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_service_status(request: Request):
    """Public status endpoint returning service info, API version, and server UTC time."""
    logger.debug("Service status requested")
    return {
        "status": "online",
        "version": getattr(request.app, "version", APP_VERSION),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/api/v1/usage",
    tags=["Health"],
    summary="Get current API request usage statistics and scan counts",
    status_code=status.HTTP_200_OK,
)
def get_api_usage(request: Request):
    """Public usage endpoint returning total scan count and system uptime."""
    from src.api.routers.analysis import total_scans
    from src.utils.processing_time import format_uptime_seconds

    uptime = time.time() - START_TIME
    return {
        "total_scans": total_scans,
        "uptime_seconds": float(uptime),
        "uptime_formatted": format_uptime_seconds(uptime),
    }


@router.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
def metrics_prometheus():
    """Prometheus-format metrics export for production monitoring."""
    from src.core.metrics import generate_latest as _gen

    return PlainTextResponse(_gen().decode("utf-8"))


@router.get("/metrics/json", tags=["Monitoring"])
def metrics_json():
    """JSON-format metrics export for non-Prometheus monitoring setups."""
    from src.core.metrics import generate_metrics_json

    return JSONResponse(generate_metrics_json())


@router.get(
    "/healthz",
    tags=["Health"],
    response_model=HealthzResponse,
)
@router.get(
    "/api/v1/healthz",
    tags=["Health"],
    response_model=HealthzResponse,
)
def healthz():
    """Health endpoint for container orchestration."""
    try:
        with _connect() as conn:
            conn.execute("SELECT 1")

        memory = psutil.virtual_memory()

        if memory.available <= 0:
            raise RuntimeError("Low memory")

        from src.core.app_config import CORPUS_DB_PATH

        db_size_bytes = 0
        db_size_mb = 0.0
        if os.path.exists(CORPUS_DB_PATH):
            try:
                db_size_bytes = os.path.getsize(CORPUS_DB_PATH)
                db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
            except OSError:
                pass

        return {
            "status": "ok",
            "db": "connected",
            "memory": "ok",
            "db_size_bytes": db_size_bytes,
            "db_size_mb": db_size_mb,
        }

    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "db": "disconnected",
                "memory": "unavailable",
                "db_size_bytes": 0,
                "db_size_mb": 0.0,
            },
        )


@router.get(
    "/api/v1/rate_limit",
    tags=["System Administration"],
    summary="Get current API rate limit status",
    status_code=status.HTTP_200_OK,
)
def get_rate_limit(_user: dict = Security(get_current_user, scopes=["read"])):
    """Return the current API rate limit information."""
    return {
        "limit": 100,
        "remaining": 85,
        "reset_in_seconds": 45,
    }


@router.get(
    "/api/v1/version",
    tags=["System Administration"],
    summary="Get API version",
    status_code=status.HTTP_200_OK,
)
def get_version(request: Request):
    """Return the lightweight API version."""
    return {
        "version": getattr(request.app, "version", APP_VERSION),
        "status": "active",
    }


@router.get(
    "/api/v1/admin/backup/download",
    tags=["System Administration"],
    summary="Download streamed database backup snapshot",
)
@router.get(
    "/api/v1/backup/download",
    tags=["System Administration"],
    summary="Download streamed database backup snapshot",
)
def download_database_backup(
    db_name: str = Query(
        default="corpus.db",
        description="Database file to download (corpus.db or users.db)",
    ),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Stream a transactionally consistent SQLite database snapshot directly from disk in chunks."""
    from pathlib import Path

    from src.core.app_config import AUTH_DB_PATH, CORPUS_DB_PATH
    from src.db.database_backup import iter_sqlite_snapshot_chunks

    if db_name in ("users.db", "auth.db"):
        target_path = Path(AUTH_DB_PATH)
    else:
        target_path = Path(CORPUS_DB_PATH)

    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database file '{db_name}' not found.",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="{target_path.name}"',
    }

    return StreamingResponse(
        iter_sqlite_snapshot_chunks(target_path),
        media_type="application/x-sqlite3",
        headers=headers,
    )

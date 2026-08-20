"""src/api/app.py - FastAPI REST API for LMS integration."""

import logging
import os
import sqlite3
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Query, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import (
    custom_rate_limit_exceeded_handler,
    get_current_user,
    limiter,
)
from src.api.middleware import verify_bearer_token
from src.api.routers import (
    admin_router,
    analysis_router,
    auth_router,
    clustering_router,  # ← ADDED FOR ISSUE #2811
    corpus_router,
)
from src.version import APP_VERSION

# Re-exports for backward compatibility with existing tests and scripts
logger = logging.getLogger(__name__)

# ── API Initialization ────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Plagiarism Detector API",
    description="REST API for programmatically checking documents for semantic plagiarism.",
    version=APP_VERSION,
    contact={
        "name": "API Support",
        "url": "http://example.com/support",
        "email": "support@example.com",
    },
    openapi_tags=[
        {"name": "Authentication", "description": "Authenticate user"},
        {"name": "Plagiarism Detection", "description": "Scanning operations"},
        {"name": "System Administration", "description": "Admin operations"},
        {"name": "Health", "description": "Health checks"},
        {"name": "Clustering", "description": "Background clustering operations"},
    ],
    dependencies=[Depends(verify_bearer_token)],
)

# Enable CORS for external LMS frontends
origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
if origins.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        origin.strip() for origin in origins.split(",") if origin.strip()
    ]

# Browser spec: allow_credentials cannot be True when wildcard '*' is used in allowed_origins
allow_credentials = False if "*" in allowed_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# SlowAPI Rate Limiting setup
app.state.limiter = limiter


@app.middleware("http")
async def otel_tracing_middleware(request: Request, call_next):
    """Middleware to create an OpenTelemetry root span for every HTTP request."""
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                try:
                    from src.security.jwt_utils import verify_access_token
                    payload = verify_access_token(token)
                    user_id = str(
                        payload.get("sub")
                        or payload.get("user_id")
                        or payload.get("username")
                        or ""
                    )
                except Exception:
                    pass

    if not user_id:
        user_id = "anonymous"

    request.state.user_id = user_id

    try:
        from src.utils.tracing import get_tracer
        tracer = get_tracer()
    except Exception:
        tracer = None

    if not tracer:
        return await call_next(request)

    request_id = request.headers.get("X-Request-ID", "unknown")
    span_name = f"HTTP {request.method} {request.url.path}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.route", request.url.path)
        span.set_attribute("http.request_id", request_id)
        span.set_attribute("user.id", user_id)

        try:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            final_user_id = getattr(request.state, "user_id", user_id)
            if final_user_id:
                span.set_attribute("user.id", str(final_user_id))
            return response
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("http.status_code", 500)
            raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors from malformed API requests."""
    logger.warning(
        "Request validation failed for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "message": "Validation failed.",
            "details": [
                {
                    "field": ".".join(map(str, err["loc"])),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ],
        },
    )

@app.exception_handler(404)
async def not_found_handler(request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": True,
            "code": 404,
            "message": "API endpoint or resource not found",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions by returning a standardized 400 Bad Request JSON response."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": True,
            "code": status.HTTP_400_BAD_REQUEST,
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(sqlite3.OperationalError)
async def sqlite_operational_error_handler(request: Request, exc: sqlite3.OperationalError):
    """Handle sqlite3.OperationalError, particularly database is locked, returning 503 Service Unavailable."""
    err_msg = str(exc)
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE if "locked" in err_msg.lower() or "busy" in err_msg.lower() else status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "Service busy, please retry" if status_code == status.HTTP_503_SERVICE_UNAVAILABLE else f"Database error: {err_msg}"

    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"
    logger.error(f"SQLite operational error: {exc}", exc_info=not is_production)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "code": status_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler that returns a standardized JSON error payload for any unhandled exception."""
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"

    logging.getLogger(__name__).error(
        f"Unhandled exception: {exc}", exc_info=not is_production
    )

    message = "An internal server error occurred." if is_production else str(exc)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "code": status_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP errors to return standardized JSON payloads."""
    status_code = exc.status_code
    if status_code == 404:
        message = "API endpoint or resource not found"
    else:
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    log_level = logging.WARNING if 400 <= status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        "HTTP %d error on %s %s: %s",
        status_code,
        request.method,
        request.url.path,
        message,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "code": status_code,
            "message": message,
        },
    )


app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── Register Sub-Routers ──────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(corpus_router)
app.include_router(clustering_router)  # ← ADDED FOR ISSUE #2811
app.include_router(admin_router)

# ── Audit Events Endpoint (Issue #2732) ───────────────────────────────────────

@app.get(
    "/api/v1/audit/events",
    tags=["System Administration"],
    summary="Get paginated security audit events",
    status_code=status.HTTP_200_OK,
)
def get_audit_events_api(
    limit: int = Query(default=20, ge=1, le=100, description="Max events per page"),
    offset: int = Query(default=0, ge=0, description="Number of events to skip (pagination)"),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    username: str | None = Query(default=None, description="Filter by username"),
    _user: dict = Security(get_current_user, scopes=["admin"])
):
    """Retrieve paginated security audit events."""
    from src.db.auth import get_security_audit_log_count, get_security_audit_logs
    
    events = get_security_audit_logs(
        limit=limit,
        offset=offset,
        event_type=event_type,
        username=username
    )
    
    total_count = get_security_audit_log_count(
        event_type=event_type,
        username=username
    )
    
    return {
        "events": events,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total_count": total_count,
            "total_pages": (total_count + limit - 1) // limit if limit > 0 else 0
        }
    }
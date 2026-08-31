"""src/api/app.py - FastAPI REST API for LMS integration."""

import logging
import os
import sqlite3
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Query, Request, Security, status, WebSocket, WebSocketDisconnect
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
    # 1. Check if user_id was pre-set on request.state
    user_id = getattr(request.state, "user_id", None)

    # 2. If missing, attempt to extract token from Authorization header
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
            
            # Use request.scope.get("route").path to extract generalized FastAPI route template for span name
            route = request.scope.get("route")
            if route and hasattr(route, "path"):
                route_path = route.path
                span.update_name(f"HTTP {request.method} {route_path}")
                span.set_attribute("http.route", route_path)

            span.set_attribute("http.status_code", response.status_code)
            # Update user.id if set or modified by route handler/dependencies
            final_user_id = getattr(request.state, "user_id", user_id)
            if final_user_id:
                span.set_attribute("user.id", str(final_user_id))
            return response
        except Exception as exc:
            route = request.scope.get("route")
            if route and hasattr(route, "path"):
                route_path = route.path
                span.update_name(f"HTTP {request.method} {route_path}")
                span.set_attribute("http.route", route_path)
            span.record_exception(exc)
            span.set_attribute("http.status_code", 500)
            raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors from malformed API requests adhering to RFC 7807."""
    # Issue #2564: Log the detailed validation errors for backend debugging
    logger.warning(
        "Request validation failed for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    errors_list = [
        {
            "field": ".".join(map(str, err["loc"])),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "about:blank",
            "title": "Unprocessable Entity",
            "status": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "detail": "Validation failed.",
            "instance": getattr(getattr(request, "url", None), "path", None),
            "error": True,
            "message": "Validation failed.",
            "details": errors_list,
            "invalid_params": errors_list,
        },
        media_type="application/problem+json",
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP 404 errors adhering to RFC 7807."""
    path = getattr(getattr(request, "url", None), "path", None)
    return JSONResponse(
        status_code=404,
        content={
            "type": "about:blank",
            "title": "Not Found",
            "status": 404,
            "detail": "API endpoint or resource not found",
            "instance": path,
            "error": True,
            "code": 404,
            "message": "API endpoint or resource not found",
        },
        media_type="application/problem+json",
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions by returning an RFC 7807 400 Bad Request response."""
    path = getattr(getattr(request, "url", None), "path", None)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "type": "about:blank",
            "title": "Bad Request",
            "status": status.HTTP_400_BAD_REQUEST,
            "detail": str(exc),
            "instance": path,
            "error": True,
            "code": status.HTTP_400_BAD_REQUEST,
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        media_type="application/problem+json",
    )


@app.exception_handler(sqlite3.OperationalError)
async def sqlite_operational_error_handler(request: Request, exc: sqlite3.OperationalError):
    """Handle sqlite3.OperationalError, particularly database is locked, returning RFC 7807 response."""
    err_msg = str(exc)
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if "locked" in err_msg.lower() or "busy" in err_msg.lower()
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    title = (
        "Service Unavailable"
        if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        else "Database Error"
    )
    message = (
        "Service busy, please retry"
        if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        else f"Database error: {err_msg}"
    )

    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"
    logger.error(f"SQLite operational error: {exc}", exc_info=not is_production)

    path = getattr(getattr(request, "url", None), "path", None)
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": message,
            "instance": path,
            "error": True,
            "code": status_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler that returns an RFC 7807 problem details JSON payload for any unhandled exception."""
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"

    logging.getLogger(__name__).error(
        f"Unhandled exception: {exc}", exc_info=not is_production
    )

    message = "An internal server error occurred." if is_production else str(exc)
    path = getattr(getattr(request, "url", None), "path", None)

    request_id = request.headers.get("x-request-id")
    trace_id = None
    try:
        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            trace_id = trace.format_trace_id(current_span.get_span_context().trace_id)
    except Exception:
        pass

    content = {
        "type": "about:blank",
        "title": "Internal Server Error",
        "status": status_code,
        "detail": message,
        "instance": path,
        "error": True,
        "code": status_code,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if trace_id:
        content["trace_id"] = trace_id
    if request_id:
        content["request_id"] = request_id

    return JSONResponse(
        status_code=status_code,
        content=content,
        media_type="application/problem+json",
    )


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP errors to return RFC 7807 problem details payloads."""
    from http import HTTPStatus

    status_code = exc.status_code
    if status_code == 404:
        title = "Not Found"
        message = "API endpoint or resource not found"
    else:
        try:
            title = HTTPStatus(status_code).phrase
        except ValueError:
            title = "HTTP Error"
        message = exc.detail if isinstance(exc.detail, (str, dict)) else str(exc.detail)

    log_level = logging.WARNING if 400 <= status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        "HTTP %d error on %s %s: %s",
        status_code,
        request.method,
        request.url.path,
        message,
    )

    path = getattr(getattr(request, "url", None), "path", None)
    detail_str = message if isinstance(message, str) else str(message)

    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail_str,
            "instance": path,
            "error": True,
            "code": status_code,
            "message": message,
        },
        media_type="application/problem+json",
    )


app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── Register Sub-Routers ──────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(corpus_router)
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
    """Retrieve paginated security audit events.
    
    Supports pagination via limit and offset parameters (Issue #2732).
    """
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

"""src/api/app.py - FastAPI REST API for LMS integration."""

import logging
import os

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import (
    custom_rate_limit_exceeded_handler,
    limiter,
)
from src.api.middleware import verify_bearer_token
from src.api.routers import (
    admin_router,
    analysis_router,
    auth_router,
    corpus_router,
)

# Re-exports for backward compatibility with existing tests and scripts

logger = logging.getLogger(__name__)

# ── API Initialization ────────────────────────────────────────────────────────

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
    """Retrieve paginated security audit events.
    
    Supports pagination via limit and offset parameters (Issue #2732).
    """
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
app.include_router(admin_router)

 
 
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/api/v1/scan/progress/{job_id}")
async def scan_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        from src.celery_app.celery_config import celery_app
        import asyncio
        while True:
            task = celery_app.AsyncResult(job_id)
            if task.state == 'PENDING':
                response = {"state": task.state, "status": "Pending..."}
            elif task.state != 'FAILURE':
                info = task.info if isinstance(task.info, dict) else {}
                response = {
                    "state": task.state, 
                    "status": info.get('step', ''), 
                    "progress": info.get('progress', 0),
                    "total": info.get('total', 0)
                }
                if task.state == 'SUCCESS':
                    response['result'] = info
            else:
                response = {"state": task.state, "status": str(task.info)}
            
            await websocket.send_json(response)
            if task.state in ['SUCCESS', 'FAILURE']:
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
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
    corpus_router,
)
from src.version import APP_VERSION

# Re-exports for backward compatibility with existing tests and scripts

logger = logging.getLogger(__name__)

# ── API Initialization ────────────────────────────────────────────────────────

from src.core.app_config import print_startup_config_summary


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
    ],
    dependencies=[Depends(verify_bearer_token)],
)


@app.on_event("startup")
def startup_event() -> None:
    """Run the one-off work the API needs before it serves its first request.

    Two things happen here, and they are deliberately independent: warming the
    embedding model is best-effort, printing the configuration summary is not.
    A warm-up that cannot run — no cached weights, no network, a machine with
    no room for the model — must not stop the process from coming up, because
    most of the API surface (auth, corpus listing, admin) does not need
    embeddings at all.
    """
    _warmup_embedding_model()
    print_startup_config_summary()


def _warmup_embedding_model() -> bool:
    """Pre-load the embedding weights so the first real request is not slow.

    The import is deliberately deferred to call time rather than module scope:
    pulling in ``src.core.embedding_model`` drags in the whole ML stack, and
    ``src/api/app.py`` is imported by tooling that has no business paying for
    that (test collection, ``--help`` on the CLI, OpenAPI schema dumps).

    Returns:
        ``True`` if the warm-up pass completed, ``False`` if it was skipped or
        failed. Never raises — a failed warm-up costs latency on the first
        request, not availability.
    """
    try:
        from src.core.embedding_model import warmup_embedding_model
    except Exception:
        # The ML extras are optional; an API deployment that only serves the
        # non-embedding routes is a supported configuration.
        logger.warning(
            "Embedding model unavailable; skipping startup warmup. "
            "The first scan request will pay the model load cost.",
            exc_info=True,
        )
        return False

    try:
        return bool(warmup_embedding_model())
    except Exception:
        logger.warning(
            "Embedding model warmup failed; continuing startup.", exc_info=True
        )
        return False


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
    # 1. Check if user_id was pre-set on request.state
    user_id = getattr(request.state, "user_id", None)

    # 2. If missing, attempt to extract token from Authorization header
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

            # Use request.scope.get("route").path to extract generalized FastAPI route template for span name
            route = request.scope.get("route")
            if route and hasattr(route, "path"):
                route_path = route.path
                span.update_name(f"HTTP {request.method} {route_path}")
                span.set_attribute("http.route", route_path)

            span.set_attribute("http.status_code", response.status_code)
            # Update user.id if set or modified by route handler/dependencies
            final_user_id = getattr(request.state, "user_id", user_id)
            if final_user_id:
                span.set_attribute("user.id", str(final_user_id))
            return response
        except Exception as exc:
            route = request.scope.get("route")
            if route and hasattr(route, "path"):
                route_path = route.path
                span.update_name(f"HTTP {request.method} {route_path}")
                span.set_attribute("http.route", route_path)
            span.record_exception(exc)
            span.set_attribute("http.status_code", 500)
            raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors from malformed API requests adhering to RFC 7807."""
    # Issue #2564: Log the detailed validation errors for backend debugging
    logger.warning(
        "Request validation failed for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    errors_list = [
        {
            "field": ".".join(map(str, err["loc"])),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "about:blank",
            "title": "Unprocessable Entity",
            "status": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "detail": "Validation failed.",
            "instance": getattr(getattr(request, "url", None), "path", None),
            "error": True,
            "message": "Validation failed.",
            "details": errors_list,
            "invalid_params": errors_list,
        },
        media_type="application/problem+json",
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP 404 errors adhering to RFC 7807."""
    path = getattr(getattr(request, "url", None), "path", None)
    return JSONResponse(
        status_code=404,
        content={
            "type": "about:blank",
            "title": "Not Found",
            "status": 404,
            "detail": "API endpoint or resource not found",
            "instance": path,
            "error": True,
            "code": 404,
            "message": "API endpoint or resource not found",
        },
        media_type="application/problem+json",
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions by returning an RFC 7807 400 Bad Request response."""
    path = getattr(getattr(request, "url", None), "path", None)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "type": "about:blank",
            "title": "Bad Request",
            "status": status.HTTP_400_BAD_REQUEST,
            "detail": str(exc),
            "instance": path,
            "error": True,
            "code": status.HTTP_400_BAD_REQUEST,
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        media_type="application/problem+json",
    )


@app.exception_handler(sqlite3.OperationalError)
async def sqlite_operational_error_handler(
    request: Request, exc: sqlite3.OperationalError
):
    """Handle sqlite3.OperationalError, particularly database is locked, returning RFC 7807 response."""
    err_msg = str(exc)
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if "locked" in err_msg.lower() or "busy" in err_msg.lower()
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    title = (
        "Service Unavailable"
        if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        else "Database Error"
    )
    message = (
        "Service busy, please retry"
        if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        else f"Database error: {err_msg}"
    )

    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"
    logger.error(f"SQLite operational error: {exc}", exc_info=not is_production)

    path = getattr(getattr(request, "url", None), "path", None)
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": message,
            "instance": path,
            "error": True,
            "code": status_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler that returns an RFC 7807 problem details JSON payload for any unhandled exception."""
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"

    logging.getLogger(__name__).error(
        f"Unhandled exception: {exc}", exc_info=not is_production
    )

    message = "An internal server error occurred." if is_production else str(exc)
    path = getattr(getattr(request, "url", None), "path", None)

    request_id = request.headers.get("x-request-id")
    trace_id = None
    try:
        from opentelemetry import trace

        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            trace_id = trace.format_trace_id(current_span.get_span_context().trace_id)
    except Exception:
        pass

    content = {
        "type": "about:blank",
        "title": "Internal Server Error",
        "status": status_code,
        "detail": message,
        "instance": path,
        "error": True,
        "code": status_code,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if trace_id:
        content["trace_id"] = trace_id
    if request_id:
        content["request_id"] = request_id

    return JSONResponse(
        status_code=status_code,
        content=content,
        media_type="application/problem+json",
    )


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP errors to return RFC 7807 problem details payloads."""
    from http import HTTPStatus

    status_code = exc.status_code
    if status_code == 404:
        title = "Not Found"
        message = "API endpoint or resource not found"
    else:
        try:
            title = HTTPStatus(status_code).phrase
        except ValueError:
            title = "HTTP Error"
        message = exc.detail if isinstance(exc.detail, (str, dict)) else str(exc.detail)

    log_level = logging.WARNING if 400 <= status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        "HTTP %d error on %s %s: %s",
        status_code,
        request.method,
        request.url.path,
        message,
    )

    path = getattr(getattr(request, "url", None), "path", None)
    detail_str = message if isinstance(message, str) else str(message)

    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail_str,
            "instance": path,
            "error": True,
            "code": status_code,
            "message": message,
        },
        media_type="application/problem+json",
    )


app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── Register Sub-Routers ──────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(corpus_router)
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
    offset: int = Query(
        default=0, ge=0, description="Number of events to skip (pagination)"
    ),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    username: str | None = Query(default=None, description="Filter by username"),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Retrieve paginated security audit events.

    Supports pagination via limit and offset parameters (Issue #2732).
    """
    from src.db.auth import get_security_audit_log_count, get_security_audit_logs

    events = get_security_audit_logs(
        limit=limit, offset=offset, event_type=event_type, username=username
    )

    total_count = get_security_audit_log_count(event_type=event_type, username=username)

    return {
        "events": events,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total_count": total_count,
            "total_pages": (total_count + limit - 1) // limit if limit > 0 else 0,
        },
    }


# Re-exports for backward compatibility with existing tests and scripts
import sys
import types


class _ApiAppModule(types.ModuleType):
    @property
    def total_scans(self) -> int:
        from src.api.routers import analysis

        return analysis.total_scans

    @total_scans.setter
    def total_scans(self, value: int) -> None:
        from src.api.routers import analysis

        analysis.total_scans = value


sys.modules[__name__].__class__ = _ApiAppModule


# Bind property to FastAPI class to support tests referencing the app instance as api_app
def _get_total_scans(self) -> int:
    from src.api.routers import analysis

    return analysis.total_scans


def _set_total_scans(self, value: int) -> None:
    from src.api.routers import analysis

    analysis.total_scans = value


FastAPI.total_scans = property(_get_total_scans, _set_total_scans)

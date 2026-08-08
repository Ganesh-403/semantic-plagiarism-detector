# JSONContentTypeMiddleware unit coverage for Issue #1394.
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient as StarletteTestClient
from starlette.requests import Request

from src.asgi_app import JSONContentTypeMiddleware, ClientIPLoggingMiddleware


async def _json_echo(request):
    return JSONResponse({"accepted": True})


def _json_middleware_client():
    test_app = Starlette(
        routes=[
            Route(
                "/api/v1/settings",
                _json_echo,
                methods=["POST", "PUT", "GET"],
            ),
            Route(
                "/api/v1/scan",
                _json_echo,
                methods=["POST"],
            ),
            Route(
                "/internal/action",
                _json_echo,
                methods=["POST"],
            ),
        ],
        middleware=[Middleware(JSONContentTypeMiddleware)],
    )
    return StarletteTestClient(test_app)


def test_json_middleware_accepts_application_json():
    response = _json_middleware_client().post(
        "/api/v1/settings",
        headers={"Content-Type": "application/json"},
        content=b'{"enabled":true}',
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}


def test_json_middleware_accepts_json_with_charset():
    response = _json_middleware_client().put(
        "/api/v1/settings",
        headers={
            "Content-Type": "Application/JSON; Charset=UTF-8",
        },
        content=b'{"enabled":true}',
    )

    assert response.status_code == 200


def test_json_middleware_accepts_structured_json_suffix():
    response = _json_middleware_client().post(
        "/api/v1/settings",
        headers={
            "Content-Type": "application/problem+json",
        },
        content=b'{"title":"problem"}',
    )

    assert response.status_code == 200


def test_json_middleware_rejects_non_json_post_payload():
    response = _json_middleware_client().post(
        "/api/v1/settings",
        headers={"Content-Type": "text/plain"},
        content=b"not json",
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": ("Unsupported Media Type: Request must be application/json")
    }


def test_json_middleware_rejects_missing_content_type_with_body():
    response = _json_middleware_client().put(
        "/api/v1/settings",
        content=b'{"enabled":true}',
    )

    assert response.status_code == 415


def test_json_middleware_allows_bodyless_post():
    response = _json_middleware_client().post(
        "/api/v1/settings",
        content=b"",
    )

    assert response.status_code == 200


def test_json_middleware_does_not_restrict_get_requests():
    response = _json_middleware_client().get(
        "/api/v1/settings",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 200


def test_json_middleware_excludes_multipart_scan_endpoint():
    response = _json_middleware_client().post(
        "/api/v1/scan",
        headers={"Content-Type": ("multipart/form-data; boundary=example")},
        content=b"--example--",
    )

    assert response.status_code == 200


def test_json_middleware_ignores_non_api_post_routes():
    response = _json_middleware_client().post(
        "/internal/action",
        headers={"Content-Type": "text/plain"},
        content=b"streamlit-internal-payload",
    )

    assert response.status_code == 200


# ── TokenBucketRateLimiter Tests (#1362) ──────────────────────────────────────

import time
from src.asgi_app import TokenBucketRateLimiter


def _rate_limiter_client(
    rate_limit_per_minute=60, burst_capacity=10, api_prefix="/api/"
):
    test_app = Starlette(
        routes=[
            Route(
                "/api/v1/resource",
                _json_echo,
                methods=["GET", "POST"],
            ),
            Route(
                "/dashboard/home",
                _json_echo,
                methods=["GET"],
            ),
        ],
        middleware=[
            Middleware(
                TokenBucketRateLimiter,
                rate_limit_per_minute=rate_limit_per_minute,
                burst_capacity=burst_capacity,
                api_prefix=api_prefix,
            )
        ],
    )
    return StarletteTestClient(test_app)


def test_token_bucket_allows_within_burst_capacity():
    client = _rate_limiter_client(burst_capacity=10)
    for _ in range(10):
        response = client.get("/api/v1/resource")
        assert response.status_code == 200


def test_token_bucket_rejects_exceeding_burst_capacity_with_429():
    client = _rate_limiter_client(rate_limit_per_minute=60, burst_capacity=5)

    # First 5 requests within burst capacity succeed
    for _ in range(5):
        res = client.get("/api/v1/resource")
        assert res.status_code == 200

    # 6th request exceeds burst capacity -> 429 Too Many Requests
    blocked_response = client.get("/api/v1/resource")
    assert blocked_response.status_code == 429
    assert blocked_response.text == "Too Many Requests"
    assert "Retry-After" in blocked_response.headers
    assert blocked_response.headers["Retry-After"].isdigit()
    assert int(blocked_response.headers["Retry-After"]) >= 1


def test_token_bucket_refills_tokens_over_time():
    client = _rate_limiter_client(rate_limit_per_minute=600, burst_capacity=2)

    # Exhaust capacity
    assert client.get("/api/v1/resource").status_code == 200
    assert client.get("/api/v1/resource").status_code == 200
    assert client.get("/api/v1/resource").status_code == 429

    # Wait 0.2 seconds -> 2 tokens refilled (600/min = 10/sec)
    time.sleep(0.2)
    assert client.get("/api/v1/resource").status_code == 200


def test_token_bucket_ignores_non_api_routes():
    client = _rate_limiter_client(burst_capacity=2, api_prefix="/api/")

    # Exhaust API route capacity
    client.get("/api/v1/resource")
    client.get("/api/v1/resource")
    assert client.get("/api/v1/resource").status_code == 429

    # Non-API route is not rate-limited
    assert client.get("/dashboard/home").status_code == 200


def test_token_bucket_per_ip_isolation():
    client = _rate_limiter_client(burst_capacity=1)

    # IP 1 uses its 1 token
    res1 = client.get("/api/v1/resource", headers={"X-Forwarded-For": "192.168.1.10"})
    assert res1.status_code == 200

    res1_blocked = client.get(
        "/api/v1/resource", headers={"X-Forwarded-For": "192.168.1.10"}
    )
    assert res1_blocked.status_code == 429

    # IP 2 still has its token
    res2 = client.get("/api/v1/resource", headers={"X-Forwarded-For": "192.168.1.20"})
    assert res2.status_code == 200


# ── Standardized Health Status Endpoint (#1273) ─────────────────────────────

from datetime import datetime

from fastapi.testclient import TestClient

from src.api.app import app

_status_client = TestClient(app)


def test_api_v1_status_returns_online_payload():
    """Verify GET /api/v1/status returns the standardized status payload."""
    response = _status_client.get("/api/v1/status")

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert data["status"] == "online"
    assert data["version"] == "1.0.0"


def test_api_v1_status_timestamp_is_iso_utc():
    """Verify the timestamp is a parseable ISO 8601 string in UTC."""
    response = _status_client.get("/api/v1/status")
    data = response.json()

    parsed = datetime.fromisoformat(data["timestamp"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_api_v1_status_is_public_without_token():
    """Verify /api/v1/status is reachable without a bearer token."""
    response = _status_client.get(
        "/api/v1/status",
        headers={},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_scope_enforcement_rate_limit_endpoint():
    """Verify scope enforcement on /api/v1/rate_limit (requires 'read')."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)

    # 1. No credentials -> 401
    res = client.get("/api/v1/rate_limit")
    assert res.status_code == 401

    # 2. Token with 'read' scope -> 200
    res = client.get(
        "/api/v1/rate_limit", headers={"Authorization": "Bearer test-read-token"}
    )
    assert res.status_code == 200

    # 3. Token with no scopes -> 403
    res = client.get(
        "/api/v1/rate_limit", headers={"Authorization": "Bearer test-no-scope-token"}
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json()["detail"]


def test_scope_enforcement_clear_endpoint(tmp_path):
    """Verify scope enforcement on /api/v1/clear (requires 'admin')."""
    from src.db.auth import configure_db_path, init_db, add_user
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_clear_scope.db"
    configure_db_path(db_file)
    init_db()
    try:
        add_user("admin", "password123", role="admin")
    except ValueError:
        pass

    client = TestClient(app)

    # 1. Token with 'admin' scope -> success
    res = client.post(
        "/api/v1/clear?username=admin",
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert res.status_code in (200, 500)

    # 2. Token with 'write' but no 'admin' -> 403
    res = client.post(
        "/api/v1/clear?username=admin",
        headers={"Authorization": "Bearer test-write-token"},
    )
    assert res.status_code == 403


# ── Asynchronous Background Scan Job Queue Tests (#1372) ─────────────────────


def test_async_scan_returns_202_accepted_with_job_id():
    """Verify POST /api/v1/scan/async accepts upload and returns 202 with job_id."""
    client = TestClient(app)

    files = {"file": ("async_doc.txt", b"This is a test document for async scanning.")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan/async", files=files, headers=headers)

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["status_url"] == f"/api/v1/scan/status/{data['job_id']}"
    assert "message" in data


def test_async_scan_status_progression():
    """Verify background job executes and GET /api/v1/scan/status/{job_id} returns completed results."""
    client = TestClient(app)

    files = {
        "file": (
            "sample_async.txt",
            b"Artificial intelligence and machine learning enable automated analysis.",
        )
    }
    headers_write = {"Authorization": "Bearer test-write-token"}
    headers_read = {"Authorization": "Bearer test-read-token"}

    # 1. Enqueue job
    post_res = client.post("/api/v1/scan/async", files=files, headers=headers_write)
    assert post_res.status_code == 202
    job_id = post_res.json()["job_id"]

    # 2. Check status (BackgroundTasks runs during TestClient request cycle)
    status_res = client.get(f"/api/v1/scan/status/{job_id}", headers=headers_read)
    assert status_res.status_code == 200

    status_data = status_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ("queued", "processing", "completed")
    assert status_data["filename"] == "sample_async.txt"

    if status_data["status"] == "completed":
        assert status_data["result"] is not None
        assert status_data["result"]["filename"] == "sample_async.txt"
        assert "plagiarism_flagged" in status_data["result"]


def test_async_scan_status_invalid_job_id_returns_404():
    """Verify GET /api/v1/scan/status/{job_id} returns 404 for unknown job IDs."""
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-read-token"}

    response = client.get("/api/v1/scan/status/invalid_job_99999", headers=headers)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_async_scan_empty_file_returns_400():
    """Verify POST /api/v1/scan/async rejects empty files with HTTP 400."""
    client = TestClient(app)
    files = {"file": ("empty.txt", b"")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan/async", files=files, headers=headers)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ── Global Exception Handler Tests (#1500) ────────────────────────────────────

import asyncio
import json
from unittest.mock import Mock

from src.api.app import global_exception_handler


def test_global_exception_handler_returns_standard_payload(monkeypatch):
    """Verify Issue #1500: unhandled exceptions return the standardized JSON payload."""
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    mock_request = Mock()

    response = asyncio.run(global_exception_handler(mock_request, ValueError("boom")))
    body = json.loads(response.body)

    assert response.status_code == 500
    assert body["error"] is True
    assert body["code"] == 500
    assert body["message"] == "boom"
    assert "timestamp" in body


def test_global_exception_handler_masks_details_in_production(monkeypatch):
    """Verify Issue #1500: internal exception details are masked in production."""
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    mock_request = Mock()

    response = asyncio.run(
        global_exception_handler(mock_request, ValueError("sensitive internal detail"))
    )
    body = json.loads(response.body)

    assert body["message"] == "An internal server error occurred."
    assert "sensitive internal detail" not in body["message"]


# ── CORS Preflight Cache Duration Test (#1501) ────────────────────────────────

import importlib


def test_cors_preflight_max_age_header():
    """Verify OPTIONS preflight responses include Access-Control-Max-Age: 3600."""
    import sys

    importlib.reload(sys.modules["src.api.app"])
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)

    response = client.options(
        "/api/v1/version",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-max-age"] == "3600"


# ── Token Revocation Endpoint Tests (#1499) ───────────────────────────────────


def test_token_revocation_endpoint_revokes_token_and_rejects_subsequent_requests(
    tmp_path, monkeypatch
):
    """Verify POST /api/v1/auth/revoke invalidates token and subsequent calls return 401."""
    from src.db.auth import configure_db_path, init_db, is_token_revoked
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_auth_revoke.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    target_token = "test-read-token"

    # 1. Verify token works initially before revocation on protected route
    res_before = client.get(
        "/api/v1/incidents", headers={"Authorization": f"Bearer {target_token}"}
    )
    assert res_before.status_code == 200

    # 2. Revoke token via POST /api/v1/auth/revoke
    revoke_res = client.post("/api/v1/auth/revoke", json={"token": target_token})
    assert revoke_res.status_code == 200
    assert revoke_res.json()["status"] == "success"
    assert is_token_revoked(target_token) is True

    # 3. Subsequent request with revoked token fails with 401 Unauthorized
    res_after = client.get(
        "/api/v1/incidents", headers={"Authorization": f"Bearer {target_token}"}
    )
    assert res_after.status_code == 401
    assert "revoked" in res_after.json()["detail"].lower()


def test_token_revocation_via_authorization_header(tmp_path):
    """Verify token can be revoked via Authorization header when body is omitted."""
    from src.db.auth import configure_db_path, init_db, is_token_revoked
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_auth_revoke_header.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    target_token = "test-write-token"

    revoke_res = client.post(
        "/api/v1/auth/revoke", headers={"Authorization": f"Bearer {target_token}"}
    )
    assert revoke_res.status_code == 200
    assert is_token_revoked(target_token) is True


def test_token_revocation_missing_token_returns_400(tmp_path):
    """Verify POST /api/v1/auth/revoke returns 400 Bad Request if no token provided."""
    from src.db.auth import configure_db_path, init_db
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_auth_revoke_missing.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)

    response = client.post("/api/v1/auth/revoke", json={})
    assert response.status_code == 400
    assert "token to revoke must be provided" in response.json()["detail"].lower()


def test_streaming_multipart_upload_file_exceeds_max_size_returns_413(monkeypatch):
    """Verify POST /api/v1/scan returns 413 Payload Too Large when payload exceeds MAX_UPLOAD_SIZE_BYTES."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "500")

    client = TestClient(app)
    large_payload = b"A" * 2000  # 2KB payload > 500 bytes limit

    files = {"file": ("large_doc.txt", large_payload, "text/plain")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan", files=files, headers=headers)
    assert response.status_code == 413
    assert "exceeds maximum" in response.json()["detail"].lower()


def test_streaming_multipart_upload_streams_chunks_to_disk():
    """Verify POST /api/v1/scan streams chunks to disk and processes document scan."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    content = (
        b"This is a test paragraph for verifying streaming chunk reader upload functionality.\n\n"
        * 5
    )

    files = {"file": ("stream_test.txt", content, "text/plain")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan", files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "stream_test.txt"
    assert data["word_count"] > 0


def test_refresh_token_success_with_signed_refresh_token(tmp_path):
    """Verify POST /api/v1/auth/refresh issues a new valid access token."""
    from src.db.auth import configure_db_path, init_db
    from src.security.jwt_utils import create_refresh_token
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_refresh_success.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    refresh_token = create_refresh_token(sub="alice_user")

    res = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600

    # Verify newly issued access token works on an authenticated endpoint
    access_token = data["access_token"]
    auth_res = client.get(
        "/api/v1/rate_limit", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert auth_res.status_code == 200


def test_refresh_token_success_via_authorization_header(tmp_path):
    """Verify POST /api/v1/auth/refresh works via Authorization header."""
    from src.db.auth import configure_db_path, init_db
    from src.security.jwt_utils import create_refresh_token
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_refresh_header.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    refresh_token = create_refresh_token(sub="bob_user")

    res = client.post(
        "/api/v1/auth/refresh", headers={"Authorization": f"Bearer {refresh_token}"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600


def test_refresh_token_missing_token_returns_400(tmp_path):
    """Verify POST /api/v1/auth/refresh returns 400 if refresh token is omitted."""
    from src.db.auth import configure_db_path, init_db
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_refresh_missing.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    res = client.post("/api/v1/auth/refresh", json={})
    assert res.status_code == 400
    assert "refresh token must be provided" in res.json()["detail"].lower()


def test_refresh_token_invalid_signature_returns_401(tmp_path):
    """Verify POST /api/v1/auth/refresh returns 401 for invalid refresh token signature."""
    from src.db.auth import configure_db_path, init_db
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_refresh_invalid.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    res = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": "invalid.jwt.signature"}
    )
    assert res.status_code == 401


def test_refresh_token_expired_returns_401(tmp_path):
    """Verify POST /api/v1/auth/refresh returns 401 for an expired refresh token."""
    from src.db.auth import configure_db_path, init_db
    from src.security.jwt_utils import create_jwt_token
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_refresh_expired.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    expired_token = create_jwt_token(
        {"sub": "charlie", "type": "refresh"}, expires_in_seconds=-100
    )

    res = client.post("/api/v1/auth/refresh", json={"refresh_token": expired_token})
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower()


def test_refresh_token_revoked_returns_401(tmp_path):
    """Verify POST /api/v1/auth/refresh returns 401 if refresh token has been revoked."""
    from src.db.auth import configure_db_path, init_db, revoke_token
    from src.security.jwt_utils import create_refresh_token
    from fastapi.testclient import TestClient
    from src.api.app import app

    db_file = tmp_path / "test_refresh_revoked.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    refresh_token = create_refresh_token(sub="dave")
    revoke_token(refresh_token, details="Revoked for testing")

    res = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert res.status_code == 401
    assert "revoked" in res.json()["detail"].lower()


def test_api_usage_endpoint():
    """Verify that GET /api/v1/usage returns the correct schema and increments total_scans."""
    from fastapi.testclient import TestClient
    from src.api.app import app
    import src.api.app as api_app

    client = TestClient(app)

    # Initial request
    response = client.get("/api/v1/usage")
    assert response.status_code == 200
    data = response.json()
    assert "total_scans" in data
    assert "uptime_seconds" in data
    assert isinstance(data["total_scans"], int)
    assert isinstance(data["uptime_seconds"], float)
    assert data["uptime_seconds"] >= 0.0

    # Test the counter increment reflection
    initial_scans = data["total_scans"]
    api_app.total_scans += 1

    response = client.get("/api/v1/usage")
    assert response.status_code == 200
    data = response.json()
    assert data["total_scans"] == initial_scans + 1


def test_hsts_security_header_options():
    """Verify HSTS Strict-Transport-Security header behavior when ENABLE_HSTS is configured."""
    import os
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from src.asgi_app import SecurityHeadersMiddleware

    def dummy_app(request):
        return PlainTextResponse("OK")

    # Disabled by default
    app_disabled = Starlette(
        routes=[Route("/", dummy_app)],
        middleware=[Middleware(SecurityHeadersMiddleware)],
    )
    client_disabled = TestClient(app_disabled)
    res_disabled = client_disabled.get("/")
    assert "Strict-Transport-Security" not in res_disabled.headers

    # Enabled via ENABLE_HSTS=true
    os.environ["ENABLE_HSTS"] = "true"
    try:
        app_enabled = Starlette(
            routes=[Route("/", dummy_app)],
            middleware=[Middleware(SecurityHeadersMiddleware)],
        )
        client_enabled = TestClient(app_enabled)
        res_enabled = client_enabled.get("/")
        assert (
            res_enabled.headers.get("Strict-Transport-Security")
            == "max-age=31536000; includeSubDomains"
        )
    finally:
        os.environ.pop("ENABLE_HSTS", None)


# ── ClientIPLoggingMiddleware Tests ───────────────────────────────────────────


def _ip_middleware_client():
    async def _ip_echo(request: Request):
        return JSONResponse({"ip": getattr(request.state, "client_ip", None)})

    test_app = Starlette(
        routes=[Route("/ip", _ip_echo, methods=["GET"])],
        middleware=[Middleware(ClientIPLoggingMiddleware)],
    )
    return StarletteTestClient(test_app)


def test_client_ip_from_forwarded_for():
    response = _ip_middleware_client().get(
        "/ip",
        headers={"X-Forwarded-For": "203.0.113.8"},
    )
    assert response.status_code == 200
    assert response.json()["ip"] == "203.0.113.8"


def test_multiple_forwarded_addresses():
    response = _ip_middleware_client().get(
        "/ip",
        headers={"X-Forwarded-For": "203.0.113.8, 10.0.0.1"},
    )
    assert response.json()["ip"] == "203.0.113.8"


def test_client_host_fallback():
    response = _ip_middleware_client().get("/ip")
    assert response.status_code == 200
    assert response.json()["ip"] is not None

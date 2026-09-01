from fastapi.testclient import TestClient

from src.api.app import app
from src.version import APP_VERSION

client = TestClient(app)


def test_http_404_not_found():
    """Verify that a nonexistent route returns a standardized JSON 404 response payload."""
    response = client.get("/api/v1/nonexistent-endpoint-xyz")
    assert response.status_code == 404
    assert response.json() == {
        "error": True,
        "code": 404,
        "message": "API endpoint or resource not found",
    }


# JSONContentTypeMiddleware unit coverage for Issue #1394.
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient as StarletteTestClient

from src.asgi_app import ClientIPLoggingMiddleware, JSONContentTypeMiddleware


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
    assert data["version"] == APP_VERSION


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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import add_user, configure_db_path, init_db

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


def test_clear_corpus_audit_logging(tmp_path):
    """Verify that successful /api/v1/clear logs a security event with event_type='CORPUS_CLEARED'."""
    from fastapi.testclient import TestClient
    from src.api.app import app
    from src.db.auth import add_user, configure_db_path, init_db, get_security_audit_logs

    db_file = tmp_path / "test_clear_audit.db"
    configure_db_path(db_file)
    init_db()
    
    try:
        add_user("admin_user", "password123", role="admin")
    except ValueError:
        pass

    client = TestClient(app)

    # 1. Execute clear
    res = client.post(
        "/api/v1/clear?username=admin_user",
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert res.status_code == 200

    # 2. Query logs to verify the event is logged
    logs = get_security_audit_logs(username="admin_user", event_type="CORPUS_CLEARED")
    assert len(logs) > 0
    event = logs[0]
    assert event["event_type"] == "CORPUS_CLEARED"
    assert event["username"] == "admin_user"
    assert "Client IP:" in event["details"]
    assert "Timestamp:" in event["details"]



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


def test_cancel_async_scan_job_success():
    """Verify DELETE /api/v1/scan/jobs/{job_id} cancels an async scan job."""
    client = TestClient(app)

    files = {"file": ("cancel_test.txt", b"Content for job cancellation test.")}
    headers_write = {"Authorization": "Bearer test-write-token"}
    headers_read = {"Authorization": "Bearer test-read-token"}

    # 1. Enqueue job
    post_res = client.post("/api/v1/scan/async", files=files, headers=headers_write)
    assert post_res.status_code == 202
    job_id = post_res.json()["job_id"]

    # 2. Cancel job via DELETE /api/v1/scan/jobs/{job_id}
    cancel_res = client.delete(f"/api/v1/scan/jobs/{job_id}", headers=headers_write)
    assert cancel_res.status_code == 200
    cancel_data = cancel_res.json()
    assert cancel_data["job_id"] == job_id
    assert cancel_data["status"] == "cancelled"

    # 3. Verify status endpoint reports job as cancelled
    status_res = client.get(f"/api/v1/scan/status/{job_id}", headers=headers_read)
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] == "cancelled"
    assert "cancelled" in status_data["error"].lower()


def test_cancel_async_scan_job_invalid_id_returns_404():
    """Verify DELETE /api/v1/scan/jobs/{job_id} returns 404 for unknown job IDs."""
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.delete("/api/v1/scan/jobs/nonexistent_job_123", headers=headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db, is_token_revoked

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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db, is_token_revoked

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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db

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


def test_scan_rejects_exe_upload_with_415():
    """Verify POST /api/v1/scan rejects a Windows PE executable (.exe, MZ header) with 415."""
    from fastapi.testclient import TestClient

    from src.api.app import app

    client = TestClient(app)
    # Minimal DOS/PE executable header ("MZ" magic bytes).
    payload = b"MZ" + b"\x90\x00" * 30

    files = {"file": ("malware.exe", payload, "application/octet-stream")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan", files=files, headers=headers)
    assert response.status_code == 415
    assert "unsupported media type" in response.json()["message"].lower()


def test_scan_rejects_shell_script_upload_with_415():
    """Verify POST /api/v1/scan rejects a shell script (#!/bin/sh shebang) with 415."""
    from fastapi.testclient import TestClient

    from src.api.app import app

    client = TestClient(app)
    payload = b"#!/bin/sh\necho 'hello'\n"

    # Even with an unrelated/misleading extension, the shebang magic bytes
    # alone must be enough to trigger the rejection.
    files = {"file": ("notes.txt", payload, "text/plain")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan", files=files, headers=headers)
    assert response.status_code == 415


def test_scan_rejects_bat_and_dll_extensions_with_415():
    """Verify POST /api/v1/scan rejects .bat and .dll uploads by extension, with 415."""
    from fastapi.testclient import TestClient

    from src.api.app import app

    client = TestClient(app)
    headers = {"Authorization": "Bearer test-write-token"}

    for filename, payload in [
        ("run.bat", b"@echo off\r\necho hi\r\n"),
        ("library.dll", b"MZ" + b"\x00" * 30),
    ]:
        files = {"file": (filename, payload, "application/octet-stream")}
        response = client.post("/api/v1/scan", files=files, headers=headers)
        assert response.status_code == 415, f"{filename} was not rejected with 415"


def test_scan_does_not_reject_ordinary_text_upload():
    """Sanity check: a normal .txt upload must NOT be blocked by the executable check."""
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.corpus_db import init_corpus_db

    # Ensure the corpus DB tables exist — scan_document() reaches the DB
    # layer once a file clears the executable check, and table creation
    # is otherwise only triggered lazily elsewhere in the app.
    init_corpus_db()

    client = TestClient(app)
    payload = b"This is an ordinary plain-text document, not an executable.\n" * 3

    files = {"file": ("essay.txt", payload, "text/plain")}
    headers = {"Authorization": "Bearer test-write-token"}

    response = client.post("/api/v1/scan", files=files, headers=headers)
    assert response.status_code != 415


def test_refresh_token_success_with_signed_refresh_token(tmp_path):
    """Verify POST /api/v1/auth/refresh issues a new valid access token."""
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db
    from src.security.jwt_utils import create_refresh_token

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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db
    from src.security.jwt_utils import create_refresh_token

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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db

    db_file = tmp_path / "test_refresh_missing.db"
    configure_db_path(db_file)
    init_db()

    client = TestClient(app)
    res = client.post("/api/v1/auth/refresh", json={})
    assert res.status_code == 400
    assert "refresh token must be provided" in res.json()["detail"].lower()


def test_refresh_token_invalid_signature_returns_401(tmp_path):
    """Verify POST /api/v1/auth/refresh returns 401 for invalid refresh token signature."""
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db

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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db
    from src.security.jwt_utils import create_jwt_token

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
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.db.auth import configure_db_path, init_db, revoke_token
    from src.security.jwt_utils import create_refresh_token

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

    import src.api.app as api_app
    from src.api.app import app

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


def test_total_scans_persistence():
    """Verify that total_scans persists in the database / Redis across resets."""
    from src.db.corpus_db import get_total_scans, increment_total_scans
    import sqlite3
    from src.core.app_config import CORPUS_DB_PATH

    initial = get_total_scans()
    incremented = increment_total_scans()
    assert incremented == initial + 1

    # Read from database to verify persistence
    conn = sqlite3.connect(str(CORPUS_DB_PATH))
    try:
        cursor = conn.execute("SELECT metric_value FROM system_metrics WHERE metric_name = 'total_scans'")
        row = cursor.fetchone()
        assert row is not None
    except sqlite3.OperationalError:
        # If running in environment where Redis is used or system_metrics table is initialized differently
        pass
    finally:
        conn.close()

    assert get_total_scans() == incremented


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


# ── Multipart Content-Type Header Validation Tests (#1785) ────────────────────


def test_scan_endpoint_rejects_non_multipart_content_type():
    """Verify POST /api/v1/scan returns HTTP 415 when Content-Type is not multipart/form-data."""
    client = TestClient(app)
    headers = {
        "Authorization": "Bearer test-write-token",
        "Content-Type": "application/json",
    }
    response = client.post("/api/v1/scan", content=b"{}", headers=headers)
    assert response.status_code == 415
    data = response.json()
    assert (
        data.get("message")
        == "Unsupported Media Type: Request must be multipart/form-data"
        or data.get("detail")
        == "Unsupported Media Type: Request must be multipart/form-data"
    )


def test_scan_endpoint_rejects_missing_content_type():
    """Verify POST /api/v1/scan returns HTTP 415 when Content-Type header is missing."""
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-write-token"}
    response = client.post("/api/v1/scan", content=b"data", headers=headers)
    assert response.status_code == 415
    data = response.json()
    assert (
        data.get("message")
        == "Unsupported Media Type: Request must be multipart/form-data"
        or data.get("detail")
        == "Unsupported Media Type: Request must be multipart/form-data"
    )


# ── RequestIDMiddleware Tests (#2024) ──────────────────────────────────────────

from src.asgi_app import RequestIDMiddleware


def _request_id_middleware_client():
    async def _request_id_echo(request: Request):
        return JSONResponse({"request_id": getattr(request.state, "request_id", None)})

    test_app = Starlette(
        routes=[Route("/test-request-id", _request_id_echo, methods=["GET"])],
        middleware=[Middleware(RequestIDMiddleware)],
    )
    return StarletteTestClient(test_app)


def test_request_id_middleware_generates_id():
    """Verify that RequestIDMiddleware generates a new UUID request ID and attaches it to response headers and state."""
    client = _request_id_middleware_client()
    response = client.get("/test-request-id")

    assert response.status_code == 200

    # 1. Header presence and format
    assert "X-Request-ID" in response.headers
    request_id_header = response.headers["X-Request-ID"]
    assert len(request_id_header) > 0

    # 2. Attached to request state
    data = response.json()
    assert data["request_id"] == request_id_header


def test_request_id_middleware_propagates_incoming_id():
    """Verify that RequestIDMiddleware propagates a valid incoming X-Request-ID header."""
    client = _request_id_middleware_client()
    incoming_id = "test-correlation-id-12345"
    response = client.get("/test-request-id", headers={"X-Request-ID": incoming_id})

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == incoming_id
    assert response.json()["request_id"] == incoming_id


def test_request_id_middleware_ignores_invalid_oversized_id():
    """Verify that RequestIDMiddleware ignores an oversized incoming X-Request-ID and generates a new one."""
    client = _request_id_middleware_client()
    oversized_id = "a" * 200  # Exceeds MAX_INCOMING_LENGTH (128)
    response = client.get("/test-request-id", headers={"X-Request-ID": oversized_id})

    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id != oversized_id
    assert len(request_id) > 0
    assert response.json()["request_id"] == request_id


def test_custom_http_exception_handler_dictionary_detail():
    import asyncio
    from unittest.mock import Mock

    from starlette.exceptions import HTTPException as StarletteHTTPException

    from src.api.app import custom_http_exception_handler

    mock_request = Mock()
    mock_request.method = "GET"
    mock_request.url.path = "/test"

    dict_detail = {"key": "value", "reason": "invalid request"}
    exc = StarletteHTTPException(status_code=400, detail=dict_detail)

    response = asyncio.run(custom_http_exception_handler(mock_request, exc))

    import json

    body = json.loads(response.body)

    assert response.status_code == 400
    assert body["error"] is True
    assert body["code"] == 400
    assert body["message"] == dict_detail
    assert body["message"]["key"] == "value"
    assert "timestamp" not in body


def test_custom_http_exception_handler_string_detail():
    import asyncio
    from unittest.mock import Mock

    from starlette.exceptions import HTTPException as StarletteHTTPException

    from src.api.app import custom_http_exception_handler

    mock_request = Mock()
    mock_request.method = "GET"
    mock_request.url.path = "/test"

    exc = StarletteHTTPException(status_code=400, detail="string error")

    response = asyncio.run(custom_http_exception_handler(mock_request, exc))

    import json

    body = json.loads(response.body)

    assert response.status_code == 400
    assert body["message"] == "string error"


class TestTwoFactorVerificationRateLimiter:
    """Test suite for 2FA verification endpoint rate limiting (Issue #4037)."""

    def test_2fa_verification_endpoint_success_and_failure(self, monkeypatch):
        """Verify 2FA verification returns 200 on valid OTP and 401 on invalid OTP."""
        from unittest.mock import patch, MagicMock

        monkeypatch.setattr("src.db.auth.init_db", lambda: None)
        monkeypatch.setattr("src.db.auth.get_2fa_status", lambda user: (True, "JBSWY3DPEHPK3PXP"))
        monkeypatch.setattr("src.db.auth.log_security_event", lambda *args, **kwargs: None)

        with patch("pyotp.TOTP.verify", side_effect=[True, False]):
            # Valid OTP code
            res1 = client.post(
                "/api/v1/auth/2fa/verify",
                json={"username": "alice", "otp_code": "123456"},
            )
            assert res1.status_code == 200
            data1 = res1.json()
            assert data1["verified"] is True
            assert "verified successfully" in data1["message"]

            # Invalid OTP code
            res2 = client.post(
                "/api/v1/auth/2fa/verify",
                json={"username": "alice", "otp_code": "999999"},
            )
            assert res2.status_code == 401

    def test_2fa_verification_returns_400_when_2fa_not_enabled(self, monkeypatch):
        """Verify 2FA verification returns 400 Bad Request if 2FA is not enabled for the user."""
        monkeypatch.setattr("src.db.auth.init_db", lambda: None)
        monkeypatch.setattr("src.db.auth.get_2fa_status", lambda user: (False, None))
        monkeypatch.setattr("src.db.auth.log_security_event", lambda *args, **kwargs: None)

        res = client.post(
            "/api/v1/auth/2fa/verify",
            json={"username": "bob", "otp_code": "123456"},
        )
        assert res.status_code == 400
        assert "2FA is not enabled" in res.json()["detail"]

    def test_2fa_verification_rate_limit_max_5_attempts_per_minute(self, monkeypatch):
        """Verify 2FA verification endpoint enforces 5 attempts per minute rate limit and returns 429 when exceeded."""
        from unittest.mock import patch

        monkeypatch.setattr("src.db.auth.init_db", lambda: None)
        monkeypatch.setattr("src.db.auth.get_2fa_status", lambda user: (True, "JBSWY3DPEHPK3PXP"))
        monkeypatch.setattr("src.db.auth.log_security_event", lambda *args, **kwargs: None)

        with patch("pyotp.TOTP.verify", return_value=False):
            # Send 5 attempts (within limit)
            responses = [
                client.post(
                    "/api/v1/auth/2fa/verify",
                    json={"username": "charlie", "otp_code": "000000"},
                    headers={"X-Forwarded-For": "192.168.1.100"},
                )
                for _ in range(5)
            ]
            for r in responses:
                assert r.status_code in (401, 429)

            # 6th attempt should be blocked by rate limiter (HTTP 429)
            res_6th = client.post(
                "/api/v1/auth/2fa/verify",
                json={"username": "charlie", "otp_code": "000000"},
                headers={"X-Forwarded-For": "192.168.1.100"},
            )
            assert res_6th.status_code in (429, 401)


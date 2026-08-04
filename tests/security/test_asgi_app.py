import os

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.asgi_app import (
    ContentLengthLimitMiddleware,
    SecurityHeadersMiddleware,
)


def homepage(request):
    return PlainTextResponse("OK")


@pytest.mark.unit
def test_security_headers_middleware():
    app = Starlette(
        routes=[Route("/", homepage)],
        middleware=[Middleware(SecurityHeadersMiddleware)],
    )
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert (
        response.headers["Content-Security-Policy"]
        == "frame-ancestors 'none'; default-src 'self';"
    )


@pytest.mark.unit
def test_content_length_limit_middleware_under_limit():
    app = Starlette(
        routes=[Route("/", homepage)],
        middleware=[Middleware(ContentLengthLimitMiddleware)],
    )
    client = TestClient(app)
    # Request body size under default limit (50MB)
    response = client.post("/", content=b"a" * 100)
    assert response.status_code == 200
    assert response.text == "OK"


@pytest.mark.unit
def test_content_length_limit_middleware_over_limit():
    app = Starlette(
        routes=[Route("/", homepage)],
        middleware=[Middleware(ContentLengthLimitMiddleware)],
    )
    client = TestClient(app)

    # Exceed limit using custom env var or default
    os.environ["MAX_REQUEST_BYTES"] = "10"
    try:
        response = client.post("/", content=b"a" * 11)
        assert response.status_code == 413
        assert response.text == "Payload Too Large"

        # Check that request exactly at the limit is allowed
        response_at_limit = client.post("/", content=b"a" * 10)
        assert response_at_limit.status_code == 200
    finally:
        del os.environ["MAX_REQUEST_BYTES"]


@pytest.mark.unit
def test_content_length_limit_middleware_invalid_env_fallback():
    app = Starlette(
        routes=[Route("/", homepage)],
        middleware=[Middleware(ContentLengthLimitMiddleware)],
    )
    client = TestClient(app)
    os.environ["MAX_REQUEST_BYTES"] = "not-a-number"
    try:
        # Should fallback to default 50MB (52428800), so 100 bytes is allowed
        response = client.post("/", content=b"a" * 100)
        assert response.status_code == 200
    finally:
        del os.environ["MAX_REQUEST_BYTES"]

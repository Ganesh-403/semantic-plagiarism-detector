import sys
import pytest
from fastapi.testclient import TestClient


def get_fresh_app(monkeypatch, cors_origins):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", cors_origins)
    sys.modules.pop("src.api.app", None)
    sys.modules.pop("src.api", None)
    from src.api.app import app

    return app


def test_cors_headers(monkeypatch):
    """Verify configured origins receive the correct CORS headers."""
    app = get_fresh_app(monkeypatch, "http://localhost:3000,https://example.com")
    client = TestClient(app)

    response = client.options(
        "/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_wildcard_origin_disables_credentials(monkeypatch):
    """Verify wildcard '*' origin sets allow_credentials=False and omits credentials header."""
    app = get_fresh_app(monkeypatch, "*")
    client = TestClient(app)

    response = client.options(
        "/",
        headers={
            "Origin": "http://random-domain.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-credentials" not in response.headers


def test_cors_wildcard_subdomain_success(monkeypatch):
    """Verify secure wildcard subdomain configuration matches valid origins correctly."""
    app = get_fresh_app(monkeypatch, "https://*.university.edu")
    client = TestClient(app)

    response = client.options(
        "/",
        headers={
            "Origin": "https://cs.university.edu",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert (
        response.headers["access-control-allow-origin"] == "https://cs.university.edu"
    )
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_wildcard_subdomain_rejection(monkeypatch):
    """Verify invalid wildcard subdomain formats do not grant access."""
    app = get_fresh_app(monkeypatch, "https://*.university.edu")
    client = TestClient(app)

    response = client.options(
        "/",
        headers={
            "Origin": "https://evil-university.edu.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    # FastAPI/Starlette will not echo back the origin if it fails validation/regex check
    assert "access-control-allow-origin" not in response.headers

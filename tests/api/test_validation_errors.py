import json
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_validation_error_format():
    response = client.post(
        "/api/v1/scan",
        headers={
            "Authorization": "Bearer validation-test-token",
            "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] is True
    assert "message" in data
    assert "details" in data
    assert isinstance(data["details"], list)


def test_value_error_returns_400_bad_request():
    """Verify that a ValueError in request handling returns HTTP 400 Bad Request."""
    from fastapi import APIRouter

    test_router = APIRouter()

    @test_router.get("/api/v1/test-value-error")
    def trigger_value_error(username: str = ""):
        from src.db.auth import _validate_username

        _validate_username(username)
        return {"status": "ok"}

    app.include_router(test_router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/api/v1/test-value-error?username=",
        headers={"Authorization": "Bearer validation-test-token"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] is True
    assert data["code"] == 400
    assert "Username cannot be empty" in data["message"]


@pytest.fixture(autouse=True)
def scoped_token(monkeypatch, mock_db):
    from src.api.middleware import get_valid_tokens
    monkeypatch.setenv("API_BEARER_TOKENS_MAPPING", json.dumps({"validation-test-token": ["read", "write", "scan"]}))
    get_valid_tokens.cache_clear()
    yield
    get_valid_tokens.cache_clear()

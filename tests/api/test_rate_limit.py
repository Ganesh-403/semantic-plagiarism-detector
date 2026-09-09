import json
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_rate_limit_endpoint(monkeypatch):
    monkeypatch.setattr("src.security.rate_limiter.time.time", lambda: 1000.0)
    response = client.get(
        "/api/v1/rate_limit",
        headers={"Authorization": "Bearer validation-test-token"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "limit" in data
    assert "remaining" in data
    assert "reset_in_seconds" in data

    initial_remaining = data["remaining"]
    assert initial_remaining <= data["limit"]

    # Consume one token
    from src.security.rate_limiter import get_token_bucket_limiter
    lim = get_token_bucket_limiter()
    lim.consume("validation-test-token")

    # Fetch rate limit status again
    response = client.get(
        "/api/v1/rate_limit",
        headers={"Authorization": "Bearer validation-test-token"},
    )
    assert response.status_code == 200
    data2 = response.json()
    assert data2["remaining"] == initial_remaining - 2



@pytest.fixture(autouse=True)
def scoped_token(monkeypatch, mock_db):
    from src.api.middleware import get_valid_tokens
    monkeypatch.setenv("API_BEARER_TOKENS_MAPPING", json.dumps({"validation-test-token": ["read", "write", "scan"]}))
    get_valid_tokens.cache_clear()
    yield
    get_valid_tokens.cache_clear()

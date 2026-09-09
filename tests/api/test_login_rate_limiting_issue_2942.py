"""Real credential requests are throttled after five attempts on both aliases."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.db.auth import add_user


@pytest.mark.parametrize("path", ["/auth/login", "/api/v1/auth/login"])
def test_login_endpoint_rate_limited(path, mock_db):
    add_user("rate_test", "Rate-Test-Password!984", role="teacher")
    with TestClient(app) as client:
        credentials = {"username": "rate_test", "password": "Rate-Test-Password!984"}  # pragma: allowlist secret
        for i in range(5):
            response = client.post(path, json=credentials)
            assert response.status_code == 200, response.text
        response = client.post(path, json=credentials, headers={"X-Forwarded-For": "198.51.100.43"})
        assert response.status_code == 429

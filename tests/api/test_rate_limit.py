from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_rate_limit_endpoint():
    response = client.get(
        "/api/v1/rate_limit",
        headers={"Authorization": "Bearer dummy-token"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["limit"] == 100
    assert data["remaining"] == 85
    assert data["reset_in_seconds"] == 45

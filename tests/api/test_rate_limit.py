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

    assert "limit" in data
    assert "remaining" in data
    assert "reset_in_seconds" in data

    initial_remaining = data["remaining"]
    assert initial_remaining <= data["limit"]

    # Consume one token
    from src.security.rate_limiter import get_token_bucket_limiter
    lim = get_token_bucket_limiter()
    lim.consume("dummy-token")

    # Fetch rate limit status again
    response = client.get(
        "/api/v1/rate_limit",
        headers={"Authorization": "Bearer dummy-token"},
    )
    assert response.status_code == 200
    data2 = response.json()
    assert data2["remaining"] == initial_remaining - 1


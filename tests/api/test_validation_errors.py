from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token

client = TestClient(app)


def test_validation_error_format():
    """A malformed multipart body (missing the required 'file' part) should
    return the standardized 422 validation-error JSON shape."""
    response = client.post(
        "/api/v1/scan",
        headers={
            "Authorization": f"Bearer {get_expected_bearer_token()}",
            "Content-Type": (
                "multipart/form-data; "
                "boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"
            ),
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] is True
    assert "message" in data
    assert "details" in data
    assert isinstance(data["details"], list)

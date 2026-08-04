from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_validation_error_format():
    response = client.post(
        "/api/v1/scan",
        headers={"Authorization": "Bearer dummy-token", "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] is True
    assert "message" in data
    assert "details" in data
    assert isinstance(data["details"], list)

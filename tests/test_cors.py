import importlib
from fastapi.testclient import TestClient

def test_cors_headers(monkeypatch):
    """Verify configured origins receive the correct CORS headers."""
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,https://example.com",
    )

    # Reload the app module so the environment variable is picked up during initialization
    import src.api.app
    importlib.reload(src.api.app)
    from src.api.app import app

    client = TestClient(app)

    response = client.options(
        "/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

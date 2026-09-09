"""Exercise API startup, authentication and protected reads with fresh storage."""

from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.environ["SPD_STATE_DIR"] = tempfile.mkdtemp(prefix="spd-api-smoke-")
    os.environ["APP_ENV"] = "test"
    os.environ["JWT_SECRET_KEY"] = (
        "smoke-only-ephemeral-key-91874-allow-no-external-clients"
    )
    os.environ["PRELOAD_EMBEDDING_MODEL"] = "false"
    os.environ.pop("ADMIN_BOOTSTRAP_PASSWORD", None)
    from fastapi.testclient import TestClient
    from src.api.app import app
    from src.db.auth import add_user

    with TestClient(app) as client:
        add_user("api_smoke", "API-Smoke-Password!824", role="teacher")
        assert client.get("/api/v1/corpus/stats").status_code == 401
        response = client.post(
            "/auth/login",
            json={"username": "api_smoke", "password": "API-Smoke-Password!824"},
        )
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        response = client.get(
            "/api/v1/corpus/stats", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["total_documents"] == 0
        assert client.get("/api/v1/health/scores").status_code == 401
    print("PASS: fresh API startup, password login, signed-token protected corpus read")


if __name__ == "__main__":
    main()

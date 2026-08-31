import os
from fastapi.testclient import TestClient
from main import app  # Assuming FastAPI app instance is imported from main

client = TestClient(app)

def test_clear_endpoint_creates_snapshot(tmp_path):
    response = client.delete("/api/v1/clear?create_snapshot=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    if data["snapshot_backup"]:
        assert os.path.exists(data["snapshot_backup"])

"""
test_background_clustering_issue_3122.py
-----------------------------------------
Unit test suite for Issue #3122:
Validates that document clustering tasks are offloaded to a background FastAPI endpoint,
returning a Task ID and supporting polling until task completion.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import numpy as np

from src.api.app import app
from src.api.dependencies import get_current_user
from src.api.routers.analysis import _process_clustering_job, _get_clustering_job

client = TestClient(app)

# Override user dependency for API testing
app.dependency_overrides[get_current_user] = lambda: {"username": "test_admin", "role": "admin"}


def test_start_background_clustering_returns_task_id():
    """Verify POST /api/v1/analysis/cluster initiates background task and returns task_id."""
    sim_matrix = np.eye(4, dtype=float).tolist()
    payload = {
        "algorithm": "hierarchical",
        "n_clusters": 2,
        "similarity_matrix": sim_matrix,
    }

    response = client.post("/api/v1/analysis/cluster", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    assert "status_url" in data

    task_id = data["task_id"]
    # Manually execute background processing function for testing
    _process_clustering_job(
        task_id=task_id,
        algorithm="hierarchical",
        n_clusters=2,
        linkage="ward",
        embeddings=None,
        similarity_matrix=sim_matrix,
    )

    # Poll status endpoint for results
    status_response = client.get(f"/api/v1/analysis/cluster/task/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "completed"
    assert "result" in status_data
    assert "labels" in status_data["result"]
    assert len(status_data["result"]["labels"]) == 4

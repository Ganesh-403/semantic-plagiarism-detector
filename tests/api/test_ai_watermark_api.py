"""
tests/api/test_ai_watermark_api.py
----------------------------------
Unit and integration tests for AI Watermark REST API endpoints.
"""

from pathlib import Path
import tempfile
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.endpoints.ai_watermark import router as ai_watermark_router
from src.db.watermark_verification_db import initialize_watermark_verification_db

app = FastAPI()
app.include_router(ai_watermark_router)
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Set up an isolated temporary database for each API test run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_api_watermarks.db"
        initialize_watermark_verification_db(test_db)
        monkeypatch.setattr("src.db.watermark_verification_db.DEFAULT_DB_PATH", test_db)
        yield test_db


class TestAIWatermarkAPI:
    """Tests for AI Watermark verification REST endpoints."""

    def test_verify_watermark_endpoint_basic(self):
        payload = {
            "text": "This is a sample document for testing the statistical watermark verification engine.",
            "document_id": "test_doc_001",
            "secret_key": "maryland_seed_42",
            "gamma": 0.5,
            "z_threshold": 4.0,
            "significance_alpha": 0.01,
        }

        response = client.post("/ai-watermark/verify", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "verification_id" in data
        assert data["verification_id"].startswith("WMV-")
        assert data["document_id"] == "test_doc_001"
        assert "z_score" in data
        assert "p_value" in data
        assert "confidence_interval" in data
        assert "lower_bound" in data["confidence_interval"]
        assert "upper_bound" in data["confidence_interval"]
        assert data["confidence_interval"]["confidence_level"] == 0.95
        assert "is_watermarked" in data
        assert isinstance(data["is_watermarked"], bool)
        assert "token_entropy" in data
        assert data["watermark_scheme"] == "Maryland-Kirchenbauer"

    def test_verify_watermark_with_ngrams_and_token_details(self):
        payload = {
            "text": "Artificial intelligence and statistical watermarking in large language models.",
            "document_id": "test_doc_002",
            "include_ngrams": True,
            "include_token_details": True,
        }

        response = client.post("/ai-watermark/verify", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["ngram_frequencies"] is not None
        assert "1" in data["ngram_frequencies"]
        assert "2" in data["ngram_frequencies"]
        assert data["token_details"] is not None
        assert len(data["token_details"]) > 0
        assert "is_green" in data["token_details"][0]

    def test_verify_watermark_validation_errors(self):
        # Empty text
        res_empty = client.post("/ai-watermark/verify", json={"text": ""})
        assert res_empty.status_code == 422

        # Invalid gamma (<= 0.0)
        res_gamma_low = client.post(
            "/ai-watermark/verify", json={"text": "Valid text", "gamma": 0.0}
        )
        assert res_gamma_low.status_code == 422

        # Invalid gamma (>= 1.0)
        res_gamma_high = client.post(
            "/ai-watermark/verify", json={"text": "Valid text", "gamma": 1.2}
        )
        assert res_gamma_high.status_code == 422

        # Invalid z_threshold (< 0.0)
        res_z = client.post(
            "/ai-watermark/verify", json={"text": "Valid text", "z_threshold": -1.0}
        )
        assert res_z.status_code == 422

    def test_get_verification_by_id(self):
        # First create a verification
        post_res = client.post(
            "/ai-watermark/verify",
            json={"text": "A quick test sentence.", "document_id": "doc_lookup"},
        )
        assert post_res.status_code == 200
        v_id = post_res.json()["verification_id"]

        # Retrieve by ID
        get_res = client.get(f"/ai-watermark/verifications/{v_id}")
        assert get_res.status_code == 200
        record = get_res.json()
        assert record["verification_id"] == v_id
        assert record["document_id"] == "doc_lookup"

    def test_get_nonexistent_verification_404(self):
        res = client.get("/ai-watermark/verifications/WMV-NONEXISTENT")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_get_document_verifications(self):
        # Submit two verifications for doc_group_a
        client.post(
            "/ai-watermark/verify",
            json={"text": "First scan for group A.", "document_id": "doc_group_a"},
        )
        client.post(
            "/ai-watermark/verify",
            json={"text": "Second scan for group A.", "document_id": "doc_group_a"},
        )

        res = client.get("/ai-watermark/document/doc_group_a")
        assert res.status_code == 200
        items = res.json()
        assert len(items) == 2
        assert all(item["document_id"] == "doc_group_a" for item in items)

    def test_get_stats_endpoint(self):
        client.post(
            "/ai-watermark/verify",
            json={"text": "Sample text for stats.", "document_id": "doc_stats"},
        )

        res = client.get("/ai-watermark/stats")
        assert res.status_code == 200
        stats = res.json()
        assert stats["status"] == "active"
        assert stats["total_verifications"] >= 1
        assert stats["scheme"] == "Maryland-Kirchenbauer"

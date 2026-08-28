"""
tests/db/test_watermark_verification_db.py
------------------------------------------
Unit tests for Watermark Verification SQLite database persistence.
"""

from pathlib import Path
import tempfile
import pytest

from src.db.watermark_verification_db import (
    WatermarkVerificationDB,
    delete_verification,
    get_verification_by_id,
    get_verification_count,
    get_verifications_for_document,
    initialize_watermark_verification_db,
    list_recent_verifications,
    save_verification_result,
)


@pytest.fixture
def temp_db_path():
    """Provides an isolated temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_watermark_verifications.db"
        initialize_watermark_verification_db(db_file)
        yield db_file


class TestWatermarkVerificationDB:
    """Tests for database persistence and querying of watermark verifications."""

    def test_initialize_database(self, temp_db_path):
        assert temp_db_path.exists()
        count = get_verification_count(temp_db_path)
        assert count == 0

    def test_save_and_retrieve_verification(self, temp_db_path):
        saved = save_verification_result(
            document_id="doc_101",
            total_tokens=150,
            green_tokens=110,
            red_tokens=40,
            observed_green_ratio=0.733333,
            expected_green_ratio=0.5,
            z_score=5.7154,
            p_value=0.000001,
            confidence_level=0.95,
            ci_lower=0.655,
            ci_upper=0.798,
            confidence_score=99.99,
            is_watermarked=True,
            watermark_scheme="Maryland-Kirchenbauer",
            metadata={"secret_key": "custom_key_77", "context_window_size": 1},
            db_path=temp_db_path,
        )

        assert saved is not None
        v_id = saved["verification_id"]
        assert v_id.startswith("WMV-")
        assert saved["document_id"] == "doc_101"
        assert saved["is_watermarked"] is True
        assert saved["total_tokens"] == 150
        assert saved["metadata"]["secret_key"] == "custom_key_77"

        # Query by ID
        fetched = get_verification_by_id(v_id, db_path=temp_db_path)
        assert fetched is not None
        assert fetched["verification_id"] == v_id
        assert fetched["z_score"] == 5.7154
        assert fetched["is_watermarked"] is True

    def test_get_nonexistent_verification(self, temp_db_path):
        assert get_verification_by_id("WMV-NONEXISTENT", db_path=temp_db_path) is None

    def test_get_verifications_for_document(self, temp_db_path):
        for i in range(5):
            save_verification_result(
                document_id="doc_batch_1",
                total_tokens=100 + i,
                green_tokens=50 + i,
                red_tokens=50,
                observed_green_ratio=0.5,
                expected_green_ratio=0.5,
                z_score=0.0,
                p_value=0.5,
                confidence_level=0.95,
                ci_lower=0.4,
                ci_upper=0.6,
                confidence_score=50.0,
                is_watermarked=False,
                db_path=temp_db_path,
            )

        # Another document
        save_verification_result(
            document_id="doc_other",
            total_tokens=100,
            green_tokens=80,
            red_tokens=20,
            observed_green_ratio=0.8,
            expected_green_ratio=0.5,
            z_score=6.0,
            p_value=0.0001,
            confidence_level=0.95,
            ci_lower=0.7,
            ci_upper=0.9,
            confidence_score=99.9,
            is_watermarked=True,
            db_path=temp_db_path,
        )

        doc_verifs = get_verifications_for_document("doc_batch_1", limit=10, db_path=temp_db_path)
        assert len(doc_verifs) == 5

        # Test pagination
        page_verifs = get_verifications_for_document("doc_batch_1", limit=2, offset=0, db_path=temp_db_path)
        assert len(page_verifs) == 2

        other_verifs = get_verifications_for_document("doc_other", db_path=temp_db_path)
        assert len(other_verifs) == 1
        assert other_verifs[0]["is_watermarked"] is True

    def test_list_recent_and_count(self, temp_db_path):
        for i in range(7):
            save_verification_result(
                document_id=f"doc_{i}",
                total_tokens=100,
                green_tokens=50,
                red_tokens=50,
                observed_green_ratio=0.5,
                expected_green_ratio=0.5,
                z_score=0.0,
                p_value=0.5,
                confidence_level=0.95,
                ci_lower=0.4,
                ci_upper=0.6,
                confidence_score=50.0,
                is_watermarked=False,
                db_path=temp_db_path,
            )

        assert get_verification_count(temp_db_path) == 7

        recent = list_recent_verifications(limit=4, offset=0, db_path=temp_db_path)
        assert len(recent) == 4

    def test_delete_verification(self, temp_db_path):
        saved = save_verification_result(
            document_id="doc_delete_me",
            total_tokens=100,
            green_tokens=50,
            red_tokens=50,
            observed_green_ratio=0.5,
            expected_green_ratio=0.5,
            z_score=0.0,
            p_value=0.5,
            confidence_level=0.95,
            ci_lower=0.4,
            ci_upper=0.6,
            confidence_score=50.0,
            is_watermarked=False,
            db_path=temp_db_path,
        )
        v_id = saved["verification_id"]

        assert delete_verification(v_id, db_path=temp_db_path) is True
        assert get_verification_by_id(v_id, db_path=temp_db_path) is None
        assert delete_verification("WMV-NONEXISTENT", db_path=temp_db_path) is False

    def test_class_wrapper_interface(self, temp_db_path):
        db_wrapper = WatermarkVerificationDB(db_path=temp_db_path)
        res = db_wrapper.save(
            document_id="doc_wrapper",
            total_tokens=200,
            green_tokens=150,
            red_tokens=50,
            observed_green_ratio=0.75,
            expected_green_ratio=0.5,
            z_score=7.07,
            p_value=1e-10,
            confidence_level=0.95,
            ci_lower=0.68,
            ci_upper=0.81,
            confidence_score=99.99,
            is_watermarked=True,
        )
        assert res is not None
        v_id = res["verification_id"]

        assert db_wrapper.get(v_id) is not None
        assert len(db_wrapper.get_by_document("doc_wrapper")) == 1
        assert len(db_wrapper.list_recent(limit=10)) >= 1
        assert db_wrapper.count() >= 1
        assert db_wrapper.delete(v_id) is True

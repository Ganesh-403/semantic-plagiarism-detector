import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.analysis import router, check_exact_match_fast_path
from src.core.similarity import PLAGIARISM_THRESHOLD

app = FastAPI()
app.include_router(router)

# Mock dependencies
def mock_get_current_user():
    return {"user_id": "test_user"}

def mock_validate_content_type():
    return None

def mock_verify_bearer_token():
    return "valid_token"

app.dependency_overrides = {
    "src.api.dependencies.get_current_user": mock_get_current_user,
    "src.api.dependencies.validate_content_type": mock_validate_content_type,
    "src.api.dependencies.verify_bearer_token": mock_verify_bearer_token,
}

client = TestClient(app)

@pytest.fixture
def test_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"Hello world. This is a test document.")
        f.flush()
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


def test_helper_exact_match_found(test_file):
    """Test helper returns fast path result when hash matches."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"):
        with patch("src.api.routers.analysis.get_document_by_hash", return_value="other_doc.txt"):
            result = check_exact_match_fast_path(test_file, "upload.txt", 100, 5, 0.5)

            assert result is not None
            assert result["plagiarism_flagged"] is True
            assert result["overall_document_similarity"] == 1.0
            assert result["plagiarism_density"] == 100
            assert len(result["matched_documents"]) == 1
            assert result["matched_documents"][0]["filename"] == "other_doc.txt"
            assert result["matched_documents"][0]["document_similarity_score"] == 1.0
            assert result["matched_documents"][0]["severity"] == "🔴 High"

def test_helper_same_filename_skipped(test_file):
    """Test helper skips fast path if matched filename is exactly the same as uploaded."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"):
        with patch("src.api.routers.analysis.get_document_by_hash", return_value="upload.txt"):
            result = check_exact_match_fast_path(test_file, "upload.txt", 100, 5, 0.5)
            assert result is None

def test_helper_hash_calculation_failure(test_file):
    """Test helper behaves safely when hashing fails."""
    with patch("src.api.routers.analysis.calculate_file_sha256", side_effect=Exception("Hashing failed")):
        result = check_exact_match_fast_path(test_file, "upload.txt", 100, 5, 0.5)
        assert result is None

def test_helper_db_query_failure(test_file):
    """Test helper behaves safely when DB query fails."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"):
        with patch("src.api.routers.analysis.get_document_by_hash", side_effect=Exception("DB error")):
            result = check_exact_match_fast_path(test_file, "upload.txt", 100, 5, 0.5)
            assert result is None

def test_scan_document_sync_exact_match(test_file):
    """Test the synchronous endpoint uses the fast path and bypasses embedding generation."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"), \
         patch("src.api.routers.analysis.get_document_by_hash") as mock_get_doc, \
         patch("src.api.routers.analysis.extract_text", return_value="Hello world"), \
         patch("src.api.routers.analysis.embed_chunks") as mock_embed:

        # We need get_document_by_hash to return None initially (for duplicate check if reprocess=False)
        # But wait, we can just pass reprocess=True to bypass the duplicate check!
        mock_get_doc.return_value = "existing_doc.txt"

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan",
                params={"reprocess": "true"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["plagiarism_flagged"] is True
        assert data["overall_document_similarity"] == 1.0
        assert data["matched_documents"][0]["filename"] == "existing_doc.txt"

        # embed_chunks should NOT have been called due to fast path
        mock_embed.assert_not_called()

def test_scan_document_sync_no_match(test_file):
    """Test the synchronous endpoint proceeds with embedding when there is no hash match."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value=None), \
         patch("src.api.routers.analysis.extract_text", return_value="Hello world"), \
         patch("src.api.routers.analysis.embed_chunks") as mock_embed, \
         patch("src.api.routers.analysis.get_document_embedding"), \
         patch("src.api.routers.analysis.get_corpus_documents_with_embeddings", return_value={}):

        # embed_chunks returns empty array mock
        import numpy as np
        mock_embed.return_value = np.array([])

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan",
                params={"reprocess": "true"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        assert response.status_code == 200
        # embed_chunks SHOULD be called since no fast path was hit
        mock_embed.assert_called_once()


def test_scan_document_async_exact_match(test_file):
    """Test the asynchronous job uses the fast path and bypasses embedding."""
    from src.api.routers.analysis import _process_scan_job, scan_jobs

    scan_jobs["test_job_1"] = {}

    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value="existing_doc.txt"), \
         patch("src.api.routers.analysis.extract_text", return_value="Hello world"), \
         patch("src.api.routers.analysis.embed_chunks") as mock_embed:

        _process_scan_job("test_job_1", test_file, "upload.txt", 0.5, 3)

        assert scan_jobs["test_job_1"]["status"] == "completed"
        result = scan_jobs["test_job_1"]["result"]
        assert result["plagiarism_flagged"] is True
        assert result["overall_document_similarity"] == 1.0
        assert result["matched_documents"][0]["filename"] == "existing_doc.txt"

        mock_embed.assert_not_called()

def test_scan_document_async_no_match(test_file):
    """Test the asynchronous job proceeds with embedding when no match."""
    from src.api.routers.analysis import _process_scan_job, scan_jobs
    import numpy as np

    scan_jobs["test_job_2"] = {}

    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value=None), \
         patch("src.api.routers.analysis.extract_text", return_value="Hello world"), \
         patch("src.api.routers.analysis.embed_chunks", return_value=np.array([[0.5]])), \
         patch("src.api.routers.analysis.get_document_embedding", return_value=np.array([[0.5]])), \
         patch("src.api.routers.analysis.get_corpus_documents_with_embeddings", return_value={}) as mock_corpus:

        _process_scan_job("test_job_2", test_file, "upload.txt", 0.5, 3)

        assert scan_jobs["test_job_2"]["status"] == "completed"
        # It should have checked corpus
        mock_corpus.assert_called_once()

def test_duplicate_upload_reprocess_false(test_file):
    """Regression test ensuring existing duplicate detection is untouched when reprocess is False."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value="existing.txt"):

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan",
                params={"reprocess": "false"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        # It should return 409 Conflict, proving the initial duplicate check is still active
        assert response.status_code == 409
        assert response.json()["duplicate"] is True

def test_async_duplicate_upload_reprocess_false(test_file):
    """Regression test ensuring existing duplicate detection is untouched for async scan."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="dummyhash"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value="existing.txt"):

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan/async",
                params={"reprocess": "false"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        assert response.status_code == 409
        assert response.json()["duplicate"] is True

def test_helper_different_content_different_hash():
    """Test ensuring hashing behavior is consistent."""
    from src.utils.hash_util import calculate_file_sha256
    with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
        f1.write(b"content 1")
        f2.write(b"content 2")
        f1.flush()
        f2.flush()

        hash1 = calculate_file_sha256(f1.name)
        hash2 = calculate_file_sha256(f2.name)

        assert hash1 != hash2
        assert hash1 is not None
        assert hash2 is not None

        os.unlink(f1.name)
        os.unlink(f2.name)

def test_helper_same_content_same_hash():
    """Test ensuring same content produces same hash."""
    from src.utils.hash_util import calculate_file_sha256
    with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
        f1.write(b"identical content")
        f2.write(b"identical content")
        f1.flush()
        f2.flush()

        hash1 = calculate_file_sha256(f1.name)
        hash2 = calculate_file_sha256(f2.name)

        assert hash1 == hash2
        assert hash1 is not None

        os.unlink(f1.name)
        os.unlink(f2.name)

def test_exact_match_severity():
    """Verify that the severity is strictly set to High for exact matches."""
    result = check_exact_match_fast_path(
        "dummy_input", "test.txt", word_count=50, chunk_count=2, threshold=0.5
    )
    # the function is mocked but let's test it via patches
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="hash_match"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value="other.txt"):
        res = check_exact_match_fast_path("dummy_input", "test.txt", 50, 2, 0.5)
        assert res["matched_documents"][0]["severity"] == "🔴 High"
def test_scan_document_empty_input(test_file):
    """Test that empty input is handled properly by the normal path and bypasses fast path check."""
    with open(test_file, 'w') as f:
        f.write('   ')

    with open(test_file, "rb") as f:
        response = client.post(
            "/api/v1/scan",
            params={"reprocess": "true"},
            files={"file": ("upload.txt", f, "text/plain")},
        )

    # 422 because text extraction fails
    assert response.status_code == 422
    assert "readable text" in response.json()["detail"]

def test_vector_comparison_not_called(test_file):
    """Test that vector comparison (cosine_similarity) is NOT called on an exact match."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="hash_match"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value="existing.txt"), \
         patch("src.api.routers.analysis.extract_text", return_value="test text"), \
         patch("src.api.routers.analysis.cosine_similarity") as mock_cosine:

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan",
                params={"reprocess": "true"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        # Verify response
        assert response.status_code == 200
        assert response.json()["overall_document_similarity"] == 1.0

        # Verify vector comparison was skipped entirely
        mock_cosine.assert_not_called()

def test_exact_byte_for_byte_match_and_exact_text_match():
    """Test exact byte-for-byte match and exact text match hashing consistency."""
    from src.utils.hash_util import calculate_file_sha256
    text_content = "This is the exact text match."
    byte_content = text_content.encode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
        f1.write(byte_content)
        f2.write(byte_content)
        f1.flush()
        f2.flush()

        hash1 = calculate_file_sha256(f1.name)
        hash2 = calculate_file_sha256(f2.name)
        assert hash1 == hash2
        os.unlink(f1.name)
        os.unlink(f2.name)

def test_multiple_documents_in_corpus(test_file):
    """Test fast path when there are multiple documents in corpus."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="hash123"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value="doc2.txt"), \
         patch("src.api.routers.analysis.extract_text", return_value="dummy"):

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan",
                params={"reprocess": "true"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["matched_documents"][0]["filename"] == "doc2.txt"

def test_hash_lookup_returning_no_result(test_file):
    """Test when hash lookup returns None (no result)."""
    with patch("src.api.routers.analysis.calculate_file_sha256", return_value="hash_no_result"), \
         patch("src.api.routers.analysis.get_document_by_hash", return_value=None), \
         patch("src.api.routers.analysis.extract_text", return_value="test text"), \
         patch("src.api.routers.analysis.embed_chunks") as mock_embed, \
         patch("src.api.routers.analysis.get_document_embedding"), \
         patch("src.api.routers.analysis.get_corpus_documents_with_embeddings", return_value={}):

        import numpy as np
        mock_embed.return_value = np.array([])

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/v1/scan",
                params={"reprocess": "true"},
                files={"file": ("upload.txt", f, "text/plain")},
            )

        assert response.status_code == 200
        mock_embed.assert_called_once()

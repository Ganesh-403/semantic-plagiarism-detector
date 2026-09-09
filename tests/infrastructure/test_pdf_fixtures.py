from pathlib import Path
import pytest
import json
from src.errors import EmptyDocumentError

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token
from src.core.document_parser import extract_text

client = TestClient(app)

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def isolated_api(monkeypatch, mock_db, mock_embed_chunks):
    from src.api.middleware import get_valid_tokens
    monkeypatch.setenv("API_BEARER_TOKENS_MAPPING", json.dumps({"pdf-fixture-token": ["write", "scan"]}))
    get_valid_tokens.cache_clear()
    monkeypatch.setattr("src.api.routers.analysis.embed_chunks", mock_embed_chunks)
    monkeypatch.setattr("src.api.routers.analysis.get_corpus_documents_with_embeddings", lambda: {})
    yield
    get_valid_tokens.cache_clear()


def test_clean_pdf_extraction():
    """Verify that a standard clean PDF has its text extracted successfully."""
    clean_pdf_path = FIXTURES_DIR / "clean.pdf"
    assert clean_pdf_path.exists(), "clean.pdf fixture is missing"

    text = extract_text(clean_pdf_path.read_bytes(), "clean.pdf")
    assert "Artificial Intelligence" in text
    assert "clean PDF" in text


def test_encrypted_pdf_extraction():
    """Verify that an encrypted PDF is handled gracefully (returns empty text)."""
    encrypted_pdf_path = FIXTURES_DIR / "encrypted.pdf"
    assert encrypted_pdf_path.exists(), "encrypted.pdf fixture is missing"

    with pytest.raises(EmptyDocumentError):
        extract_text(encrypted_pdf_path.read_bytes(), "encrypted.pdf")


def test_scanned_pdf_extraction():
    """Verify that a scanned PDF triggers OCR and extracts text successfully (or fails if Tesseract missing)."""
    scanned_pdf_path = FIXTURES_DIR / "scanned.pdf"
    assert scanned_pdf_path.exists(), "scanned.pdf fixture is missing"

    import shutil

    from src.core.document_parser import OCRDependencyError

    if not shutil.which("tesseract"):
        import pytest

        with pytest.raises(OCRDependencyError):
            extract_text(scanned_pdf_path.read_bytes(), "scanned.pdf")
    else:
        text = extract_text(scanned_pdf_path.read_bytes(), "scanned.pdf")
        # The OCR should pick up the text rendered into the image
        assert "Artificial Intelligence" in text
        assert "scanned text" in text


def test_api_upload_clean_pdf():
    """Verify that the API processes the clean PDF correctly."""
    clean_pdf_path = FIXTURES_DIR / "clean.pdf"
    expected_token = "pdf-fixture-token"

    with open(clean_pdf_path, "rb") as f:
        response = client.post(
            "/api/v1/scan",
            headers={"Authorization": f"Bearer {expected_token}"},
            files={"file": ("clean.pdf", f, "application/pdf")},
        )

    # We expect 200 OK since text will be extracted
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "clean.pdf"
    assert data["chunk_count"] >= 1


def test_api_upload_encrypted_pdf():
    """Verify that the API handles encrypted PDFs properly."""
    encrypted_pdf_path = FIXTURES_DIR / "encrypted.pdf"
    expected_token = "pdf-fixture-token"

    with open(encrypted_pdf_path, "rb") as f:
        response = client.post(
            "/api/v1/scan",
            headers={"Authorization": f"Bearer {expected_token}"},
            files={"file": ("encrypted.pdf", f, "application/pdf")},
        )

    # We expect 422 Unprocessable Entity since no readable text can be extracted
    assert response.status_code == 422
    assert "Failed to extract readable text" in response.json()["detail"]

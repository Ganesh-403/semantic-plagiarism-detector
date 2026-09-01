"""
test_pdf_highlighter.py
-----------------------
Unit tests for src.utils.pdf_highlighter.highlight_pdf_matches on encrypted PDFs.
"""

from __future__ import annotations

import fitz
import pytest

from src.errors import PDFEncryptedError
from src.utils.pdf_highlighter import highlight_pdf_matches


def _create_encrypted_pdf(password: str = "secret123") -> bytes:
    """Generate an in-memory PDF encrypted with the given user password."""
    doc = fitz.open()
    page = doc.new_page()
    sample_text = (
        "This is a sample document containing matching target content for plagiarism testing."
    )
    page.insert_text((50, 50), sample_text, fontsize=12)

    # Save encrypted PDF with AES-256 encryption
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw="owner_" + password,
    )
    doc.close()
    return pdf_bytes


def test_highlight_pdf_matches_encrypted_pdf_with_correct_password():
    """Verify that encrypted PDFs with password 'secret123' are authenticated and produce highlighted annotations."""
    password = "secret123"
    pdf_bytes = _create_encrypted_pdf(password=password)
    matching_phrase = "matching target content"

    # Call highlight_pdf_matches with the correct password
    highlighted_bytes = highlight_pdf_matches(
        pdf_bytes=pdf_bytes,
        matching_phrases=[matching_phrase],
        password=password,
    )

    assert isinstance(highlighted_bytes, bytes)
    assert len(highlighted_bytes) > 0

    # Verify annotations in the output PDF
    with fitz.open(stream=highlighted_bytes, filetype="pdf") as doc:
        auth_status = doc.authenticate(password)
        assert auth_status > 0, "PDF authentication failed for output document"
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) >= 1
        assert annots[0].type[0] == 8  # 8 is Highlight annotation in PyMuPDF spec


def test_highlight_pdf_matches_encrypted_pdf_invalid_password_raises_error():
    """Verify that calling highlight_pdf_matches on an encrypted PDF with an invalid password raises PDFEncryptedError."""
    pdf_bytes = _create_encrypted_pdf(password="secret123")

    with pytest.raises(PDFEncryptedError) as exc_info:
        highlight_pdf_matches(
            pdf_bytes=pdf_bytes,
            matching_phrases=["matching target content"],
            password="wrongpassword",
        )

    assert "PDF is encrypted and password was not provided or invalid." in str(exc_info.value)

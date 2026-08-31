"""
test_pdf_highlighter_encrypted_issue_3978.py
--------------------------------------------
Unit tests for Issue #3978: Supporting encrypted PDFs with owner/user passwords
in pdf_highlighter.py and raising PDFEncryptedError when authentication fails.
"""

from __future__ import annotations

import fitz
import pytest

from src.errors import PDFEncryptedError
from src.utils.pdf_highlighter import highlight_pdf_matches


@pytest.fixture
def encrypted_pdf_bytes() -> bytes:
    """Create an encrypted PDF byte stream with user and owner passwords."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "This is a confidential sample document containing overlapping text phrases for testing.",
    )
    return doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="OwnerSecret123",
        user_pw="UserSecret123",
    )


@pytest.fixture
def unencrypted_pdf_bytes() -> bytes:
    """Create an unencrypted PDF byte stream."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "This is an unencrypted sample document containing overlapping text phrases for testing.",
    )
    return doc.tobytes()


def test_encrypted_pdf_no_password_raises_pdf_encrypted_error(encrypted_pdf_bytes: bytes):
    """Verify that calling highlight_pdf_matches on an encrypted PDF without a password raises PDFEncryptedError."""
    with pytest.raises(PDFEncryptedError) as exc_info:
        highlight_pdf_matches(
            pdf_bytes=encrypted_pdf_bytes,
            matching_phrases=["confidential sample document"],
        )

    assert "PDF is encrypted and password was not provided or invalid." in str(exc_info.value)


def test_encrypted_pdf_wrong_password_raises_pdf_encrypted_error(encrypted_pdf_bytes: bytes):
    """Verify that calling highlight_pdf_matches on an encrypted PDF with wrong password raises PDFEncryptedError."""
    with pytest.raises(PDFEncryptedError) as exc_info:
        highlight_pdf_matches(
            pdf_bytes=encrypted_pdf_bytes,
            matching_phrases=["confidential sample document"],
            password="WrongPassword999",
        )

    assert "PDF is encrypted and password was not provided or invalid." in str(exc_info.value)


def test_encrypted_pdf_correct_password_succeeds(encrypted_pdf_bytes: bytes):
    """Verify that calling highlight_pdf_matches on an encrypted PDF with correct user password succeeds."""
    result = highlight_pdf_matches(
        pdf_bytes=encrypted_pdf_bytes,
        matching_phrases=["confidential sample document"],
        password="UserSecret123",
    )

    assert result != b""
    res_doc = fitz.open(stream=result, filetype="pdf")
    assert res_doc.page_count == 1


def test_encrypted_pdf_correct_owner_password_succeeds(encrypted_pdf_bytes: bytes):
    """Verify that calling highlight_pdf_matches on an encrypted PDF with correct owner password succeeds."""
    result = highlight_pdf_matches(
        pdf_bytes=encrypted_pdf_bytes,
        matching_phrases=["confidential sample document"],
        password="OwnerSecret123",
    )

    assert result != b""
    res_doc = fitz.open(stream=result, filetype="pdf")
    assert res_doc.page_count == 1


def test_unencrypted_pdf_succeeds(unencrypted_pdf_bytes: bytes):
    """Verify that calling highlight_pdf_matches on an unencrypted PDF works normally."""
    result = highlight_pdf_matches(
        pdf_bytes=unencrypted_pdf_bytes,
        matching_phrases=["unencrypted sample document"],
    )

    assert result != b""
    res_doc = fitz.open(stream=result, filetype="pdf")
    assert res_doc.page_count == 1

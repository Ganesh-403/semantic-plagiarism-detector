"""
tests/utils/test_file_parser.py
--------------------------------
Unit tests for password-protected PDF parsing.
"""

import fitz
import pytest

from src.utils.file_parser import EncryptedPDFError, extract_text_from_pdf


def test_encrypted_pdf_handling():
    # 1. Create an in-memory encrypted PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Confidential Student Assignment")
    
    # Save with encryption password 'secret123'
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret123",
        owner_pw="owner123",
    )
    doc.close()

    # 2. Test reading without password -> should raise EncryptedPDFError
    with pytest.raises(EncryptedPDFError):
        extract_text_from_pdf(pdf_bytes)

    # 3. Test reading with wrong password -> should raise EncryptedPDFError
    with pytest.raises(EncryptedPDFError):
        extract_text_from_pdf(pdf_bytes, password="wrongpass")

    # 4. Test reading with correct password -> should succeed
    text, is_protected = extract_text_from_pdf(pdf_bytes, password="secret123")
    assert "Confidential Student Assignment" in text
    assert is_protected is True
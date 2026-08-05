"""
tests/utils/test_file_parser.py
--------------------------------
Includes tests for password-protected PDF parsing and MIME categorization.
"""

import fitz
import pytest

from src.utils.file_parser import (
    EncryptedPDFError,
    extract_pdf_metadata,
    extract_text_from_pdf,
    get_file_size_formatted,
    get_file_mime_category,
    get_supported_mime_categories,
    is_extension_supported,
)


class TestEncryptedPDFHandling:
    """Test suite for password-protected PDF parsing."""

    def test_encrypted_pdf_handling(self):
        """Test reading encrypted PDFs with no password, wrong password, and correct password."""
        # 1. Create an in-memory encrypted PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Confidential Student Assignment")

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


class TestFileSizeFormatting:
    """Test suite for file size formatting utility."""

    def test_get_file_size_formatted_bytes(self):
        assert get_file_size_formatted(500) == "500 B"

    def test_get_file_size_formatted_kb(self):
        assert get_file_size_formatted(1024) == "1.00 KB"

    def test_get_file_size_formatted_mb(self):
        assert get_file_size_formatted(1024 * 1024) == "1.00 MB"

    def test_get_file_size_formatted_gb(self):
        assert get_file_size_formatted(1024 * 1024 * 1024) == "1.00 GB"

    def test_get_file_size_formatted_fractional(self):
        assert get_file_size_formatted(1536) == "1.50 KB"


class TestFileMimeCategory:
    """Test suite for MIME categorization helpers."""

    @pytest.mark.parametrize(
        "filename, expected_category",
        [
            ("document.pdf", "pdf"),
            ("report.PDF", "pdf"),  # Case insensitive
            ("essay.docx", "word_document"),
            ("notes.doc", "word_document"),
            ("readme.txt", "text"),
            ("documentation.md", "text"),
            ("guide.markdown", "text"),
            ("notes.mdown", "text"),
            ("NOTES.MARKDOWN", "text"),  # Case insensitive
            ("data.csv", "text"),
            ("script.py", "code"),
            ("app.js", "code"),
            ("Main.java", "code"),
            ("archive.zip", "archive"),
            ("backup.tar.gz", "archive"),  # Splits on last dot, so 'gz' -> archive
            ("no_extension", "unknown"),
            ("", "unknown"),
            (".hidden_file", "unknown"),
            (None, "unknown"),
            (12345, "unknown"),  # Non-string input
        ]
    )
    def test_get_file_mime_category(self, filename, expected_category):
        """Test MIME categorization for various file extensions and edge cases."""
        assert get_file_mime_category(filename) == expected_category

    def test_get_supported_mime_categories(self):
        """Test retrieval of supported categories list."""
        categories = get_supported_mime_categories()
        assert isinstance(categories, list)
        assert "pdf" in categories
        assert "word_document" in categories
        assert "text" in categories
        assert "code" in categories
        assert "archive" in categories
        assert "unknown" in categories

    @pytest.mark.parametrize(
        "filename, allowed_categories, expected_result",
        [
            ("document.pdf", ["pdf", "text"], True),
            ("script.py", ["pdf", "text"], False),
            ("notes.txt", None, True),  # Defaults to all known except unknown
            ("guide.markdown", None, True),
            ("notes.mdown", None, True),
            ("archive.zip", ["text", "code"], False),
        ]
    )
    def test_is_extension_supported(self, filename, allowed_categories, expected_result):
        """Test extension support validation against allowed categories."""
        assert is_extension_supported(filename, allowed_categories) == expected_result


class TestPdfMetadataExtraction:
    """Test suite for PDF metadata extraction."""

    def _create_pdf(self, metadata: dict) -> bytes:
        """Create an in-memory PDF with the given metadata."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Metadata Test Content")
        doc.set_metadata(metadata)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_extract_pdf_metadata_with_all_fields(self):
        """Test extracting metadata from a PDF with complete metadata."""
        pdf_bytes = self._create_pdf(
            {
                "title": "Test Report",
                "author": "Rishab",
                "creationDate": "D:20240101120000Z",
                "modDate": "D:20240201120000Z",
            }
        )
        result = extract_pdf_metadata(pdf_bytes)
        assert result["title"] == "Test Report"
        assert result["author"] == "Rishab"
        assert result["creation_date"] == "D:20240101120000Z"
        assert result["mod_date"] == "D:20240201120000Z"
        assert result["page_count"] == 1

    def test_extract_pdf_metadata_missing_fields_default_to_none(self):
        """Test that empty or missing metadata fields become None."""
        pdf_bytes = self._create_pdf({})
        result = extract_pdf_metadata(pdf_bytes)
        assert result["title"] is None
        assert result["author"] is None
        assert result["creation_date"] is None
        assert result["mod_date"] is None
        assert result["page_count"] == 1

    def test_extract_pdf_metadata_page_count(self):
        """Test that page_count reflects the number of pages in the PDF."""
        doc = fitz.open()
        doc.new_page()
        doc.new_page()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()
        result = extract_pdf_metadata(pdf_bytes)
        assert result["page_count"] == 3

    def test_extract_pdf_metadata_encrypted_raises(self):
        """Test that encrypted PDFs raise EncryptedPDFError."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Confidential")
        pdf_bytes = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            user_pw="secret123",
            owner_pw="owner123",
        )
        doc.close()
        with pytest.raises(EncryptedPDFError):
            extract_pdf_metadata(pdf_bytes)

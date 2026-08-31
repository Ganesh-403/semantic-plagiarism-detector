"""
tests/utils/test_file_parser.py
--------------------------------
Includes tests for password-protected PDF parsing and MIME categorization.
"""

import logging

import fitz
import pytest

from src.utils.file_parser import (
    EncryptedPDFError,
    extract_text_from_pdf,
    get_file_mime_category,
    get_file_size_formatted,
    get_supported_mime_categories,
    is_extension_supported,
    validate_pdf_page_count,
)

logger = logging.getLogger(__name__)


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
        ],
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
        ],
    )
    def test_is_extension_supported(
        self, filename, allowed_categories, expected_result
    ):
        """Test extension support validation against allowed categories."""
        assert is_extension_supported(filename, allowed_categories) == expected_result


class TestPDFPageCountValidation:
    """Tests for the PDF page-count safety guard."""

    @staticmethod
    def _make_pdf(page_count: int) -> bytes:
        doc = fitz.open()
        for page_number in range(page_count):
            page = doc.new_page()
            page.insert_text(
                (50, 50),
                f"Page {page_number + 1}",
            )
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_validate_pdf_page_count_returns_page_count(self):
        pdf_bytes = self._make_pdf(3)

        assert validate_pdf_page_count(pdf_bytes) == 3

    def test_validate_pdf_page_count_allows_exact_limit(self):
        pdf_bytes = self._make_pdf(5)

        assert (
            validate_pdf_page_count(
                pdf_bytes,
                max_pages=5,
            )
            == 5
        )

    def test_validate_pdf_page_count_rejects_over_default_limit(
        self,
    ):
        # Avoid constructing 501 real pages by mocking the opened document.
        class FakePDF:
            page_count = 501

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        fake_pdf = FakePDF()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "src.utils.file_parser.fitz.open",
                lambda **_kwargs: fake_pdf,
            )
            with pytest.raises(
                ValueError,
                match=(r"^PDF exceeds maximum allowed page limit " r"\(500 pages\)$"),
            ):
                validate_pdf_page_count(b"%PDF-test")

        assert fake_pdf.closed is True

    def test_validate_pdf_page_count_rejects_custom_limit(self):
        pdf_bytes = self._make_pdf(4)

        with pytest.raises(
            ValueError,
            match=(r"^PDF exceeds maximum allowed page limit " r"\(3 pages\)$"),
        ):
            validate_pdf_page_count(
                pdf_bytes,
                max_pages=3,
            )

    @pytest.mark.parametrize("max_pages", [0, -1])
    def test_validate_pdf_page_count_rejects_non_positive_limit(
        self,
        max_pages,
    ):
        with pytest.raises(
            ValueError,
            match="max_pages must be at least 1",
        ):
            validate_pdf_page_count(
                b"%PDF-test",
                max_pages=max_pages,
            )

    @pytest.mark.parametrize(
        "max_pages",
        [True, 1.5, "500", None],
    )
    def test_validate_pdf_page_count_rejects_non_integer_limit(
        self,
        max_pages,
    ):
        with pytest.raises(
            TypeError,
            match="max_pages must be an integer",
        ):
            validate_pdf_page_count(
                b"%PDF-test",
                max_pages=max_pages,
            )

    @pytest.mark.parametrize(
        "file_bytes",
        ["pdf", 123, None],
    )
    def test_validate_pdf_page_count_rejects_non_bytes_input(
        self,
        file_bytes,
    ):
        with pytest.raises(
            TypeError,
            match="file_bytes must be bytes-like",
        ):
            validate_pdf_page_count(file_bytes)

    def test_validate_pdf_page_count_rejects_malformed_pdf(self):
        with pytest.raises(fitz.FileDataError):
            validate_pdf_page_count(
                b"this is not a valid PDF",
            )

    def test_extract_text_from_pdf_applies_page_count_guard(
        self,
        monkeypatch,
    ):
        calls = []

        def fake_guard(file_bytes, max_pages=500):
            calls.append((file_bytes, max_pages))
            raise ValueError("PDF exceeds maximum allowed page limit (500 pages)")

        monkeypatch.setattr(
            "src.utils.file_parser.validate_pdf_page_count",
            fake_guard,
        )

        with pytest.raises(
            ValueError,
            match=(r"^PDF exceeds maximum allowed page limit " r"\(500 pages\)$"),
        ):
            extract_text_from_pdf(b"oversized")

        assert calls == [(b"oversized", 500)]

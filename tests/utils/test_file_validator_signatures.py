"""
tests/utils/test_file_validator_signatures.py
-----------------------------------------------
Comprehensive unit tests for MAGIC_SIGNATURES in file_validator.py.

Asserts that every extension defined in MAGIC_SIGNATURES correctly validates
matching headers and rejects invalid/mismatched headers.
"""

import pytest
from src.utils.file_validator import MAGIC_SIGNATURES, FileValidator


class TestMagicSignaturesValidHeaders:
    """Test suite verifying that valid headers pass for every format in MAGIC_SIGNATURES."""

    @pytest.mark.parametrize(
        "ext,header_data",
        [
            (".pdf", b"%PDF-1.7 sample content"),
            (".docx", b"PK\x03\x04\x14\x00\x06\x00word/document.xml"),
            (".doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1sample binary doc"),
            (".rtf", b"{\\rtf1\\ansi\\deff0 sample rtf text}"),
            (".odt", b"PK\x03\x04\x14\x00\x06\x00content.xml"),
            (".epub", b"PK\x03\x04" + b"\x00" * 20 + b"mimetypeapplication/epub+zip"),
            (".png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
            (".jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF"),
            (".jpeg", b"\xff\xd8\xff\xe0\x00\x10JFIF"),
        ],
    )
    def test_valid_signatures_pass_in_strict_mode(self, ext: str, header_data: bytes):
        """Each extension with its proper magic header must pass validation."""
        validator = FileValidator(strict_mode=True)
        filename = f"sample_document{ext}"
        result = validator.validate(header_data, filename)

        assert result.is_valid is True, f"Failed validation for {ext}: {result.error_message}"
        assert result.error_code is None


class TestMagicSignaturesMismatchedHeaders:
    """Test suite verifying that mismatched headers are rejected in strict mode."""

    def test_png_header_with_pdf_extension_fails(self):
        """PNG magic bytes with a .pdf extension must fail with MAGIC_BYTE_MISMATCH."""
        validator = FileValidator(strict_mode=True)
        png_data = b"\x89PNG\r\n\x1a\nfake png data"
        result = validator.validate(png_data, "fake_doc.pdf")

        assert result.is_valid is False
        assert result.error_code == "MAGIC_BYTE_MISMATCH"
        assert "does not match extension" in result.error_message

    def test_pdf_header_with_png_extension_fails(self):
        """PDF magic bytes with a .png extension must fail with MAGIC_BYTE_MISMATCH."""
        validator = FileValidator(strict_mode=True)
        pdf_data = b"%PDF-1.4 fake pdf data"
        result = validator.validate(pdf_data, "fake_image.png")

        assert result.is_valid is False
        assert result.error_code == "MAGIC_BYTE_MISMATCH"

    def test_jpg_header_with_docx_extension_fails(self):
        """JPG magic bytes with a .docx extension must fail with MAGIC_BYTE_MISMATCH."""
        validator = FileValidator(strict_mode=True)
        jpg_data = b"\xff\xd8\xff\xe0fake jpg data"
        result = validator.validate(jpg_data, "fake_word.docx")

        assert result.is_valid is False
        assert result.error_code == "MAGIC_BYTE_MISMATCH"

    def test_exe_mz_header_with_doc_extension_fails(self):
        """Executable (MZ) header with a .doc extension must fail with MAGIC_BYTE_MISMATCH."""
        validator = FileValidator(strict_mode=True)
        exe_data = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00"
        result = validator.validate(exe_data, "disguised_exe.doc")

        assert result.is_valid is False
        assert result.error_code == "MAGIC_BYTE_MISMATCH"

    @pytest.mark.parametrize("ext", list(MAGIC_SIGNATURES.keys()))
    def test_garbage_bytes_rejected_for_all_magic_signatures(self, ext: str):
        """Arbitrary garbage bytes must fail validation for every extension in MAGIC_SIGNATURES."""
        validator = FileValidator(strict_mode=True)
        garbage_data = b"INVALID_MAGIC_HEADER_GARBAGE_BYTES_1234567890"
        filename = f"corrupt_file{ext}"
        result = validator.validate(garbage_data, filename)

        assert result.is_valid is False, f"Expected failure for {ext} with garbage bytes"
        assert result.error_code == "MAGIC_BYTE_MISMATCH"


class TestMagicSignaturesCompleteness:
    """Test suite ensuring all MAGIC_SIGNATURES dictionary entries are tested."""

    def test_all_defined_magic_signatures_have_tests(self):
        """Verify that all keys in MAGIC_SIGNATURES are covered."""
        expected_extensions = {".pdf", ".docx", ".doc", ".rtf", ".odt", ".png", ".jpg", ".jpeg", ".epub"}
        assert expected_extensions.issubset(set(MAGIC_SIGNATURES.keys()))

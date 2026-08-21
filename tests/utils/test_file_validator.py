"""
tests/utils/test_file_validator.py
----------------------------------
Comprehensive unit tests for the file validation utilities (Issue #2926).

Verifies that file size limits, extension checks, and magic byte validation
work correctly to prevent RAM spikes and malicious file processing.
"""

import pytest
from src.utils.file_validator import (
    FileValidator,
    ValidationResult,
    validate_upload,
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
)


class TestFileValidatorSize:
    """Test suite for file size validation."""

    def test_valid_size_passes(self):
        """Verify files under the size limit pass validation."""
        validator = FileValidator(max_size_bytes=1024)
        result = validator.validate(b"x" * 500, "test.txt")
        assert result.is_valid is True

    def test_exact_limit_passes(self):
        """Verify files exactly at the size limit pass validation."""
        validator = FileValidator(max_size_bytes=1024)
        result = validator.validate(b"x" * 1024, "test.txt")
        assert result.is_valid is True

    def test_exceeds_limit_fails(self):
        """Verify files over the size limit fail with FILE_TOO_LARGE."""
        validator = FileValidator(max_size_bytes=1024)
        result = validator.validate(b"x" * 1025, "large.pdf")

        assert result.is_valid is False
        assert result.error_code == "FILE_TOO_LARGE"
        assert "too large" in result.error_message.lower()

    def test_empty_file_fails(self):
        """Verify empty files (0 bytes) fail with FILE_EMPTY."""
        validator = FileValidator()
        result = validator.validate(b"", "empty.txt")

        assert result.is_valid is False
        assert result.error_code == "FILE_EMPTY"

    def test_default_limit_is_50mb(self):
        """Verify the default validator uses the 50MB limit."""
        assert MAX_FILE_SIZE_BYTES == 50 * 1024 * 1024


class TestFileValidatorExtension:
    """Test suite for file extension validation."""

    def test_valid_extension_passes(self):
        """Verify allowed extensions pass validation."""
        validator = FileValidator()
        for ext in [".pdf", ".docx", ".txt", ".md"]:
            result = validator.validate(b"content", f"file{ext}")
            assert result.is_valid is True, f"Failed for {ext}"

    def test_invalid_extension_fails(self):
        """Verify unsupported extensions fail with UNSUPPORTED_EXTENSION."""
        validator = FileValidator()
        result = validator.validate(b"content", "malware.exe")

        assert result.is_valid is False
        assert result.error_code == "UNSUPPORTED_EXTENSION"

    def test_missing_extension_fails(self):
        """Verify files with no extension fail with MISSING_EXTENSION."""
        validator = FileValidator()
        result = validator.validate(b"content", "noextension")

        assert result.is_valid is False
        assert result.error_code == "MISSING_EXTENSION"

    def test_case_insensitive_extension(self):
        """Verify extension matching is case-insensitive."""
        validator = FileValidator()
        result = validator.validate(b"%PDF-1.4", "DOCUMENT.PDF")
        assert result.is_valid is True

    def test_custom_allowed_extensions(self):
        """Verify custom allowed_extensions set is respected."""
        validator = FileValidator(allowed_extensions={".custom"})

        assert validator.validate(b"data", "file.custom").is_valid is True
        assert validator.validate(b"data", "file.txt").is_valid is False


class TestFileValidatorMagicBytes:
    """Test suite for magic byte (content) validation."""

    def test_valid_pdf_magic_bytes(self):
        """Verify PDF magic bytes (%PDF) are recognized."""
        validator = FileValidator()
        result = validator.validate(b"%PDF-1.4\n%fake content", "test.pdf")
        assert result.is_valid is True

    def test_valid_docx_magic_bytes(self):
        """Verify DOCX magic bytes (PK\x03\x04) are recognized."""
        validator = FileValidator()
        result = validator.validate(b"PK\x03\x04\x14\x00\x06\x00", "test.docx")
        assert result.is_valid is True

    def test_mismatched_magic_bytes_logs_warning(self, caplog):
        """Verify mismatched magic bytes log a warning but don't fail hard."""
        import logging

        validator = FileValidator()

        # Pass a text file disguised as a PDF
        with caplog.at_level(logging.WARNING):
            result = validator.validate(b"This is plain text", "fake.pdf")

        # Currently configured to pass but log warning
        assert result.is_valid is True
        assert any("Magic byte mismatch" in record.message for record in caplog.records)

    def test_txt_files_skip_magic_byte_check(self):
        """Verify plain text files skip magic byte validation."""
        validator = FileValidator()
        result = validator.validate(b"Just some plain text content.", "readme.txt")
        assert result.is_valid is True


class TestValidateUploadConvenience:
    """Test suite for the validate_upload convenience function."""

    def test_validate_upload_uses_default_validator(self):
        """Verify validate_upload uses the global default validator."""
        # Valid file
        result = validate_upload(b"%PDF-1.4", "test.pdf")
        assert result.is_valid is True

        # Invalid file (too large)
        large_data = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        result = validate_upload(large_data, "huge.pdf")
        assert result.is_valid is False
        assert result.error_code == "FILE_TOO_LARGE"

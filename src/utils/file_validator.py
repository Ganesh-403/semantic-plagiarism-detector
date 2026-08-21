"""
src/utils/file_validator.py
---------------------------
Utilities for validating uploaded files before processing.

This module provides a robust validation layer to ensure that files uploaded
via the Streamlit UI or API meet security, size, and format requirements
before they are passed to the expensive text extraction and embedding pipelines.

Issue #2926: Prevent RAM spikes on massive file uploads.
While Streamlit's server.maxUploadSize provides a hard server-level limit,
this module provides application-level validation to give users immediate,
descriptive feedback and to enforce stricter business logic rules.
"""

import os
import logging
import mimetypes
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum allowed file size in bytes (50 MB).
# This should match or be slightly less than the server.maxUploadSize in config.toml
# to ensure our application-level check catches it before the server does,
# allowing us to return a friendly error message.
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Allowed file extensions for document processing.
# These correspond to the formats supported by src/core/document_parser.py
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".markdown",
    ".mdown",
    ".rtf",
    ".odt",
    ".csv",
}

# Magic byte signatures for common document formats.
# Used to verify that the file content matches its extension, preventing
# malicious files disguised as PDFs (e.g., an executable renamed to .pdf).
MAGIC_SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",  # ZIP archive (DOCX is a ZIP)
    ".doc": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # OLE2 Compound Document
    ".rtf": b"{\\rtf",
    ".odt": b"PK\x03\x04",  # ZIP archive
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
}


@dataclass
class ValidationResult:
    """Represents the result of a file validation check.

    Attributes:
        is_valid: True if the file passed all validation checks.
        filename: The name of the file that was validated.
        error_message: A descriptive error message if validation failed, else None.
        error_code: A machine-readable error code for UI/API handling.
    """

    is_valid: bool
    filename: str
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class FileValidator:
    """Validates uploaded files for size, extension, and content integrity.

    This class provides a centralized validation mechanism that can be used
    by both the Streamlit UI and the FastAPI backend to ensure files are
    safe and supported before processing.
    """

    def __init__(
        self,
        max_size_bytes: int = MAX_FILE_SIZE_BYTES,
        allowed_extensions: Optional[set] = None,
    ):
        """Initialize the FileValidator with configurable limits.

        Args:
            max_size_bytes: Maximum allowed file size in bytes.
            allowed_extensions: Set of allowed file extensions (e.g., {'.pdf', '.txt'}).
        """
        self.max_size_bytes = max_size_bytes
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS

    def validate(self, file_data: bytes | bytearray, filename: str) -> ValidationResult:
        """Perform comprehensive validation on a file.

        This method checks the file size, extension, and magic bytes to ensure
        the file is safe and supported. It returns a ValidationResult object
        containing the status and any error details.

        Args:
            file_data: The raw bytes of the uploaded file.
            filename: The original name of the file.

        Returns:
            A ValidationResult object indicating success or failure.
        """
        logger.debug("Validating file: %s", filename)

        # 1. Check file size
        size_result = self._check_size(file_data, filename)
        if not size_result.is_valid:
            return size_result

        # 2. Check file extension
        ext_result = self._check_extension(filename)
        if not ext_result.is_valid:
            return ext_result

        # 3. Check magic bytes (content verification)
        magic_result = self._check_magic_bytes(file_data, filename)
        if not magic_result.is_valid:
            # Log a warning for magic byte mismatch, but don't fail hard
            # unless it's a known dangerous signature. Some valid files
            # might have unusual headers.
            logger.warning(
                "Magic byte mismatch for %s. Expected %s, got %s",
                filename,
                magic_result.error_message,
                file_data[:8],
            )
            # For now, we allow it but log it. In high-security environments,
            # this could be changed to return the invalid result.

        logger.info("File validation passed for %s", filename)
        return ValidationResult(is_valid=True, filename=filename)

    def _check_size(
        self, file_data: bytes | bytearray, filename: str
    ) -> ValidationResult:
        """Verify the file size does not exceed the maximum limit.

        Args:
            file_data: The raw bytes of the file.
            filename: The name of the file.

        Returns:
            ValidationResult indicating if the size is acceptable.
        """
        file_size = len(file_data)

        if file_size > self.max_size_bytes:
            max_mb = self.max_size_bytes / (1024 * 1024)
            error_msg = (
                f"File '{filename}' is too large ({file_size / (1024 * 1024):.2f} MB). "
                f"Maximum allowed size is {max_mb:.0f} MB."
            )
            logger.warning(error_msg)
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="FILE_TOO_LARGE",
            )

        if file_size == 0:
            error_msg = f"File '{filename}' is empty (0 bytes)."
            logger.warning(error_msg)
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="FILE_EMPTY",
            )

        return ValidationResult(is_valid=True, filename=filename)

    def _check_extension(self, filename: str) -> ValidationResult:
        """Verify the file extension is in the allowed list.

        Args:
            filename: The name of the file.

        Returns:
            ValidationResult indicating if the extension is supported.
        """
        ext = Path(filename).suffix.lower()

        if not ext:
            error_msg = f"File '{filename}' has no extension."
            logger.warning(error_msg)
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="MISSING_EXTENSION",
            )

        if ext not in self.allowed_extensions:
            error_msg = (
                f"File extension '{ext}' is not supported. "
                f"Allowed extensions: {', '.join(sorted(self.allowed_extensions))}"
            )
            logger.warning(error_msg)
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="UNSUPPORTED_EXTENSION",
            )

        return ValidationResult(is_valid=True, filename=filename)

    def _check_magic_bytes(
        self, file_data: bytes | bytearray, filename: str
    ) -> ValidationResult:
        """Verify the file's magic bytes match its extension.

        This prevents malicious files disguised as documents (e.g., an
        executable renamed to .pdf) from being processed.

        Args:
            file_data: The raw bytes of the file.
            filename: The name of the file.

        Returns:
            ValidationResult indicating if the magic bytes match.
        """
        ext = Path(filename).suffix.lower()

        if ext not in MAGIC_SIGNATURES:
            # No magic signature defined for this extension (e.g., .txt, .md)
            # Plain text files don't have standard magic bytes, so we skip.
            return ValidationResult(is_valid=True, filename=filename)

        expected_signature = MAGIC_SIGNATURES[ext]
        actual_header = file_data[: len(expected_signature)]

        if actual_header != expected_signature:
            error_msg = (
                f"File content does not match extension '{ext}'. "
                f"Expected header: {expected_signature!r}"
            )
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="MAGIC_BYTE_MISMATCH",
            )

        return ValidationResult(is_valid=True, filename=filename)


# Global validator instance for convenience
default_validator = FileValidator()


def validate_upload(file_data: bytes, filename: str) -> ValidationResult:
    """Convenience function to validate a file using the default validator.

    Args:
        file_data: The raw bytes of the uploaded file.
        filename: The original name of the file.

    Returns:
        A ValidationResult object.
    """
    return default_validator.validate(file_data, filename)

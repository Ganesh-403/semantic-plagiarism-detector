# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Maximum allowed file size in bytes (50 MB).
# This should match or be slightly less than the server.maxUploadSize in config.toml
# to ensure our application-level check catches it before the server does,
# allowing us to return a friendly error message.
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024

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
    ".epub",
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
    ".epub": b"PK\x03\x04",  # ZIP archive (EPUB is a ZIP)
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

    By default a magic-byte mismatch is logged but tolerated. Pass
    ``strict_mode=True`` to make such mismatches a hard failure with
    ``error_code="MAGIC_BYTE_MISMATCH"`` (Issue #3201).
    """

    def __init__(
        self,
        max_size_bytes: int = MAX_FILE_SIZE_BYTES,
        allowed_extensions: Optional[set] = None,
        strict_mode: bool = False,
    ):
        """Initialize the FileValidator with configurable limits.

        Args:
            max_size_bytes: Maximum allowed file size in bytes.
            allowed_extensions: Set of allowed file extensions (e.g., {'.pdf', '.txt'}).
            strict_mode: When True, a magic-byte/extension mismatch fails
                validation with ``MAGIC_BYTE_MISMATCH`` instead of logging a
                warning and passing (Issue #3201). Off by default so existing
                callers keep the permissive behaviour.
        """
        self.max_size_bytes = max_size_bytes
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS
        self.strict_mode = strict_mode

    def validate(self, file_data: bytes | bytearray, filename: str) -> ValidationResult:
        """Perform comprehensive validation on a file.

        This method checks the file size, extension, and magic bytes to ensure
        the file is safe and supported. It returns a ValidationResult object
        containing the status and any error details.

        In permissive mode (the default) a magic-byte mismatch only logs a
        warning; with ``strict_mode=True`` it fails validation with
        ``MAGIC_BYTE_MISMATCH``.

        Args:
            file_data: The raw bytes of the uploaded file.
            filename: The original name of the file.

        Returns:
            A ValidationResult object indicating success or failure.
        """
        if "\x00" in filename:
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message="Filename contains invalid characters.",
                error_code="INVALID_FILENAME_CHARACTERS",
            )

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
            # In strict mode a mismatch is a hard failure: an executable
            # renamed to .pdf must not reach the processing pipeline
            # (Issue #3201).
            if self.strict_mode:
                logger.error(
                    "Magic byte mismatch for %s (strict mode). Expected %s, got %s",
                    filename,
                    magic_result.error_message,
                    file_data[:8],
                )
                return magic_result

            # Permissive default: log the mismatch and let the file through,
            # because some valid files carry unusual headers.
            logger.warning(
                "Magic byte mismatch for %s. Expected %s, got %s",
                filename,
                magic_result.error_message,
                file_data[:8],
            )

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
                f"File '{filename}' is too large ({file_size / (1024*1024):.2f} MB). "
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

        Also detects "double extension" disguise attempts (e.g.
        ``thesis.pdf.exe`` or ``report.docx.vbs``), where an earlier part of
        the filename looks like a legitimate document extension but the
        actual (final) extension is something else — a common technique for
        tricking users and naive server-side checks into treating a
        dangerous file as safe.

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

        suffixes = [s.lower() for s in Path(filename).suffixes]
        if len(suffixes) > 1:
            final_suffix = suffixes[-1]
            non_final_suffixes = suffixes[:-1]
            if final_suffix not in self.allowed_extensions and any(
                s in self.allowed_extensions for s in non_final_suffixes
            ):
                disguised_as = next(
                    s for s in non_final_suffixes if s in self.allowed_extensions
                )
                error_msg = (
                    f"File '{filename}' has a suspicious double extension: "
                    f"it looks like a '{disguised_as}' document but the actual "
                    f"file extension is '{final_suffix}', which is not supported. "
                    f"This pattern (e.g. 'report.docx.vbs') is often used to "
                    f"disguise malicious files — the file was rejected."
                )
                logger.warning(error_msg)
                return ValidationResult(
                    is_valid=False,
                    filename=filename,
                    error_message=error_msg,
                    error_code="DOUBLE_EXTENSION",
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

    def _check_epub_mimetype(
        self, file_data: bytes | bytearray, filename: str
    ) -> ValidationResult:
        """Verify that an EPUB file contains a valid EPUB mimetype entry."""
        import io
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(file_data)) as z:
                names = [name.lower() for name in z.namelist()]
                if "mimetype" in names:
                    target = next(n for n in z.namelist() if n.lower() == "mimetype")
                    mimetype_content = (
                        z.read(target).decode("utf-8", errors="ignore").strip().lower()
                    )
                    if (
                        "application/epub+zip" in mimetype_content
                        or "epub" in mimetype_content
                    ):
                        return ValidationResult(is_valid=True, filename=filename)
        except Exception:
            pass

        # Fallback byte pattern scan for mimetype in EPUB container
        if b"application/epub+zip" in file_data or b"mimetype" in file_data:
            return ValidationResult(is_valid=True, filename=filename)

        error_msg = f"File '{filename}' content is missing EPUB mimetype declaration."
        return ValidationResult(
            is_valid=False,
            filename=filename,
            error_message=error_msg,
            error_code="MAGIC_BYTE_MISMATCH",
        )

    def _check_csv_content(
        self, file_data: bytes | bytearray, filename: str
    ) -> ValidationResult:
        """Verify CSV content is valid UTF-8/ASCII text without binary null bytes."""
        if b"\x00" in file_data:
            error_msg = f"CSV file '{filename}' contains binary null bytes."
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="MAGIC_BYTE_MISMATCH",
            )

        try:
            file_data.decode("utf-8")
        except UnicodeDecodeError:
            error_msg = f"CSV file '{filename}' is not valid UTF-8/ASCII text."
            return ValidationResult(
                is_valid=False,
                filename=filename,
                error_message=error_msg,
                error_code="MAGIC_BYTE_MISMATCH",
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

        if ext == ".csv":
            return self._check_csv_content(file_data, filename)

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

        if ext == ".epub":
            return self._check_epub_mimetype(file_data, filename)

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

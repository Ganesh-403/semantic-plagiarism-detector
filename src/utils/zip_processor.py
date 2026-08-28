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
src/utils/zip_processor.py
--------------------------
In-memory extraction of supported documents from uploaded ZIP archives.

Two entry points are exposed:

``iter_zip_files``
    A generator yielding ``(filename, file_bytes)`` one entry at a time so a
    large archive never has every decompressed member resident at once.

``process_zip_file``
    A thin convenience wrapper that materialises the generator into a dict for
    callers that want the whole archive up front.
"""

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Generator, Tuple

from src.utils.filename import (
    InvalidFileExtensionError,
    unique_filename,
    validate_document_extension,
)

logger = logging.getLogger(__name__)

# Safety limits for ZIP bomb protection
MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_SINGLE_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
# Issue #1364: Structural zip bomb protection via decompression ratio
MAX_DECOMPRESSION_RATIO = 100  # 100:1 ratio limit
MAX_ABSOLUTE_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500 MB absolute limit

ALLOWED_ZIP_MEMBER_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".rtf",
    ".csv",
    ".odt",
    ".md",
}


def is_safe_zip_path(target_dir: Path, extracted_path: Path) -> bool:
    """
    Validates that extracting the given path into target_dir is safe.
    Prevents path traversal (Zip Slip) attacks.
    """
    resolved_target = target_dir.resolve()
    resolved_extracted = (target_dir / extracted_path).resolve()

    if not resolved_extracted.is_relative_to(resolved_target):
        raise ValueError("Malicious path traversal detected in ZIP archive entry")

    return True


def iter_zip_files(
    zip_bytes: bytes, skip_corrupted: bool = False
) -> Generator[Tuple[str, bytes], None, None]:
    """
    Yields (filename, file_bytes) tuples one entry at a time from a ZIP archive.

    Memory Optimization (Issue #3197):
    Instead of reading all decompressed file entry bytes into memory simultaneously,
    this generator processes and yields each entry sequentially to minimize peak RAM usage.

    Args:
        zip_bytes: The raw binary data of the ZIP archive.
        skip_corrupted: When True, encrypted and unreadable members are logged
            and skipped instead of aborting the whole archive.

    Yields:
        Tuple[str, bytes]: (sanitized_unique_filename, raw_file_bytes)

    Raises:
        ValueError: The archive is empty, malformed, exceeds a decompression
            safety limit, contains a path-traversal entry, or (when
            ``skip_corrupted`` is False) holds an encrypted or unreadable member.
    """
    if not zip_bytes:
        raise ValueError("ZIP archive is empty.")

    # Fast-fail if the standard ZIP magic signature is missing
    magic = zip_bytes[:4]
    if magic not in (b"PK\x03\x04", b"PK\x05\x06"):
        raise ValueError(
            "Invalid or corrupted ZIP archive: missing ZIP header signature."
        )

    # Tracks the names already yielded so colliding basenames from different
    # archive directories are disambiguated rather than silently overwritten.
    used_filenames: set[str] = set()

    try:
        zip_stream = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_stream) as zf:
            # 1. ZIP Bomb Protection: Check sizes and ratios of all entries before reading
            total_size = 0
            for zip_info in zf.infolist():
                if zip_info.is_dir():
                    continue

                uncompressed_size = zip_info.file_size
                compressed_size = zip_info.compress_size

                # Issue #1364: Check absolute uncompressed size limit (500 MB)
                if uncompressed_size > MAX_ABSOLUTE_UNCOMPRESSED_SIZE:
                    raise ValueError(
                        "Decompression ratio exceeds security limit (Zip Bomb detected)"
                    )

                # Issue #1364: Check decompression ratio (100:1)
                # Only check ratio when compressed_size > 0 to avoid division by zero
                # (stored/uncompressed entries have compress_size == file_size)
                if compressed_size > 0:
                    ratio = uncompressed_size / compressed_size
                    if ratio > MAX_DECOMPRESSION_RATIO:
                        raise ValueError(
                            "Decompression ratio exceeds security limit (Zip Bomb detected)"
                        )

                if uncompressed_size > MAX_SINGLE_FILE_SIZE:
                    raise ValueError(
                        f"Entry '{zip_info.filename}' exceeds single file decompression safety limit of {MAX_SINGLE_FILE_SIZE // (1024 * 1024)}MB."
                    )
                total_size += uncompressed_size
                if total_size > MAX_TOTAL_DECOMPRESSED_SIZE:
                    raise ValueError(
                        f"ZIP archive total decompressed size exceeds safety limit of {MAX_TOTAL_DECOMPRESSED_SIZE // (1024 * 1024)}MB."
                    )

            # 2. Extract and yield entries sequentially
            for zip_info in zf.infolist():
                # Skip directories
                if zip_info.is_dir():
                    continue

                # Check for encryption
                if zip_info.flag_bits & 0x1:
                    if skip_corrupted:
                        logger.warning(
                            f"Skipping encrypted ZIP entry '{zip_info.filename}'"
                        )
                        continue
                    else:
                        raise ValueError(
                            "Password-protected or encrypted ZIP files are not supported."
                        )

                # Normalize filename slashes (Windows to Unix format)
                filename = zip_info.filename.replace("\\", "/")

                # Path Traversal Protection: Validate the target path
                dummy_target = Path("/safe_extract_root")
                is_safe_zip_path(dummy_target, Path(filename))

                # Filter by supported document extensions
                _, ext = os.path.splitext(filename)
                ext = ext.lower()
                if ext not in ALLOWED_ZIP_MEMBER_EXTENSIONS:
                    continue

                # Read entry bytes
                try:
                    file_data = zf.read(zip_info)
                except (zipfile.BadZipFile, RuntimeError) as e:
                    if skip_corrupted:
                        logger.warning(
                            f"Skipping corrupted or protected ZIP entry '{zip_info.filename}': {e}"
                        )
                        continue
                    else:
                        raise ValueError(
                            f"Corrupted or protected entry: {zip_info.filename}"
                        ) from e

                # Skip empty files
                if not file_data:
                    continue

                try:
                    validate_document_extension(
                        filename,
                        allowed_extensions=ALLOWED_ZIP_MEMBER_EXTENSIONS,
                    )
                except InvalidFileExtensionError:
                    # Unsafe or unsupported archive members are ignored.
                    continue

                # Keep only the entry basename and sanitize it as untrusted
                # input after strict final-extension validation.
                unique_name = unique_filename(
                    filename,
                    used_filenames,
                )
                used_filenames.add(unique_name)

                yield unique_name, file_data

    except zipfile.BadZipFile as e:
        raise ValueError("Invalid or corrupted ZIP archive.") from e


def process_zip_file(
    zip_bytes: bytes, skip_corrupted: bool = False
) -> dict[str, bytes]:
    """
    Extracts supported documents (.pdf, .docx, .txt, .rtf, .csv, .odt, .md) from a ZIP archive entirely in memory.

    Args:
        zip_bytes: The raw binary data of the ZIP archive.
        skip_corrupted: Forwarded to :func:`iter_zip_files`. When True,
            encrypted and unreadable members are skipped instead of aborting.

    Returns:
        Dict[str, bytes]: A dictionary mapping unique, sanitized filenames to their raw bytes.
    """
    return dict(iter_zip_files(zip_bytes, skip_corrupted=skip_corrupted))

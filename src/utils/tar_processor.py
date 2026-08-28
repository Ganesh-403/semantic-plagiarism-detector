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
In-memory extraction of supported documents from TAR archives.

Supports gzip-compressed (.tar.gz/.tgz) and bzip2-compressed
(.tar.bz2/.tbz2) archives while applying the same safety guarantees as the
ZIP processor: member-size limits, total decompressed-size limits,
decompression-ratio checks, path-traversal protection, and strict document
extension filtering.
"""

from __future__ import annotations

import io
import logging
import os
import posixpath
import tarfile
from pathlib import Path
from typing import Generator, Tuple

from src.utils.filename import (
    InvalidFileExtensionError,
    unique_filename,
    validate_document_extension,
)

logger = logging.getLogger(__name__)

# Keep archive extraction limits aligned with zip_processor.py.
MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024
MAX_SINGLE_FILE_SIZE = 100 * 1024 * 1024
MAX_DECOMPRESSION_RATIO = 100
MAX_ABSOLUTE_UNCOMPRESSED_SIZE = 500 * 1024 * 1024

ALLOWED_TAR_MEMBER_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".rtf",
    ".csv",
    ".odt",
    ".md",
}

_SUPPORTED_TAR_MODES = ("r:gz", "r:bz2")


def is_safe_tar_path(filename: str) -> bool:
    """Reject absolute paths and normalized paths that escape the archive root."""
    normalized = filename.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError("Malicious path traversal detected in TAR archive entry")

    normalized_path = posixpath.normpath(normalized)
    if normalized_path == ".." or normalized_path.startswith("../"):
        raise ValueError("Malicious path traversal detected in TAR archive entry")

    # Windows drive-qualified paths must never be treated as relative names.
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError("Malicious path traversal detected in TAR archive entry")

    # Resolve against a synthetic root as an additional defense-in-depth check.
    root = Path("/safe_extract_root").resolve()
    candidate = (root / Path(normalized_path)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Malicious path traversal detected in TAR archive entry")

    return True


def _open_tar_archive(tar_bytes: bytes) -> tarfile.TarFile:
    """Open a supported compressed TAR stream and normalize archive errors."""
    if tar_bytes.startswith(b"\x1f\x8b"):
        mode = "r:gz"
    elif tar_bytes.startswith(b"BZh"):
        mode = "r:bz2"
    else:
        raise ValueError(
            "Unsupported TAR compression. Only .tar.gz and .tar.bz2 archives are supported."
        )

    try:
        return tarfile.open(fileobj=io.BytesIO(tar_bytes), mode=mode)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("Invalid or corrupted TAR archive.") from exc


def iter_tar_files(
    tar_bytes: bytes, skip_corrupted: bool = False
) -> Generator[Tuple[str, bytes], None, None]:
    """
    Yield supported document members from a gzip/bzip2 TAR archive.

    Extraction is performed one member at a time so callers can avoid keeping
    every decompressed member resident simultaneously.
    """
    if not tar_bytes:
        raise ValueError("TAR archive is empty.")

    used_filenames: set[str] = set()

    try:
        archive = _open_tar_archive(tar_bytes)
        with archive:
            members = archive.getmembers()

            total_size = 0
            archive_size = max(len(tar_bytes), 1)

            # Validate every regular member before extracting any bytes.
            for member in members:
                # Validate every archive name, including links and metadata,
                # before deciding whether the member is extractable.
                is_safe_tar_path(member.name)

                if not member.isfile():
                    continue

                uncompressed_size = member.size

                if uncompressed_size > MAX_ABSOLUTE_UNCOMPRESSED_SIZE:
                    raise ValueError(
                        "Decompression ratio exceeds security limit (Tar Bomb detected)"
                    )

                if uncompressed_size > MAX_SINGLE_FILE_SIZE:
                    raise ValueError(
                        f"Entry '{member.name}' exceeds single file decompression "
                        f"safety limit of {MAX_SINGLE_FILE_SIZE // (1024 * 1024)}MB."
                    )

                total_size += uncompressed_size
                if total_size > MAX_TOTAL_DECOMPRESSED_SIZE:
                    raise ValueError(
                        "TAR archive total decompressed size exceeds safety limit of "
                        f"{MAX_TOTAL_DECOMPRESSED_SIZE // (1024 * 1024)}MB."
                    )

            # TAR has no per-member compressed-size field. Compare the declared
            # total expansion against the uploaded compressed archive size to
            # detect highly compressed archive bombs.
            if total_size > archive_size * MAX_DECOMPRESSION_RATIO:
                raise ValueError(
                    "Decompression ratio exceeds security limit (Tar Bomb detected)"
                )

            for member in members:
                if not member.isfile():
                    # Symlinks, hardlinks, devices, FIFOs, and directories are
                    # deliberately never materialized from an untrusted archive.
                    continue

                filename = member.name.replace("\\", "/")
                _, ext = os.path.splitext(filename)
                ext = ext.lower()
                if ext not in ALLOWED_TAR_MEMBER_EXTENSIONS:
                    continue

                try:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise tarfile.TarError("member has no readable file stream")
                    file_data = extracted.read(MAX_SINGLE_FILE_SIZE + 1)
                except (OSError, tarfile.TarError) as exc:
                    if skip_corrupted:
                        logger.warning(
                            "Skipping corrupted TAR entry '%s': %s", member.name, exc
                        )
                        continue
                    raise ValueError(
                        f"Corrupted or unreadable TAR entry: {member.name}"
                    ) from exc

                if len(file_data) > MAX_SINGLE_FILE_SIZE:
                    raise ValueError(
                        f"Entry '{member.name}' exceeds single file decompression "
                        f"safety limit of {MAX_SINGLE_FILE_SIZE // (1024 * 1024)}MB."
                    )

                if not file_data:
                    continue

                try:
                    validate_document_extension(
                        filename,
                        allowed_extensions=ALLOWED_TAR_MEMBER_EXTENSIONS,
                    )
                except InvalidFileExtensionError:
                    continue

                unique_name = unique_filename(filename, used_filenames)
                used_filenames.add(unique_name)
                yield unique_name, file_data

    except tarfile.TarError as exc:
        raise ValueError("Invalid or corrupted TAR archive.") from exc


def process_tar_file(
    tar_bytes: bytes, skip_corrupted: bool = False
) -> dict[str, bytes]:
    """
    Extract supported documents from a .tar.gz or .tar.bz2 archive.

    Returns a mapping of sanitized, unique filenames to their raw bytes.
    """
    return dict(iter_tar_files(tar_bytes, skip_corrupted=skip_corrupted))

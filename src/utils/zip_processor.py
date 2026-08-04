import io
import os
import zipfile
from pathlib import Path
from typing import Dict

from src.utils.filename import (
    InvalidFileExtensionError,
    unique_filename,
    validate_document_extension,
)

# Safety limits for ZIP bomb protection
MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_SINGLE_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


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


def process_zip_file(zip_bytes: bytes) -> Dict[str, bytes]:
    """
    Extracts supported documents (PDF, DOCX, TXT) from a ZIP archive entirely in memory.

    Handles:
    - Invalid or corrupted ZIP files (raises ValueError).
    - Encrypted ZIP entries (raises ValueError).
    - Subdirectory filtering (ignores folder entries, flattens paths to keep filenames safe).
    - Path traversal attempts (skips any entries containing ".." or starting with "/").
    - ZIP bombs (checks total/individual decompressed size limits before extraction).
    - Duplicate filename collisions (resolves them uniquely).

    Args:
        zip_bytes: The raw binary data of the ZIP archive.

    Returns:
        Dict[str, bytes]: A dictionary mapping unique, sanitized filenames to their raw bytes.
    """
    if not zip_bytes:
        raise ValueError("ZIP archive is empty.")

    extracted_files: Dict[str, bytes] = {}

    try:
        zip_stream = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_stream) as zf:
            # 1. ZIP Bomb Protection: Check sizes of all entries before reading
            total_size = 0
            for zip_info in zf.infolist():
                if zip_info.file_size > MAX_SINGLE_FILE_SIZE:
                    raise ValueError(
                        f"Entry '{zip_info.filename}' exceeds single file decompression safety limit of {MAX_SINGLE_FILE_SIZE // (1024 * 1024)}MB."
                    )
                total_size += zip_info.file_size
                if total_size > MAX_TOTAL_DECOMPRESSED_SIZE:
                    raise ValueError(
                        f"ZIP archive total decompressed size exceeds safety limit of {MAX_TOTAL_DECOMPRESSED_SIZE // (1024 * 1024)}MB."
                    )

            # 2. Extract and sanitize entries
            for zip_info in zf.infolist():
                # Skip directories
                if zip_info.is_dir():
                    continue

                # Check for encryption
                if zip_info.flag_bits & 0x1:
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
                if ext not in [".pdf", ".docx", ".txt"]:
                    continue

                # Read entry bytes
                try:
                    file_data = zf.read(zip_info)
                except (zipfile.BadZipFile, RuntimeError) as e:
                    raise ValueError(
                        f"Corrupted or protected entry: {zip_info.filename}"
                    ) from e

                # Skip empty files
                if not file_data:
                    continue

                try:
                    validate_document_extension(
                        filename,
                        allowed_extensions={
                            ".csv",
                            ".docx",
                            ".pdf",
                            ".rtf",
                            ".txt",
                        },
                    )
                except InvalidFileExtensionError:
                    # Unsafe or unsupported archive members are ignored.
                    continue

                # Keep only the entry basename and sanitize it as untrusted
                # input after strict final-extension validation.
                unique_name = unique_filename(
                    filename,
                    extracted_files,
                )
                extracted_files[unique_name] = file_data

    except zipfile.BadZipFile as e:
        raise ValueError("Invalid or corrupted ZIP archive.") from e

    return extracted_files

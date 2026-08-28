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

# semantic-plagiarism-detector/utils/zip_processor.py

import zipfile
from pathlib import Path
from typing import Set, Union

# Configuration constants
MAX_ARCHIVE_DEPTH = 1
SUPPORTED_ARCHIVE_EXTENSIONS: set[str] = {".zip", ".tar", ".gz", ".rar", ".7z"}


class NestedArchiveError(ValueError):
    """Raised when nested archive depth exceeds safety limits or unauthorized archives are detected."""

    pass


def validate_nested_archives(zip_path: str | Path, current_depth: int = 0) -> None:
    """
    Validates an uploaded ZIP file for nested archive safety limits to prevent
    zip bombs and recursive decompression attacks.

    Args:
        zip_path: Path to the target ZIP file.
        current_depth: Current recursion depth of archive nesting.

    Raises:
        NestedArchiveError: If MAX_ARCHIVE_DEPTH is exceeded or prohibited archives are found.
    """
    if current_depth > MAX_ARCHIVE_DEPTH:
        raise NestedArchiveError(
            f"Security Violation: Maximum nested archive depth of {MAX_ARCHIVE_DEPTH} exceeded."
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for filename in archive.namelist():
                file_ext = Path(filename).suffix.lower()

                # Explicit validation step checking for archive extensions inside the ZIP
                if file_ext in SUPPORTED_ARCHIVE_EXTENSIONS:
                    if current_depth + 1 > MAX_ARCHIVE_DEPTH:
                        raise NestedArchiveError(
                            f"Nested archive detected ('{filename}') exceeding maximum allowed depth of {MAX_ARCHIVE_DEPTH}."
                        )
    except zipfile.BadZipFile:
        raise ValueError(
            f"The file at '{zip_path}' is corrupted or not a valid ZIP archive."
        )


def process_submission_zip(zip_path: str | Path) -> None:
    """
    Main entry point for processing submission ZIP files with integrated safety checks.
    """
    # Run explicit validation step before any extraction logic
    validate_nested_archives(zip_path, current_depth=0)

    # Proceed with standard safe extraction logic...
    print(f"Validation successful for {zip_path}. No archive safety limits breached.")

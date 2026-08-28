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
tests/utils/__init__.py
-----------------------
Test utilities for the utils package.

Includes PDF comparison functionality for golden fixture testing.
"""

import hashlib
from pathlib import Path
from typing import Optional

# Directory containing golden fixture files
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def compare_pdf_bytes(
    generated_bytes: bytes,
    golden_path: Path,
    tolerance: float = 0.0,
) -> tuple[bool, Optional[str]]:
    """
    Compare a generated PDF against a golden fixture.

    The comparison is deterministic by extracting and comparing the PDF text content.
    This avoids false positives from differing metadata (timestamps, file IDs, etc.).
    Timestamps and other variable metadata are stripped from comparison.

    Args:
        generated_bytes: PDF bytes to test
        golden_path: Path to the golden fixture PDF file
        tolerance: Unused parameter for API consistency (reserved for future image-based comparison)

    Returns:
        Tuple of (is_match, error_message)
        - is_match: True if PDF content matches the golden fixture
        - error_message: Description of mismatch or None if match
    """
    if not golden_path.exists():
        return (False, f"Golden fixture not found: {golden_path}")

    # Extract text from both PDFs using pypdf
    import re
    from io import BytesIO

    from pypdf import PdfReader

    def extract_text(pdf_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def normalize_text(text: str) -> str:
        """Normalize text by removing variable metadata like timestamps."""
        # Remove "Generated: YYYY-MM-DD HH:MM:SS" lines
        text = re.sub(r"Generated:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", "", text)
        # Remove "Generated: YYYY-MM-DD HH:MM:SS" with optional leading/trailing space
        text = re.sub(
            r"\s*Generated:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*", "\n", text
        )
        # Clean up multiple consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    generated_text = normalize_text(extract_text(generated_bytes))
    golden_text = normalize_text(extract_text(golden_path.read_bytes()))

    # Compare extracted text
    if generated_text == golden_text:
        return (True, None)

    # Calculate diff-like comparison for better error messages
    generated_lines = generated_text.splitlines()
    golden_lines = golden_text.splitlines()

    if len(generated_lines) != len(golden_lines):
        return (
            False,
            f"Line count mismatch: generated={len(generated_lines)}, golden={len(golden_lines)}",
        )

    # Find first differing line
    for i, (gen_line, golden_line) in enumerate(zip(generated_lines, golden_lines)):
        if gen_line != golden_line:
            return (
                False,
                f"Line {i + 1} differs:\n  Generated: {repr(gen_line)}\n  Golden:    {repr(golden_line)}",
            )

    # Fallback: compare checksums of the raw bytes
    # Use only content-related parts for deterministic comparison
    generated_hash = hashlib.md5(generated_bytes).hexdigest()
    golden_hash = hashlib.md5(golden_path.read_bytes()).hexdigest()

    return (
        False,
        f"PDF content differs (hash check):\n  Generated: {generated_hash}\n  Golden:    {golden_hash}",
    )


def assert_pdf_matches(
    generated_bytes: bytes,
    golden_path: Path,
    tolerance: float = 0.0,
) -> None:
    """
    Assert that generated PDF matches golden fixture, raising AssertionError on mismatch.

    Args:
        generated_bytes: PDF bytes to test
        golden_path: Path to the golden fixture PDF file
        tolerance: Unused parameter for API consistency (reserved for future image-based comparison)

    Raises:
        AssertionError: If PDF does not match golden fixture
    """
    is_match, error_msg = compare_pdf_bytes(generated_bytes, golden_path, tolerance)
    if not is_match:
        raise AssertionError(f"PDF mismatch: {error_msg}")

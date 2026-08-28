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
tests/test_errors.py
----------------------
Sanity checks on src/errors.py — mainly guarding against the
"# API Errors" section header being accidentally pulled back onto the
same line as SSRF_CIRCULAR_REDIRECT_LOOP's declaration (it should be its
own line, preceded by a blank line, matching every other section header
in this file).
"""

from __future__ import annotations

from pathlib import Path

ERRORS_PATH = Path("src/errors.py")


def _source_lines() -> list[str]:
    return ERRORS_PATH.read_text(encoding="utf-8").splitlines()


def test_no_trailing_section_header_comments_on_declaration_lines():
    """No error-string declaration line should carry a trailing '# ... Errors'
    section-header comment — every section header must live on its own
    line, consistent with the rest of the file."""
    lines = _source_lines()
    offending = [
        line
        for line in lines
        if "=" in line and line.rstrip().endswith("Errors") and "#" in line
    ]
    assert (
        offending == []
    ), f"Section header comment(s) trailing a declaration: {offending}"


def test_api_errors_header_is_on_its_own_line_preceded_by_blank_line():
    lines = _source_lines()
    header_indices = [
        i for i, line in enumerate(lines) if line.strip() == "# API Errors"
    ]

    assert (
        len(header_indices) == 1
    ), "Expected exactly one '# API Errors' section header"
    header_index = header_indices[0]

    # The header line itself must be nothing but the comment.
    assert lines[header_index].strip() == "# API Errors"
    # Preceded by a blank line, matching every other section header's style.
    assert lines[header_index - 1].strip() == ""
    # Immediately followed by the first API_* declaration, not a stray blank line.
    assert lines[header_index + 1].startswith("API_")


def test_ssrf_circular_redirect_loop_declaration_has_no_trailing_comment():
    lines = _source_lines()
    matches = [line for line in lines if line.startswith("SSRF_CIRCULAR_REDIRECT_LOOP")]

    assert len(matches) == 1
    assert "#" not in matches[0]
    assert matches[0].rstrip().endswith('"Circular HTTP redirect loop detected"')


def test_ssrf_circular_redirect_loop_value_unchanged():
    """The fix must only relocate the comment — the string value itself
    (used by src.security.ssrf_protector) must be untouched."""
    from src.errors import SSRF_CIRCULAR_REDIRECT_LOOP

    assert SSRF_CIRCULAR_REDIRECT_LOOP == "Circular HTTP redirect loop detected"


def test_ocr_file_batch_error_formatting():
    """Test OCRFileBatchError message formatting logic with empty and non-empty failure_details."""
    from src.exceptions import OCRFileBatchError

    # Test instantiation with failed_files but empty failure_details
    err_empty_details = OCRFileBatchError(failed_files=["a.pdf"], failure_details=[])
    assert str(err_empty_details) == "OCR extraction failed for 1 file(s): a.pdf"

    # Test instantiation with multiple details provided
    err_multiple_details = OCRFileBatchError(
        failed_files=["a.pdf", "b.pdf"],
        failure_details=["a.pdf: Timeout", "b.pdf: Corrupt image"],
    )
    assert (
        str(err_multiple_details)
        == "OCR extraction failed for 2 file(s): a.pdf: Timeout; b.pdf: Corrupt image"
    )

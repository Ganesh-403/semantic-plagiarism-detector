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

from pathlib import Path

SOURCE = Path("src/utils/file_parser.py")
TESTS = Path("tests/utils/test_file_parser.py")


def test_required_helper_signature_exists():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def validate_pdf_page_count(" in source
    assert "file_bytes: bytes" in source
    assert "max_pages: int = 500" in source
    assert ") -> int:" in source


def test_required_default_limit_error_exists():
    source = SOURCE.read_text(encoding="utf-8")

    assert '"PDF exceeds maximum allowed page limit "' in source
    assert 'f"({max_pages} pages)"' in source


def test_unit_test_and_extraction_integration_exist():
    source = SOURCE.read_text(encoding="utf-8")
    tests = TESTS.read_text(encoding="utf-8")

    assert "validate_pdf_page_count(file_bytes)" in source
    assert "class TestPDFPageCountValidation:" in tests
    assert "test_validate_pdf_page_count_rejects_over_default_limit" in tests

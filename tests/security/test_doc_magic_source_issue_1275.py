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

SOURCE = Path("src/security/mime_validator.py")
TESTS = Path("tests/security/test_mime_validator.py")


def test_complete_ole_header_is_configured():
    source = SOURCE.read_text(encoding="utf-8")
    assert r'b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"' in source


def test_doc_header_is_checked_before_mime_detection():
    source = SOURCE.read_text(encoding="utf-8")
    doc_check = source.index('if extension == "doc":')
    magic_check = source.index(
        "magic_result = _check_magic_bytes",
        doc_check,
    )
    assert doc_check < magic_check


def test_invalid_doc_header_regression_test_exists():
    source = TESTS.read_text(encoding="utf-8")
    assert "test_validate_mime_type_rejects_invalid_legacy_doc_header" in source

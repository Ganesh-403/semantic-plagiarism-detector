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

MIME_PATH = Path("src/security/mime_validator.py")


def test_ooxml_container_validation_exists():
    source = MIME_PATH.read_text(encoding="utf-8")

    assert "def _validate_ooxml_archive(" in source
    assert '"[Content_Types].xml"' in source
    assert '"word/document.xml"' in source
    assert '"xl/workbook.xml"' in source


def test_ooxml_validation_runs_before_libmagic():
    source = MIME_PATH.read_text(encoding="utf-8")

    ooxml_position = source.index("if extension in OOXML_EXTENSIONS:")
    magic_position = source.index("magic_result = _check_magic_bytes(")

    assert ooxml_position < magic_position


def test_xlsx_extension_is_registered():
    source = MIME_PATH.read_text(encoding="utf-8")

    assert '"xlsx": {' in source
    assert "spreadsheetml.sheet" in source

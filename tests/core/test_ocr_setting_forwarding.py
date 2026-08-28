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

from io import BytesIO
from unittest.mock import Mock

import src.core.document_parser as parser


def test_extract_text_forwards_ocr_settings_to_pdf_parser(monkeypatch):
    pdf_parser = Mock(return_value="recognized text")
    monkeypatch.setattr(parser, "extract_text_from_pdf", pdf_parser)

    result = parser.extract_text(
        BytesIO(b"%PDF-test"),
        "spanish-scan.pdf",
        ocr_language="spa",
        ocr_dpi=350,
    )

    assert result == "recognized text"
    pdf_parser.assert_called_once()
    _, kwargs = pdf_parser.call_args
    assert kwargs == {
        "ocr_language": "spa",
        "ocr_dpi": 350,
    }


def test_docx_and_txt_still_validate_settings(monkeypatch):
    txt_parser = Mock(return_value="plain text")
    monkeypatch.setattr(parser, "extract_text_from_txt", txt_parser)

    result = parser.extract_text(
        b"plain text",
        "notes.txt",
        ocr_language="fra",
        ocr_dpi=200,
    )

    assert result == "plain text"
    txt_parser.assert_called_once_with(b"plain text")


def test_invalid_settings_fail_before_pdf_processing(monkeypatch):
    pdf_parser = Mock()
    monkeypatch.setattr(parser, "extract_text_from_pdf", pdf_parser)

    try:
        parser.extract_text(
            b"%PDF-test",
            "scan.pdf",
            ocr_language="deu",
            ocr_dpi=250,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid language to raise ValueError")

    pdf_parser.assert_not_called()

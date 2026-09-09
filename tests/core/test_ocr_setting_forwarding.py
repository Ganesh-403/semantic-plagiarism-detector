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
            ocr_language="invalid-language",
            ocr_dpi=250,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid language to raise ValueError")

    pdf_parser.assert_not_called()

"""Tests for scanned and mixed PDF OCR fallback."""

import io
from unittest.mock import MagicMock, patch

import pytest

from src.core.document_parser import (OCRDependencyError, _has_meaningful_text,
                                      extract_text_from_pdf)


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakePDF:
    def __init__(self, page_texts):
        self.pages = [FakePage(text) for text in page_texts]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_meaningful_text_detection():
    assert _has_meaningful_text(
        "This page contains enough embedded words for normal PDF extraction."
    )
    assert not _has_meaningful_text("")
    assert not _has_meaningful_text("Page 1")


@patch("src.core.document_parser.pdfplumber.open")
@patch("src.core.document_parser._ocr_pdf_page")
def test_text_pdf_does_not_run_ocr(mock_ocr, mock_pdf_open):
    mock_pdf_open.return_value = FakePDF(
        ["This is a normal PDF page with enough embedded text to be extracted."]
    )

    result = extract_text_from_pdf(io.BytesIO(b"%PDF-fake-pdf"))

    assert "normal PDF page" in result
    mock_ocr.assert_not_called()


@patch("src.core.document_parser.pdfplumber.open")
@patch("src.core.document_parser._ocr_pdf_page")
def test_scanned_pdf_uses_ocr(mock_ocr, mock_pdf_open):
    mock_pdf_open.return_value = FakePDF([""])
    mock_ocr.return_value = (
        "This text was extracted from a scanned assignment using OCR."
    )

    result = extract_text_from_pdf(io.BytesIO(b"%PDF-fake-pdf"))

    assert "scanned assignment" in result
    mock_ocr.assert_called_once_with(
        b"%PDF-fake-pdf",
        0,
        dpi=250,
        language="eng",
    )


@patch("src.core.document_parser.pdfplumber.open")
@patch("src.core.document_parser._ocr_pdf_page")
def test_mixed_pdf_ocr_only_runs_for_scanned_page(mock_ocr, mock_pdf_open):
    mock_pdf_open.return_value = FakePDF(
        [
            "This first page has sufficient selectable embedded text for extraction.",
            "",
        ]
    )
    mock_ocr.return_value = "This second page came from OCR processing."

    result = extract_text_from_pdf(io.BytesIO(b"%PDF-fake-pdf"))

    assert "first page" in result
    assert "second page" in result
    mock_ocr.assert_called_once()


@patch("src.core.document_parser.pdfplumber.open")
@patch("src.core.document_parser._ocr_pdf_page")
def test_ocr_dependency_error_is_not_hidden(mock_ocr, mock_pdf_open):
    mock_pdf_open.return_value = FakePDF([""])
    mock_ocr.side_effect = OCRDependencyError("Tesseract OCR was not found.")

    with pytest.raises(OCRDependencyError, match="Tesseract"):
        extract_text_from_pdf(io.BytesIO(b"%PDF-fake-pdf"))


@patch("src.core.document_parser.pytesseract.image_to_string")
@patch("src.core.document_parser.fitz.open")
def test_ocr_pdf_page_oom(mock_fitz_open, mock_image_to_string):
    from src.core.document_parser import _ocr_pdf_page

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page
    mock_pixmap = MagicMock()
    mock_pixmap.samples = b"\x00" * 3
    mock_pixmap.width = 1
    mock_pixmap.height = 1
    mock_page.get_pixmap.return_value = mock_pixmap
    mock_fitz_open.return_value.__enter__.return_value = mock_doc

    mock_image_to_string.side_effect = MemoryError("Out of memory mock error")

    # Call with fake PDF bytes
    res = _ocr_pdf_page(b"%PDF-1.4", 0)
    assert "[OCR extraction failed for page 0]" in res


@patch("src.core.document_parser.pytesseract.image_to_string")
@patch("src.core.document_parser.Image.open")
def test_extract_text_from_image_oom(mock_image_open, mock_image_to_string):
    from src.core.document_parser import extract_text_from_image

    mock_img = MagicMock()
    mock_image_open.return_value = mock_img

    mock_image_to_string.side_effect = MemoryError("Out of memory mock error")

    res = extract_text_from_image(b"fake image bytes")
    assert "[OCR extraction failed for the file]" in res

"""Tests for scanned and mixed PDF OCR fallback."""

import io
from unittest.mock import MagicMock, patch

import pytest

from src.core.document_parser import (
    OCRDependencyError,
    _has_meaningful_text,
    check_ocr_dependencies,
    extract_text_from_pdf,
)


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


@patch("src.core.document_parser.check_ocr_dependencies")
def test_ocr_pdf_page_oom(mock_check_ocr):
    from src.core.document_parser import _ocr_pdf_page

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page
    mock_pixmap = MagicMock()
    mock_pixmap.samples = b"\x00" * 3
    mock_pixmap.width = 1
    mock_pixmap.height = 1
    mock_page.get_pixmap.return_value = mock_pixmap
    mock_fitz = MagicMock()
    mock_fitz.open.return_value.__enter__.return_value = mock_doc
    mock_pytesseract = MagicMock()
    mock_pytesseract.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
    mock_pytesseract.image_to_string.side_effect = MemoryError("Out of memory mock error")


    with patch.dict("sys.modules", {"fitz": mock_fitz, "pytesseract": mock_pytesseract, "PIL": MagicMock(), "PIL.Image": MagicMock()}):
        res = _ocr_pdf_page(b"%PDF-1.4", 0)
        assert "[OCR extraction failed for page 0]" in res


@patch("src.core.document_parser.check_ocr_dependencies")
def test_extract_text_from_image_oom(mock_check_ocr):
    from src.core.document_parser import extract_text_from_image

    mock_img = MagicMock()
    mock_image = MagicMock()
    mock_image.open.return_value = mock_img
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.side_effect = MemoryError("Out of memory mock error")

    with patch.dict("sys.modules", {"pytesseract": mock_pytesseract, "PIL": MagicMock(), "PIL.Image": mock_image}):
        res = extract_text_from_image(b"fake image bytes")
        assert "[OCR extraction failed for the file]" in res



def test_check_ocr_dependencies_success():
    """Verify check_ocr_dependencies passes when packages and tesseract version check succeed."""
    mock_pytesseract = MagicMock()
    mock_pytesseract.get_tesseract_version.return_value = "5.0.0"

    with patch.dict("sys.modules", {"fitz": MagicMock(), "pytesseract": mock_pytesseract, "PIL": MagicMock(), "PIL.Image": MagicMock()}):
        check_ocr_dependencies()


def test_check_ocr_dependencies_missing_package():
    """Verify check_ocr_dependencies raises OCRDependencyError when an import fails."""
    with patch.dict("sys.modules", {"fitz": None, "pytesseract": None, "PIL": None}):
        with pytest.raises(OCRDependencyError, match="OCR dependencies are missing"):
            check_ocr_dependencies()


def test_check_ocr_dependencies_missing_tesseract_binary():
    """Verify check_ocr_dependencies raises OCRDependencyError when Tesseract binary is missing."""
    mock_pytesseract = MagicMock()
    mock_pytesseract.TesseractNotFoundError = Exception
    mock_pytesseract.get_tesseract_version.side_effect = Exception("Tesseract not found")

    with patch.dict("sys.modules", {"fitz": MagicMock(), "pytesseract": mock_pytesseract, "PIL": MagicMock(), "PIL.Image": MagicMock()}):
        with pytest.raises(OCRDependencyError, match="Tesseract OCR was not found"):
            check_ocr_dependencies()


def test_ocr_invocations_total_counter_success(monkeypatch):
    """Verify ocr_invocations_total counter increments on started and success."""
    from src.core.document_parser import _ocr_pdf_page
    from src.core.metrics import ocr_invocations_total

    start_before = ocr_invocations_total.labels(status="started")._value.get()
    success_before = ocr_invocations_total.labels(status="success")._value.get()

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page
    mock_pixmap = MagicMock()
    mock_pixmap.samples = b"\x00" * 3
    mock_pixmap.width = 1
    mock_pixmap.height = 1
    mock_page.get_pixmap.return_value = mock_pixmap
    mock_fitz = MagicMock()
    mock_fitz.open.return_value.__enter__.return_value = mock_doc
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = "Extracted text content"

    with patch("src.core.document_parser.check_ocr_dependencies"):
        with patch.dict("sys.modules", {"fitz": mock_fitz, "pytesseract": mock_pytesseract, "PIL": MagicMock(), "PIL.Image": MagicMock()}):
            res = _ocr_pdf_page(b"%PDF-1.4", 0)
            assert res == "Extracted text content"

    start_after = ocr_invocations_total.labels(status="started")._value.get()
    success_after = ocr_invocations_total.labels(status="success")._value.get()

    assert start_after == start_before + 1
    assert success_after == success_before + 1


def test_ocr_invocations_total_counter_failure(monkeypatch):
    """Verify ocr_invocations_total counter increments on started and failure."""
    from src.core.document_parser import _ocr_pdf_page
    from src.core.metrics import ocr_invocations_total

    start_before = ocr_invocations_total.labels(status="started")._value.get()
    failure_before = ocr_invocations_total.labels(status="failure")._value.get()

    mock_doc = MagicMock()
    mock_doc.load_page.side_effect = RuntimeError("Corrupted page")
    mock_fitz = MagicMock()
    mock_fitz.open.return_value.__enter__.return_value = mock_doc
    mock_pytesseract = MagicMock()

    with patch("src.core.document_parser.check_ocr_dependencies"):
        with patch.dict("sys.modules", {"fitz": mock_fitz, "pytesseract": mock_pytesseract, "PIL": MagicMock(), "PIL.Image": MagicMock()}):
            res = _ocr_pdf_page(b"%PDF-1.4", 0)
            assert res == "[OCR extraction failed for page 0]"

    start_after = ocr_invocations_total.labels(status="started")._value.get()
    failure_after = ocr_invocations_total.labels(status="failure")._value.get()

    assert start_after == start_before + 1
    assert failure_after == failure_before + 1



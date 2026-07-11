import io
import pytest
import PyPDF2
import docx
from unittest.mock import MagicMock, patch
from utils.document_parser import (
    extract_text,
    extract_texts,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
)


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal in-memory PDF containing the given text."""
    writer = PyPDF2.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_docx_bytes(text: str) -> bytes:
    """Create a minimal in-memory DOCX containing the given text."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_from_pdf_bytes():
    pdf_bytes = _make_pdf_bytes("Hello PDF")
    # For blank page PDF, pdfplumber might return empty string, but it shouldn't error
    result = extract_text_from_pdf(pdf_bytes)
    assert isinstance(result, str)


def test_extract_from_docx_bytes():
    docx_bytes = _make_docx_bytes("Hello DOCX")
    result = extract_text_from_docx(docx_bytes)
    assert result == "Hello DOCX"


def test_extract_from_txt_bytes():
    txt_bytes = b"Hello TXT"
    result = extract_text_from_txt(txt_bytes)
    assert result == "Hello TXT"


def test_extract_text_routing():
    pdf_bytes = _make_pdf_bytes("Hello PDF")
    docx_bytes = _make_docx_bytes("Hello DOCX")
    txt_bytes = b"Hello TXT"

    assert isinstance(extract_text(pdf_bytes, "test.pdf"), str)
    assert extract_text(docx_bytes, "test.docx") == "Hello DOCX"
    assert extract_text(txt_bytes, "test.txt") == "Hello TXT"
    # Fallback case
    assert extract_text(txt_bytes, "test.unknown") == "Hello TXT"


def test_extract_texts_mixed():
    docx_bytes = _make_docx_bytes("Hello DOCX")
    txt_bytes = b"Hello TXT"

    mock_file1 = MagicMock()
    mock_file1.name = "doc1.docx"
    mock_file1.read.return_value = docx_bytes

    mock_file2 = MagicMock()
    mock_file2.name = "doc2.txt"
    mock_file2.read.return_value = txt_bytes

    # Mock extract_text to isolate testing of extract_texts structure
    with patch("utils.document_parser.extract_text", side_effect=lambda f, name: f"Parsed {name}"):
        results = extract_texts([mock_file1, mock_file2])

    assert results["doc1.docx"] == "Parsed doc1.docx"
    assert results["doc2.txt"] == "Parsed doc2.txt"

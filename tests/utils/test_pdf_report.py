"""Tests for src/utils/pdf_report.py PDF plagiarism report generation."""

from io import BytesIO

from PyPDF2 import PdfReader

from src.utils.pdf_report import (
    generate_plagiarism_report,
    generate_qr_code,
    get_similarity_color,
    wrap_text,
)


def _read_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_generates_valid_pdf_with_required_fields():
    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            ("First matching paragraph.", "Second matching paragraph.", 0.96),
        ],
    )
    pdf_bytes = pdf_buffer.getvalue()

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    text = _read_text(pdf_bytes)
    assert "student_a.pdf" in text
    assert "student_b.pdf" in text
    assert "93.4%" in text
    assert "First matching paragraph" in text


def test_wrap_text_truncates_long_strings():
    short = "Hello world"
    assert wrap_text(short, max_chars=20) == "Hello world"

    long_str = "A" * 100
    wrapped = wrap_text(long_str, max_chars=20)
    assert len(wrapped) == 20
    assert wrapped.endswith("...")


def test_similarity_color_palette():
    high_color = get_similarity_color(0.95)
    medium_color = get_similarity_color(0.80)
    low_color = get_similarity_color(0.50)

    assert high_color.hexval().lower() == "0xff4b4b"
    assert medium_color.hexval().lower() == "0xffa500"
    assert low_color.hexval().lower() == "0x21c55d"


def test_compress_pdf_buffer_reduces_size(monkeypatch):
    # Mock compress_pdf_buffer to get the raw uncompressed buffer size
    from src.utils import pdf_report

    original_compress = pdf_report.compress_pdf_buffer

    monkeypatch.setattr(pdf_report, "compress_pdf_buffer", lambda x: x)

    # Generate uncompressed report (with many matching pairs to make it larger)
    uncompressed_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            ("First matching paragraph.", "Second matching paragraph.", 0.96),
        ]
        * 50,
    )
    uncompressed_size = len(uncompressed_buffer.getvalue())

    # Call original compress function on the uncompressed buffer
    compressed_buffer = original_compress(uncompressed_buffer)
    compressed_size = len(compressed_buffer.getvalue())

    # Verify that the compressed version is smaller
    assert compressed_size < uncompressed_size

    # Verify it is still a valid PDF and the text matches
    compressed_bytes = compressed_buffer.getvalue()
    assert compressed_bytes.startswith(b"%PDF")
    text = _read_text(compressed_bytes)
    assert "student_a.pdf" in text
    assert "First matching paragraph" in text


def test_compress_pdf_buffer_fallback(monkeypatch):
    import fitz

    def mock_fitz_open(*args, **kwargs):
        raise Exception("Mock PyMuPDF error")

    monkeypatch.setattr(fitz, "open", mock_fitz_open)

    # Generate plagiarism report which will trigger the fallback pipeline
    compressed_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            ("First matching paragraph.", "Second matching paragraph.", 0.96),
        ],
    )
    compressed_bytes = compressed_buffer.getvalue()

    # The PDF should still be valid even when PyMuPDF fails
    assert compressed_bytes.startswith(b"%PDF")
    text = _read_text(compressed_bytes)
    assert "student_a.pdf" in text


def test_compress_pdf_buffer_all_fail(monkeypatch):
    import sys

    import fitz

    def mock_fitz_open(*args, **kwargs):
        raise Exception("Mock PyMuPDF error")

    monkeypatch.setattr(fitz, "open", mock_fitz_open)

    # Disable pypdf and PyPDF2 locally to test full fallback safety
    original_pypdf = sys.modules.get("pypdf")
    original_PyPDF2 = sys.modules.get("PyPDF2")
    sys.modules["pypdf"] = None
    sys.modules["PyPDF2"] = None

    try:
        # Generate plagiarism report where all compression libraries are unavailable/fail
        pdf_buffer = generate_plagiarism_report(
            doc_a="student_a.pdf",
            doc_b="student_b.pdf",
            overall_similarity=0.934,
            threshold=0.59,
            top_pairs=[
                ("First matching paragraph.", "Second matching paragraph.", 0.96),
            ],
        )
        pdf_bytes = pdf_buffer.getvalue()

        # The PDF generation should still produce a valid uncompressed PDF report
        assert pdf_bytes.startswith(b"%PDF")
        text = _read_text(pdf_bytes)
        assert "student_a.pdf" in text
    finally:
        # Restore sys.modules safely
        if original_pypdf is not None:
            sys.modules["pypdf"] = original_pypdf
        else:
            sys.modules.pop("pypdf", None)

        if original_PyPDF2 is not None:
            sys.modules["PyPDF2"] = original_PyPDF2
        else:
            sys.modules.pop("PyPDF2", None)


def test_generate_qr_code_returns_valid_png():
    """generate_qr_code should return a non-empty PNG BytesIO buffer."""
    buf = generate_qr_code("https://verify.example.com/verify/abc-123")
    data = buf.getvalue()
    # PNG magic bytes: \x89PNG
    assert data[:4] == b"\x89PNG", "QR buffer does not start with PNG magic bytes"
    assert len(data) > 100, "QR PNG data is unexpectedly small"


def test_qr_code_drawing_does_not_throw():
    """
    Passing incident_id to generate_plagiarism_report should embed a QR code
    in the header without raising any ReportLab errors.
    The resulting buffer must be a valid, parseable PDF.
    """
    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.85,
        threshold=0.75,
        top_pairs=[
            ("Alpha paragraph text.", "Beta paragraph text.", 0.88),
        ],
        incident_id="test-incident-123",
        verification_base_url="https://verify.example.com",
    )
    pdf_bytes = pdf_buffer.getvalue()

    # Must be a valid PDF
    assert pdf_bytes.startswith(b"%PDF"), "Output is not a valid PDF"
    assert len(pdf_bytes) > 1000, "PDF is unexpectedly small"

    # Must be parseable (corrupt PDF would raise here)
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1, "PDF has no pages"

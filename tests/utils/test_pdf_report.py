"""Tests for src/utils/pdf_report.py PDF plagiarism report generation."""

import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from unittest.mock import patch

from PyPDF2 import PdfReader

from src.utils.pdf_report import (generate_plagiarism_report,
                                  get_similarity_color, wrap_text)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
GOLDEN_PATH = os.path.join(FIXTURE_DIR, "pdf_report_golden.hash")

FROZEN_TIME = datetime(2025, 6, 15, 12, 0, 0)

SNAPSHOT_INPUTS = {
    "doc_a": "essay_john_doe.pdf",
    "doc_b": "essay_jane_smith.pdf",
    "overall_similarity": 0.873,
    "threshold": 0.60,
    "top_pairs": [
        (
            "The mitochondria is the powerhouse of the cell and plays a crucial role in energy production.",
            "The mitochondria serves as the cell's primary energy generator through ATP synthesis.",
            0.94,
        ),
        (
            "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
            "Plants transform sunlight into chemical energy via the process of photosynthesis.",
            0.91,
        ),
        (
            "DNA replication occurs during the S phase of the cell cycle before mitosis begins.",
            "The cell replicates its DNA in the synthesis phase prior to mitotic division.",
            0.88,
        ),
    ],
}


def _generate_snapshot_pdf():
    """Generate a deterministic PDF for snapshot comparison."""
    with patch("src.utils.pdf_report.datetime") as mock_dt:
        mock_dt.now.return_value = FROZEN_TIME
        mock_dt.strftime = datetime.strftime
        return generate_plagiarism_report(**SNAPSHOT_INPUTS)


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


# ── Snapshot / Golden Fixture Tests ────────────────────────────────────────


def _load_golden_hash() -> str | None:
    """Load the golden hash from the fixture file if it exists."""
    if not os.path.isfile(GOLDEN_PATH):
        return None
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    return data.get("hash")


def _save_golden_hash(pdf_hash: str) -> None:
    """Persist the golden hash to the fixture file."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    data = {
        "hash": pdf_hash,
        "inputs": {
            "doc_a": SNAPSHOT_INPUTS["doc_a"],
            "doc_b": SNAPSHOT_INPUTS["doc_b"],
            "overall_similarity": SNAPSHOT_INPUTS["overall_similarity"],
            "threshold": SNAPSHOT_INPUTS["threshold"],
            "top_pairs_count": len(SNAPSHOT_INPUTS["top_pairs"]),
        },
        "generated_at": FROZEN_TIME.isoformat(),
    }
    with open(GOLDEN_PATH, "w") as f:
        json.dump(data, f, indent=2)


def test_snapshot_pdf_content_match():
    """Verify generated PDF content matches the golden fixture.

    The test generates a PDF with deterministic inputs (datetime is mocked),
    computes a SHA-256 hash of the output bytes, and compares it against a
    pre-computed golden hash stored in tests/fixtures/pdf_report_golden.hash.

    To update the golden fixture (e.g. after intentional layout changes), set
    the environment variable ``UPDATE_PDF_GOLDEN=1`` and run:
        UPDATE_PDF_GOLDEN=1 pytest tests/utils/test_pdf_report.py -k snapshot
    """
    pdf_buffer = _generate_snapshot_pdf()
    pdf_bytes = pdf_buffer.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    golden_hash = _load_golden_hash()

    if golden_hash is None or os.environ.get("UPDATE_PDF_GOLDEN") == "1":
        _save_golden_hash(current_hash)
        return

    assert current_hash == golden_hash, (
        f"PDF content hash mismatch.\n"
        f"  Expected: {golden_hash}\n"
        f"  Got:      {current_hash}\n"
        f"  Run with UPDATE_PDF_GOLDEN=1 to update the golden fixture."
    )


def test_snapshot_pdf_structure_valid():
    """Verify snapshot PDF is a valid PDF with expected text content."""
    pdf_buffer = _generate_snapshot_pdf()
    pdf_bytes = pdf_buffer.getvalue()

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    text = _read_text(pdf_bytes)
    assert "essay_john_doe.pdf" in text
    assert "essay_jane_smith.pdf" in text
    assert "87.3%" in text
    assert "mitochondria" in text
    assert "photosynthesis" in text
    assert "DNA replication" in text


def test_generate_plagiarism_report_dark_mode():
    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            ("First matching paragraph.", "Second matching paragraph.", 0.96),
        ],
        dark_mode=True,
    )
    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    text = _read_text(pdf_bytes)
    assert "student_a.pdf" in text


def test_generate_plagiarism_report_auto_detect_dark_mode():
    import streamlit as st

    st.session_state.theme = "Dark"
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
    st.session_state.theme = "Light"


def test_pdf_generation_memory_leak():
    """Verify that generating multiple PDFs sequentially does not leak memory."""
    import gc
    import psutil

    # 1. Warm-up run to initialize any lazy-loaded libraries (fonts, caches)
    _ = generate_plagiarism_report(**SNAPSHOT_INPUTS)
    
    # 2. Force garbage collection and take baseline
    gc.collect()
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    
    # 3. Generate 100 PDF reports sequentially
    iterations = 100
    for _ in range(iterations):
        buffer = generate_plagiarism_report(**SNAPSHOT_INPUTS)
        buffer.close()
        del buffer
        
    # 4. Force garbage collection again
    gc.collect()
    
    # 5. Measure final RSS
    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / (1024 * 1024)
    
    # 6. Assert memory growth is bounded
    # Allow 25 MB growth for normal Python allocator and caching behaviour
    assert growth_mb < 25.0, f"Memory leak detected: growth {growth_mb:.2f} MB exceeds threshold of 25.0 MB"

"""Tests for src/utils/pdf_report.py PDF plagiarism report generation."""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from pypdf import PdfReader

import pytest

from src.utils.pdf_report import (
    generate_plagiarism_report,
    generate_qr_code,
    get_similarity_color,
    wrap_text,
)
from unittest.mock import patch



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

# Test utilities for golden fixture comparison
from tests.utils import FIXTURES_DIR, compare_pdf_bytes, assert_pdf_matches

# Test utilities for golden fixture comparison

# Text stats utilities
from src.utils.text_stats import (
    count_words,
    count_sentences,
    count_unique_words,
    get_unique_word_ratio,
    compute_text_stats,
    format_stats_for_pdf,
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


def test_pdf_matches_golden_fixture():
    """Verify generated PDF matches the golden fixture (deterministic comparison)."""
    golden_path = FIXTURES_DIR / "generate_plagiarism_report.pdf"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    # Use same parameters as generate_golden_pdf.py to ensure deterministic comparison
    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            (
                "This is the first paragraph from document A that contains some text about the subject being discussed.",
                "This is the first paragraph from document B that contains similar text about the same subject being discussed.",
                0.96,
            ),
            (
                "The second paragraph discusses the methodology used in the research study and includes various statistical analyses.",
                "Methodology section describes the research approach and includes statistical analysis similar to the previous paragraph.",
                0.87,
            ),
            (
                "In the conclusion, the authors summarize their findings and suggest areas for future research.",
                "The authors conclude by summarizing their key findings and identifying potential areas for further investigation.",
                0.79,
            ),
            (
                "The introduction provides background information on the topic and establishes the context for the study.",
                "Introduction section gives background on the topic and sets up the research context.",
                0.72,
            ),
        ],
    )

    assert_pdf_matches(pdf_buffer.getvalue(), golden_path)


def test_pdf_generation_detection_fails_with_modified_content():
    """Verify that modified PDF content fails the golden fixture test."""
    golden_path = FIXTURES_DIR / "generate_plagiarism_report.pdf"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    # Generate PDF with MODIFIED content - should fail comparison
    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            (
                "MODIFIED PARAGRAPH CONTENT - THIS SHOULD FAIL",
                "Another modified paragraph that differs from the golden.",
                0.96,
            ),
        ],
    )

    is_match, error_msg = compare_pdf_bytes(pdf_buffer.getvalue(), golden_path)
    assert not is_match, f"Expected PDF comparison to fail but it passed: {error_msg}"


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


# ── Tests for text_stats.py ───────────────────────────────────────────────────


def test_count_words():
    """Test word counting function."""
    assert count_words("") == 0
    assert count_words("hello") == 1
    assert count_words("hello world") == 2
    assert count_words("Hello, world! How are you?") == 5


def test_count_sentences():
    """Test sentence counting function."""
    assert count_sentences("") == 0
    assert count_sentences("Hello.") == 1
    assert count_sentences("Hello. World.") == 2
    assert count_sentences("Hello! How are you? I'm fine.") == 3


def test_count_unique_words():
    """Test unique word counting function."""
    assert count_unique_words("") == 0
    assert count_unique_words("hello") == 1
    assert count_unique_words("hello world") == 2
    assert count_unique_words("Hello hello world") == 2  # Case insensitive


def test_get_unique_word_ratio():
    """Test unique word ratio calculation."""
    assert get_unique_word_ratio("") == 0.0
    assert get_unique_word_ratio("hello") == 1.0
    assert get_unique_word_ratio("hello world") == 1.0
    assert get_unique_word_ratio("hello hello world") == pytest.approx(2/3, rel=0.01)


def test_compute_text_stats():
    """Test comprehensive text statistics computation."""
    text = "Hello world. Hello there. The world is beautiful."
    stats = compute_text_stats(text)

    assert stats['word_count'] > 0
    assert stats['sentence_count'] > 0
    assert stats['unique_word_count'] > 0
    assert 0.0 <= stats['unique_word_ratio'] <= 1.0


def test_format_stats_for_pdf():
    """Test statistics formatting for PDF table."""
    stats = {
        'word_count': 150,
        'sentence_count': 12,
        'unique_word_count': 100,
        'unique_word_ratio': 0.67,
    }

    rows = format_stats_for_pdf(stats)

    assert len(rows) == 4
    assert rows[0] == ['Word Count', '150']
    assert rows[1] == ['Sentence Count', '12']
    assert rows[2] == ['Unique Words', '100']
    assert rows[3] == ['Unique Word Ratio', '67.00%']


def test_generate_plagiarism_report_with_text_stats():
    """Test PDF generation with text statistics included."""
    sample_text_a = "This is the first document with some text. It has multiple sentences and words. The content is designed to test the text statistics feature in the PDF report generation."
    sample_text_b = "This is the second document with different content. It has some similar words but mostly unique text. The purpose is to compare with the first document for plagiarism detection purposes."

    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.934,
        threshold=0.59,
        top_pairs=[
            ("First matching paragraph from document A.",
             "First matching paragraph from document B.", 0.96),
        ],
        doc_a_text=sample_text_a,
        doc_b_text=sample_text_b,
    )

    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

    # Verify statistics are in the PDF
    text = _read_text(pdf_bytes)
    assert "Document Statistics" in text
    assert "Word Count" in text
    assert "Sentence Count" in text
    assert "Unique Word Ratio" in text


def test_generate_plagiarism_report_without_text_stats():
    """Test PDF generation without text statistics (backward compatibility)."""
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

    # Statistics section should not be present when text not provided
    text = _read_text(pdf_bytes)
    assert "Document Statistics" not in text


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

    # Disable pypdf locally to test full fallback safety
    original_pypdf = sys.modules.get("pypdf")
    sys.modules["pypdf"] = None

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
    """Verify generated PDF text content matches the golden fixture.

    Compares extracted text content (not raw bytes) since ReportLab embeds
    a non-deterministic creation timestamp in the PDF binary on every run.

    To update the golden fixture (e.g. after intentional layout changes), set
    the environment variable ``UPDATE_PDF_GOLDEN=1`` and run:
        UPDATE_PDF_GOLDEN=1 pytest tests/utils/test_pdf_report.py -k snapshot
    """
    pdf_buffer = _generate_snapshot_pdf()
    pdf_bytes = pdf_buffer.getvalue()
    current_text = _read_text(pdf_bytes)
    current_hash = hashlib.sha256(current_text.encode()).hexdigest()

    golden_hash = _load_golden_hash()

    if golden_hash is None or os.environ.get("UPDATE_PDF_GOLDEN") == "1":
        _save_golden_hash(current_hash)
        return

    assert current_hash == golden_hash, (
        f"PDF text content hash mismatch.\n"
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


# ── Branding logo tests ────────────────────────────────────────────────────


def test_load_branding_logo_returns_bytes_for_valid_path(tmp_path):
    """load_branding_logo returns bytes when logo_path points to a real file."""
    import json
    from src.utils.pdf_report import load_branding_logo

    logo_file = tmp_path / "logo.png"
    logo_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)  # minimal PNG header

    cfg = {"logo_path": str(logo_file)}
    with patch("builtins.open", side_effect=[
        __import__("io").StringIO(json.dumps(cfg)),
        open(str(logo_file), "rb"),
    ]):
        pass  # use monkeypatch approach below

    # Directly patch _BRANDING_CONFIG_PATH via monkeypatch on the module
    config_file = tmp_path / "branding_config.json"
    config_file.write_text(json.dumps({"logo_path": str(logo_file)}))

    import src.utils.pdf_report as pdf_mod
    original = pdf_mod._BRANDING_CONFIG_PATH
    pdf_mod._BRANDING_CONFIG_PATH = str(config_file)
    try:
        result = load_branding_logo()
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
    finally:
        pdf_mod._BRANDING_CONFIG_PATH = original


def test_load_branding_logo_returns_none_for_missing_path(tmp_path):
    """load_branding_logo returns None when logo_path is empty."""
    import json
    from src.utils.pdf_report import load_branding_logo

    config_file = tmp_path / "branding_config.json"
    config_file.write_text(json.dumps({"logo_path": ""}))

    import src.utils.pdf_report as pdf_mod
    original = pdf_mod._BRANDING_CONFIG_PATH
    pdf_mod._BRANDING_CONFIG_PATH = str(config_file)
    try:
        assert load_branding_logo() is None
    finally:
        pdf_mod._BRANDING_CONFIG_PATH = original


def test_load_branding_logo_returns_none_for_invalid_path(tmp_path):
    """load_branding_logo returns None when logo_path points to a non-existent file."""
    import json
    from src.utils.pdf_report import load_branding_logo

    config_file = tmp_path / "branding_config.json"
    config_file.write_text(json.dumps({"logo_path": "/nonexistent/logo.png"}))

    import src.utils.pdf_report as pdf_mod
    original = pdf_mod._BRANDING_CONFIG_PATH
    pdf_mod._BRANDING_CONFIG_PATH = str(config_file)
    try:
        assert load_branding_logo() is None
    finally:
        pdf_mod._BRANDING_CONFIG_PATH = original


def test_pdf_generation_succeeds_with_custom_logo(tmp_path):
    """PDF generation succeeds when load_branding_logo returns valid image bytes."""
    from PIL import Image
    import io

    img = Image.new("RGB", (200, 80), color=(30, 58, 138))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    logo_bytes = buf.getvalue()

    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.85,
        threshold=0.59,
        top_pairs=[("Paragraph A text.", "Paragraph B text.", 0.87)],
        logo_image=logo_bytes,
    )
    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    assert _read_text(pdf_bytes) is not None


def test_generate_plagiarism_report_uses_configured_logo_when_no_bytes_are_provided(monkeypatch):
    """PDF generation should fall back to the branding helper when no logo bytes are passed."""
    import src.utils.pdf_report as pdf_mod

    seen_payloads = []

    class FakeImageReader:
        def __init__(self, payload):
            seen_payloads.append(payload.getvalue())

        def getSize(self):
            return (100, 40)

    monkeypatch.setattr(pdf_mod, "ImageReader", FakeImageReader)
    monkeypatch.setattr(pdf_mod, "load_branding_logo", lambda: b"configured-logo")

    pdf_buffer = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        overall_similarity=0.85,
        threshold=0.59,
        top_pairs=[("Paragraph A text.", "Paragraph B text.", 0.87)],
    )

    assert pdf_buffer.getvalue().startswith(b"%PDF")
    assert seen_payloads == [b"configured-logo", b"configured-logo", b"configured-logo"]


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


# ── i18n / language header tests ──────────────────────────────────────────


def test_pdf_report_headers_spanish():
    """PDF table headers are translated to Spanish when language='es'."""
    pdf_buffer = generate_plagiarism_report(
        doc_a="alumno_a.pdf",
        doc_b="alumno_b.pdf",
        overall_similarity=0.80,
        threshold=0.59,
        top_pairs=[("Párrafo A.", "Párrafo B.", 0.82)],
        language="es",
    )
    text = _read_text(pdf_buffer.getvalue())
    assert "Nombre del Documento" in text
    assert "Puntuación de Similitud" in text or "Puntuaci" in text
    assert "Umbral de Detección" in text or "Umbral de Detecci" in text


def test_pdf_report_headers_french():
    """PDF table headers are translated to French when language='fr'."""
    pdf_buffer = generate_plagiarism_report(
        doc_a="etudiant_a.pdf",
        doc_b="etudiant_b.pdf",
        overall_similarity=0.80,
        threshold=0.59,
        top_pairs=[("Paragraphe A.", "Paragraphe B.", 0.82)],
        language="fr",
    )
    text = _read_text(pdf_buffer.getvalue())
    assert "Nom du Document" in text
    assert "Score de Similarit" in text
    assert "Seuil de D" in text


# ── QR Code Verification Tests (#1078) ────────────────────────────────────


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

"""
test_standalone_image_ocr_issue_2720.py
----------------------------------------
Comprehensive unit test suite for Issue #2720:
Direct extraction of text from standalone image uploads (.png, .jpg, .jpeg) via Tesseract OCR.

This suite validates:
1. Registration of image extensions (.png, .jpg, .jpeg) in `ALLOWED_EXTENSIONS`.
2. Direct routing of image files to `extract_text_from_image` in `dispatch.py`.
3. Image preprocessing logic (RGBA/palette mode conversion, contrast enhancement, median filtering).
4. Handling of corrupted or invalid image file bytes.
5. Error handling when Tesseract is missing or raises MemoryError.
6. Integration tests across `src/core/parsers/dispatch.py`, `src/core/document_parser.py`, and `src/core/parsers/ocr_parser.py`.
7. Batch extraction dispatching for multiple standalone images.
8. Dynamic Tesseract PATH configuration logic.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.core.document_parser import (
    ALLOWED_EXTENSIONS as DOC_PARSER_ALLOWED_EXTENSIONS,
)
from src.core.parsers.dispatch import (
    ALLOWED_EXTENSIONS as DISPATCH_ALLOWED_EXTENSIONS,
)
from src.core.parsers.dispatch import (
    extract_text,
    get_supported_file_extensions,
)
from src.core.parsers.ocr_parser import (
    _configure_tesseract,
    extract_text_from_image,
    preprocess_image_for_ocr,
)
try:
    from src.errors import OCRDependencyError
except ImportError:
    class OCRDependencyError(Exception):
        pass


# Helper to generate dummy PNG/JPG image bytes containing simple shapes/text
def _generate_test_image_bytes(format_name: str = "PNG", mode: str = "RGB", color=(255, 255, 255)) -> bytes:
    img = Image.new(mode, (300, 100), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Section 1: Extension Registration & Dispatch Routing Assertions
# ---------------------------------------------------------------------------

def test_image_extensions_registered_in_dispatch_and_document_parser():
    """Assert that .png, .jpg, and .jpeg are properly registered in ALLOWED_EXTENSIONS."""
    expected_exts = {".png", ".jpg", ".jpeg"}
    assert expected_exts.issubset(DISPATCH_ALLOWED_EXTENSIONS)
    assert expected_exts.issubset(DOC_PARSER_ALLOWED_EXTENSIONS)


def test_get_supported_file_extensions_includes_images():
    """Assert get_supported_file_extensions helper returns image extensions sorted."""
    exts = get_supported_file_extensions()
    assert ".png" in exts
    assert ".jpg" in exts
    assert ".jpeg" in exts
    assert exts == sorted(exts)


@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
@patch("pytesseract.image_to_string")
def test_dispatch_routes_standalone_png_image_directly_to_ocr(mock_tesseract, mock_check_ocr):
    """Verify dispatch.extract_text() routes PNG files directly to image OCR engine."""
    mock_tesseract.return_value = "Extracted OCR text from standalone PNG essay screenshot."
    img_bytes = _generate_test_image_bytes("PNG")

    extracted = extract_text(img_bytes, "sample_essay_screenshot.png")
    assert extracted == "Extracted OCR text from standalone PNG essay screenshot."
    assert mock_tesseract.called


@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
@patch("pytesseract.image_to_string")
def test_dispatch_routes_standalone_jpg_image_directly_to_ocr(mock_tesseract, mock_check_ocr):
    """Verify dispatch.extract_text() routes JPG/JPEG files directly to image OCR engine."""
    mock_tesseract.return_value = "Extracted OCR text from JPG photo."
    img_bytes = _generate_test_image_bytes("JPEG")

    extracted = extract_text(img_bytes, "photo_submission.jpg")
    assert extracted == "Extracted OCR text from JPG photo."

    extracted_jpeg = extract_text(img_bytes, "photo_submission.jpeg")
    assert extracted_jpeg == "Extracted OCR text from JPG photo."


# ---------------------------------------------------------------------------
# Section 2: Standalone Image Preprocessing Suite
# ---------------------------------------------------------------------------

def test_preprocess_image_for_ocr_rgb_mode():
    """Test preprocessing on standard RGB image."""
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    processed = preprocess_image_for_ocr(img)

    assert processed.mode == "L"  # Converted to grayscale for OCR optimization
    assert processed.size == (200, 200)


def test_preprocess_image_for_ocr_rgba_transparency_mode():
    """Test preprocessing on RGBA image with alpha channel transparency."""
    img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    processed = preprocess_image_for_ocr(img)

    assert processed.mode == "L"
    assert processed.size == (200, 200)


def test_preprocess_image_for_ocr_palette_mode():
    """Test preprocessing on 8-bit palette (P) mode image."""
    img = Image.new("P", (100, 100))
    processed = preprocess_image_for_ocr(img)

    assert processed.mode == "L"
    assert processed.size == (100, 100)


def test_preprocess_image_for_ocr_bilevel_mode():
    """Test preprocessing on 1-bit bilevel (1) mode image."""
    img = Image.new("1", (150, 150), 1)
    processed = preprocess_image_for_ocr(img)

    assert processed.mode == "L"
    assert processed.size == (150, 150)


def test_preprocess_image_exception_fallback():
    """Verify image preprocessing fallback on broken PIL Image objects."""
    broken_img = MagicMock()
    broken_img.mode = "INVALID_MODE"
    broken_img.convert.side_effect = Exception("Convert error")

    res = preprocess_image_for_ocr(broken_img)
    assert res == broken_img


# ---------------------------------------------------------------------------
# Section 3: Error Handling & Resilience Matrix
# ---------------------------------------------------------------------------

@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
def test_extract_text_from_image_invalid_file_bytes(mock_check_ocr):
    """Verify extract_text_from_image handles corrupted file bytes cleanly."""
    corrupted_bytes = b"NOT_AN_IMAGE_FILE_DATA_BYTES"
    extracted = extract_text_from_image(corrupted_bytes)
    assert extracted == ""


@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
@patch("pytesseract.image_to_string", side_effect=MemoryError("Out of memory during OCR"))
def test_extract_text_from_image_memory_error_fallback(mock_tesseract, mock_check_ocr):
    """Verify memory exhaustion during pytesseract execution returns fallback string."""
    img_bytes = _generate_test_image_bytes("PNG")
    result = extract_text_from_image(img_bytes)
    assert result == "[OCR extraction failed for the file]"


@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
@patch("pytesseract.image_to_string", side_effect=Exception("Generic pytesseract failure"))
def test_extract_text_from_image_generic_exception_fallback(mock_tesseract, mock_check_ocr):
    """Verify generic pytesseract exception returns fallback string."""
    img_bytes = _generate_test_image_bytes("PNG")
    result = extract_text_from_image(img_bytes)
    assert result == "[OCR extraction failed for the file]"


def test_extract_text_from_image_missing_tesseract_dependency():
    """Verify missing tesseract dependency raises OCRDependencyError."""
    with patch("src.core.parsers.ocr_parser.check_ocr_dependencies", side_effect=OCRDependencyError("Tesseract not installed")):
        img_bytes = _generate_test_image_bytes("PNG")
        with pytest.raises(OCRDependencyError):
            extract_text_from_image(img_bytes)


# ---------------------------------------------------------------------------
# Section 4: End-to-End Extraction Pipeline Verification
# ---------------------------------------------------------------------------

@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
@patch("pytesseract.image_to_string")
def test_end_to_end_standalone_image_ocr_pipeline(mock_tesseract, mock_check_ocr):
    """Simulate complete submission of a student scanned essay PNG file."""
    essay_ocr_text = (
        "Plagiarism Detection in Academic Submissions\n"
        "Student Name: Alex Morgan\n"
        "Abstract: This research explores multi-lingual semantic alignment and document hashing."
    )
    mock_tesseract.return_value = essay_ocr_text
    png_data = _generate_test_image_bytes("PNG", "RGB", (255, 255, 255))

    extracted = extract_text(png_data, "essay_submission_page1.png")

    assert "Plagiarism Detection in Academic Submissions" in extracted
    assert "Alex Morgan" in extracted
    assert "Abstract:" in extracted


@patch("src.core.parsers.ocr_parser.check_ocr_dependencies")
@patch("pytesseract.image_to_string")
def test_end_to_end_batch_image_ocr_processing(mock_tesseract, mock_check_ocr):
    """Simulate multiple image uploads in a single batch dispatch call."""
    mock_tesseract.side_effect = ["Page 1 Text Content", "Page 2 Text Content"]
    img1 = _generate_test_image_bytes("PNG")
    img2 = _generate_test_image_bytes("JPEG")

    res1 = extract_text(img1, "page1.png")
    res2 = extract_text(img2, "page2.jpeg")

    assert res1 == "Page 1 Text Content"
    assert res2 == "Page 2 Text Content"


def test_configure_tesseract_custom_env(monkeypatch):
    """Test _configure_tesseract with explicit TESSERACT_CMD env variable."""
    monkeypatch.setenv("TESSERACT_CMD", "/custom/bin/tesseract")
    mock_pytesseract = MagicMock()

    _configure_tesseract(mock_pytesseract)
    assert mock_pytesseract.pytesseract.tesseract_cmd == "/custom/bin/tesseract"

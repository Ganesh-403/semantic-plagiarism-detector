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

"""
test_ocr_mixed_page_heuristic_issue_2710.py
--------------------------------------------
Extensive, high-coverage unit test suite for Issue #2710:
Optimizing PDF OCR Heuristics for mixed native text and scanned image pages.

This suite thoroughly validates page text-to-image coverage ratios, multi-image page structures,
edge-case bounding boxes, layout geometries, threshold limits, and OCR fallback decision matrices
across both `src/core/parsers/pdf_parser.py` and `src/core/document_parser.py`.
"""

from unittest.mock import MagicMock

import pytest

from src.core.document_parser import (
    _calculate_document_image_coverage as _calculate_document_image_coverage_doc,
)
from src.core.document_parser import _has_meaningful_text as _has_meaningful_text_doc
from src.core.parsers.pdf_parser import (
    _calculate_image_area_coverage as _calculate_image_area_coverage_pdf,
)
from src.core.parsers.pdf_parser import _has_meaningful_text as _has_meaningful_text_pdf

# ---------------------------------------------------------------------------
# Section 1: Core Heuristic Base Rules & Text Density Checks
# ---------------------------------------------------------------------------


def test_has_meaningful_text_normal_sufficient_text():
    """Verify that pure text with >= 15 native words and >= 30 alphanumeric chars evaluates to True."""
    text = "This is a document header with more than fifteen native words embedded in the header section of the page."
    assert _has_meaningful_text_pdf(text) is True
    assert _has_meaningful_text_doc(text) is True


def test_has_meaningful_text_insufficient_words():
    """Verify that text under 15 words evaluates to False regardless of character count."""
    text = "Short header section"
    assert _has_meaningful_text_pdf(text) is False
    assert _has_meaningful_text_doc(text) is False


def test_has_meaningful_text_empty_and_whitespace():
    """Verify that empty, None, or whitespace-only text evaluates to False."""
    assert _has_meaningful_text_pdf("") is False
    assert _has_meaningful_text_pdf("   \n\t  ") is False
    assert _has_meaningful_text_pdf(None) is False

    assert _has_meaningful_text_doc("") is False
    assert _has_meaningful_text_doc("   \n\t  ") is False
    assert _has_meaningful_text_doc(None) is False


def test_has_meaningful_text_special_characters_only():
    """Verify that text with special symbols/punctuation but under 30 alphanumeric chars fails."""
    text = "!@#$%^&*() _+=- []{}|;:',.<>?/~` !@#$%^&*()"
    assert _has_meaningful_text_pdf(text) is False
    assert _has_meaningful_text_doc(text) is False


# ---------------------------------------------------------------------------
# Section 2: Mixed-Media Pages with Large Scanned Images (Issue #2710 Core)
# ---------------------------------------------------------------------------


def test_has_meaningful_text_mixed_page_with_large_scanned_image_dict():
    """Issue #2710: Native header (>15 words) + massive scanned image (dict format).
    Must evaluate to False to force Tesseract OCR execution on the scanned essay."""
    header_text = (
        "Official University Assignment Header Document Title Student ID 109283 "
        "Course Name Deep Learning Systems Spring Semester 2026 Department of Computer Science"
    )
    mock_page = MagicMock()
    mock_page.width = 600
    mock_page.height = 800
    mock_page.images = [{"width": 500, "height": 600}]

    assert _has_meaningful_text_pdf(header_text, page=mock_page) is False
    assert _has_meaningful_text_doc(header_text, page=mock_page) is False


def test_has_meaningful_text_mixed_page_with_large_scanned_image_tuple():
    """Issue #2710: Native header + large scanned image (PyMuPDF tuple format [xref, smask, width, height, ...])."""
    header_text = (
        "Official University Assignment Header Document Title Student ID 109283 "
        "Course Name Deep Learning Systems Spring Semester 2026 Department of Computer Science"
    )
    mock_page = MagicMock()
    mock_page.width = 600
    mock_page.height = 800
    mock_page.images = [(12, 0, 550, 700, 8, "DeviceRGB")]

    assert _has_meaningful_text_pdf(header_text, page=mock_page) is False
    assert _has_meaningful_text_doc(header_text, page=mock_page) is False


def test_has_meaningful_text_mixed_page_with_get_images_method():
    """Verify PyMuPDF get_images() method compatibility on mock page."""
    header_text = (
        "Official University Assignment Header Document Title Student ID 109283 "
        "Course Name Deep Learning Systems Spring Semester 2026 Department of Computer Science"
    )
    mock_page = MagicMock(spec=["get_images", "width", "height"])
    mock_page.width = 600
    mock_page.height = 800
    mock_page.get_images.return_value = [(1, 0, 400, 500, 8, "DeviceRGB")]

    assert _has_meaningful_text_pdf(header_text, page=mock_page) is False
    assert _has_meaningful_text_doc(header_text, page=mock_page) is False


# ---------------------------------------------------------------------------
# Section 3: Small Icons & Logotype Exclusion Matrix
# ---------------------------------------------------------------------------


def test_has_meaningful_text_mixed_page_small_logo_icon():
    """Verify that a tiny header logo (30x30px) does NOT trigger OCR bypass when native text is abundant."""
    header_text = (
        "Official University Assignment Header Document Title Student ID 109283 "
        "Course Name Deep Learning Systems Spring Semester 2026 Department of Computer Science"
    )
    mock_page = MagicMock()
    mock_page.width = 600
    mock_page.height = 800
    mock_page.images = [{"width": 30, "height": 30}]

    assert _has_meaningful_text_pdf(header_text, page=mock_page) is True
    assert _has_meaningful_text_doc(header_text, page=mock_page) is True


def test_has_meaningful_text_multiple_tiny_bullet_icons():
    """Verify that multiple tiny icon bullets (< 10x10px) do not trigger forced OCR on a full text page."""
    body_text = (
        "Introduction to neural networks and gradient descent optimization algorithms. "
        "We discuss backpropagation, loss functions, learning rate schedules, momentum, and regularization techniques."
    )
    mock_page = MagicMock()
    mock_page.width = 612
    mock_page.height = 792
    mock_page.images = [
        {"width": 8, "height": 8},
        {"width": 8, "height": 8},
        {"width": 8, "height": 8},
    ]

    assert _has_meaningful_text_pdf(body_text, page=mock_page) is True
    assert _has_meaningful_text_doc(body_text, page=mock_page) is True


# ---------------------------------------------------------------------------
# Section 4: Boundary Threshold & Coverage Ratio Edge Cases
# ---------------------------------------------------------------------------


def test_has_meaningful_text_exact_20_percent_coverage_threshold():
    """Verify threshold boundary: 20% area coverage exactly triggers OCR evaluation."""
    text = "Comprehensive analysis of semantic similarity algorithms in automated plagiarism detection pipelines."
    mock_page = MagicMock()
    mock_page.width = 1000
    mock_page.height = 1000  # Total area = 1,000,000
    mock_page.images = [{"width": 400, "height": 500}]  # 200,000 area = 20.0%

    assert _has_meaningful_text_pdf(text, page=mock_page) is False
    assert _has_meaningful_text_doc(text, page=mock_page) is False


def test_has_meaningful_text_19_percent_coverage_small_dimensions():
    """Verify threshold boundary: 19% area coverage with dimensions < 200px allows native text."""
    text = "Comprehensive analysis of semantic similarity algorithms in automated plagiarism detection pipelines."
    mock_page = MagicMock()
    mock_page.width = 1000
    mock_page.height = 1000  # Total area = 1,000,000
    mock_page.images = [{"width": 190, "height": 100}]

    assert _has_meaningful_text_pdf(text, page=mock_page) is True
    assert _has_meaningful_text_doc(text, page=mock_page) is True


def test_has_meaningful_text_exact_200px_dimension_threshold():
    """Verify dimension threshold: Image with 200x200px forces OCR even if coverage < 20%."""
    text = "Comprehensive analysis of semantic similarity algorithms in automated plagiarism detection pipelines."
    mock_page = MagicMock()
    mock_page.width = 2000
    mock_page.height = 2000  # Total area = 4,000,000
    mock_page.images = [{"width": 200, "height": 200}]

    assert _has_meaningful_text_pdf(text, page=mock_page) is False
    assert _has_meaningful_text_doc(text, page=mock_page) is False


# ---------------------------------------------------------------------------
# Section 5: Area Coverage Helper Direct Functions
# ---------------------------------------------------------------------------


def test_calculate_image_area_coverage_empty_or_invalid():
    """Test _calculate_image_area_coverage directly with empty/invalid arguments."""
    ratio, has_large = _calculate_image_area_coverage_pdf([], 600, 800)
    assert ratio == 0.0
    assert has_large is False

    ratio, has_large = _calculate_image_area_coverage_pdf(
        [{"width": 100, "height": 100}], 0, 800
    )
    assert ratio == 0.0
    assert has_large is False

    ratio_doc, has_large_doc = _calculate_document_image_coverage_doc([], 600, 800)
    assert ratio_doc == 0.0
    assert has_large_doc is False


def test_calculate_image_area_coverage_multiple_images():
    """Test combined ratio calculation over multiple images on a single page."""
    images = [
        {"width": 200, "height": 300},  # 60,000
        {"width": 100, "height": 150},  # 15,000
    ]
    # Total image area = 75,000. Page area = 600x800 = 480,000 -> ratio = 0.15625 (15.625%)
    ratio, has_large = _calculate_image_area_coverage_pdf(images, 600, 800)
    assert pytest.approx(ratio, 0.001) == 0.15625
    assert has_large is True  # 200x300 triggers large dimension check


# ---------------------------------------------------------------------------
# Section 6: Robustness & Exception Handling
# ---------------------------------------------------------------------------


def test_has_meaningful_text_malformed_page_object_fallback():
    """Verify that broken or malformed page objects do not raise exceptions and fall back cleanly."""
    text = "Standard native document text containing sufficient words to pass the default native word count check."
    broken_page = MagicMock()
    broken_page.width = "invalid_string_width"
    broken_page.images = [None]

    assert _has_meaningful_text_pdf(text, page=broken_page) is True
    assert _has_meaningful_text_doc(text, page=broken_page) is True


def test_has_meaningful_text_zero_area_page_fallback():
    """Verify page with width=0 or height=0 handles division by zero safely."""
    text = "Standard native document text containing sufficient words to pass the default native word count check."
    zero_page = MagicMock()
    zero_page.width = 0
    zero_page.height = 0
    zero_page.images = [{"width": 100, "height": 100}]

    assert _has_meaningful_text_pdf(text, page=zero_page) is False
    assert _has_meaningful_text_doc(text, page=zero_page) is False


# ---------------------------------------------------------------------------
# Section 7: Comprehensive Page Layout Matrix Simulations
# ---------------------------------------------------------------------------


def test_simulation_academic_paper_with_header_and_scanned_diagram():
    """Simulation: Academic paper page with title, abstract, and scanned equation block."""
    text = (
        "Abstract - We present a novel framework for cross-lingual plagiarism detection "
        "using dense vector embeddings and contextual alignment. Our results indicate superior "
        "recall on benchmark datasets compared to classic n-gram overlap heuristics."
    )
    mock_page = MagicMock()
    mock_page.width = 612
    mock_page.height = 792
    mock_page.images = [{"width": 350, "height": 450}]

    assert _has_meaningful_text_pdf(text, page=mock_page) is False
    assert _has_meaningful_text_doc(text, page=mock_page) is False


def test_simulation_scanned_handwritten_assignment_with_printed_cover():
    """Simulation: Student assignment with printed cover metadata and scanned handwritten pages."""
    text = (
        "Course: CS101 Introduction to Algorithms. Assignment 3 - Sorting and Searching. "
        "Submitted by: Student Name (ID: 987654321). Date: October 24, 2026."
    )
    mock_page = MagicMock()
    mock_page.width = 595
    mock_page.height = 842  # A4 size
    mock_page.images = [{"width": 550, "height": 700}]

    assert _has_meaningful_text_pdf(text, page=mock_page) is False
    assert _has_meaningful_text_doc(text, page=mock_page) is False


def test_simulation_full_text_document_with_small_footer_brand():
    """Simulation: Standard multi-paragraph article with a small corporate footer brand logo."""
    text = (
        "Plagiarism detection systems rely on lexical analysis, syntactic parsing, and semantic "
        "representation. Lexical methods compare string n-grams or hash values to compute Jaccard "
        "or Levenshtein distances. Syntactic techniques analyze parse trees and sentence structures. "
        "Semantic models leverage transformers and vector embeddings to capture conceptual similarity."
    )
    mock_page = MagicMock()
    mock_page.width = 612
    mock_page.height = 792
    mock_page.images = [{"width": 120, "height": 40}]

    assert _has_meaningful_text_pdf(text, page=mock_page) is True
    assert _has_meaningful_text_doc(text, page=mock_page) is True

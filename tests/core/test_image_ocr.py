"""
tests/core/test_image_ocr.py
----------------------------
Unit tests for Image Screenshot OCR and Layout Plagiarism Detection.
"""

import pytest
import struct
from src.core.image_ocr_extractor import (
    extract_image_ocr,
    _parse_png_dimensions,
    BoundingBox,
)
from src.core.screenshot_layout_analyzer import (
    reconstruct_reading_order,
    compute_layout_coherence,
)


class TestImageOCRExtractor:
    def test_parse_png_dimensions(self):
        # Construct a minimal valid PNG header
        header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 800, 600)
        dims = _parse_png_dimensions(header)
        assert dims == (800, 600)

    def test_extract_image_ocr_empty(self):
        result = extract_image_ocr(b"")
        assert result.image_width == 0
        assert len(result.blocks) == 0

    def test_extract_image_ocr_simulated(self):
        # Use a fake byte string that fails PNG/JPEG parsing, triggering fallback
        result = extract_image_ocr(b"fake image data")
        assert result.image_width == 800  # Fallback width
        assert len(result.blocks) > 0
        assert result.image_hash != ""


class TestScreenshotLayoutAnalyzer:
    def test_reconstruct_reading_order_single_column(self):
        blocks = [
            BoundingBox(x=10, y=50, width=100, height=20, text="Line 2"),
            BoundingBox(x=10, y=10, width=100, height=20, text="Line 1"),
            BoundingBox(x=10, y=90, width=100, height=20, text="Line 3"),
        ]
        ordered = reconstruct_reading_order(blocks)
        assert ordered[0].text == "Line 1"
        assert ordered[1].text == "Line 2"
        assert ordered[2].text == "Line 3"

    def test_reconstruct_reading_order_multi_line(self):
        blocks = [
            BoundingBox(x=100, y=10, width=50, height=20, text="Right"),
            BoundingBox(x=10, y=10, width=50, height=20, text="Left"),
        ]
        ordered = reconstruct_reading_order(blocks)
        assert ordered[0].text == "Left"
        assert ordered[1].text == "Right"

    def test_compute_layout_coherence(self):
        from src.core.image_ocr_extractor import OCRResult

        blocks = [
            BoundingBox(x=10, y=10, width=100, height=20, text="A"),
            BoundingBox(x=10, y=40, width=100, height=20, text="B"),
        ]
        result = OCRResult(
            image_width=200, image_height=200, image_hash="test", blocks=blocks
        )
        coherence = compute_layout_coherence(result)
        assert coherence["block_count"] == 2
        assert coherence["layout_coherence"] > 0.0
        assert "A" in coherence["extracted_text"]

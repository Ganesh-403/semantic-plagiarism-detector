"""
src/core/image_ocr_extractor.py
-------------------------------
Image Screenshot OCR and Bounding Box Extractor.

Processes image bytes to extract text blocks and their spatial bounding boxes.
To avoid introducing heavy external dependencies like Tesseract or OpenCV,
this module implements a heuristic OCR simulator that parses image dimensions
from raw PNG/JPEG headers and generates structured bounding box grids.
In a production environment with heavy dependencies allowed, this module
would wrap a real OCR engine (e.g., pytesseract) while maintaining the
exact same output interface.
"""

import struct
import hashlib
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a spatial bounding box for a text block."""

    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class OCRResult:
    """Represents the complete OCR extraction result."""

    image_width: int
    image_height: int
    image_hash: str
    blocks: List[BoundingBox] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_hash": self.image_hash,
            "blocks": [b.to_dict() for b in self.blocks],
        }


def _parse_png_dimensions(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    """Extract width and height from a PNG file header."""
    if image_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if len(image_bytes) < 24:
        return None
    width, height = struct.unpack(">II", image_bytes[16:24])
    return width, height


def _parse_jpeg_dimensions(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    """Extract width and height from a JPEG file header."""
    if image_bytes[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(image_bytes) - 1:
        if image_bytes[i] != 0xFF:
            break
        marker = image_bytes[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            if i + 9 < len(image_bytes):
                height = struct.unpack(">H", image_bytes[i + 5 : i + 7])[0]
                width = struct.unpack(">H", image_bytes[i + 7 : i + 9])[0]
                return width, height
            break
        if i + 3 < len(image_bytes):
            length = struct.unpack(">H", image_bytes[i + 2 : i + 4])[0]
            i += 2 + length
        else:
            break
    return None


def extract_image_ocr(image_bytes: bytes) -> OCRResult:
    """Process image bytes and extract text blocks with bounding boxes.

    This implementation uses a heuristic grid-based approach to simulate
    OCR extraction. It parses the image dimensions from the file header
    and divides the image into a grid of text blocks. This provides a
    deterministic spatial layout for testing the layout analyzer without
    requiring external OCR libraries.

    Args:
        image_bytes: Raw bytes of the image file (PNG or JPEG).

    Returns:
        An OCRResult object containing dimensions and bounding boxes.
    """
    if not image_bytes:
        return OCRResult(image_width=0, image_height=0, image_hash="", blocks=[])

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # Attempt to parse dimensions
    dims = _parse_png_dimensions(image_bytes)
    if not dims:
        dims = _parse_jpeg_dimensions(image_bytes)

    if not dims:
        # Fallback for unsupported formats or corrupted headers
        logger.warning("Could not parse image dimensions, using default 800x600.")
        width, height = 800, 600
    else:
        width, height = dims

    blocks = []

    # Heuristic OCR simulation: divide image into a grid of text lines
    # This simulates the output of a real OCR engine for layout analysis testing
    line_height = max(20, height // 20)
    margin_x = int(width * 0.1)
    text_width = int(width * 0.8)

    y_pos = int(height * 0.1)
    line_num = 1

    while y_pos + line_height < height * 0.9:
        # Simulate varying line widths (paragraph indents, short lines)
        current_width = text_width
        if line_num % 5 == 0:
            current_width = int(text_width * 0.6)  # Simulate end of paragraph

        block = BoundingBox(
            x=margin_x,
            y=y_pos,
            width=current_width,
            height=line_height,
            text=f"Simulated text line {line_num}",
            confidence=0.95,
        )
        blocks.append(block)

        y_pos += line_height + 5
        line_num += 1

    logger.info("Extracted %d simulated OCR blocks from image.", len(blocks))

    return OCRResult(
        image_width=width, image_height=height, image_hash=image_hash, blocks=blocks
    )

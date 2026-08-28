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

"""Multimodal Image Document OCR Domain Model.

Defines data classes for extracted document OCR text chunks, bounding box coordinates,
image resolution quality telemetry, and OCR plagiarism scan records.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class BoundingBoxCoordinates:
    """Represents bounding box spatial coordinates of OCR extracted text."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence_score: float  # Range: 0.0 - 1.0


@dataclass
class OcrExtractedChunk:
    """Represents a text chunk extracted from a document image via OCR."""

    chunk_id: str
    image_id: str
    extracted_text: str
    bounding_box: BoundingBoxCoordinates
    page_number: int = 1


@dataclass
class MultimodalImageOcrMatch:
    """Represents a detected plagiarism match from image document OCR analysis."""

    scan_id: str
    source_image_id: str
    source_image_name: str
    target_reference_id: str
    target_reference_title: str
    ocr_text_similarity: float  # Range: 0.0 - 1.0
    layout_structure_similarity: float  # Range: 0.0 - 1.0
    overall_multimodal_score: float
    ocr_engine_used: str  # e.g., 'Tesseract-5.0', 'PaddleOCR-v3', 'EasyOCR'
    scanned_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MultimodalOcrAuditReport:
    """Audit report summary aggregating multimodal image OCR plagiarism scans."""

    report_id: str
    total_images_scanned: int
    total_ocr_text_chunks: int
    highest_similarity_ratio: float
    report_generated_at: datetime = field(default_factory=datetime.utcnow)
    matches: List[MultimodalImageOcrMatch] = field(default_factory=list)

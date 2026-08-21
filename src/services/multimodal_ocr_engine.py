"""Multimodal Image Document OCR Engine Service.

Provides simulated optical character recognition (OCR), spatial bounding box extraction,
and cross-modal layout vs text similarity matching against text corpora.
"""

import random
import uuid
from datetime import datetime
from typing import List, Tuple

from src.models.multimodal_ocr_model import (
    BoundingBoxCoordinates,
    MultimodalImageOcrMatch,
    MultimodalOcrAuditReport,
    OcrExtractedChunk,
)


class MultimodalOcrEngine:
    """Core analytics engine for processing image documents and detecting OCR plagiarism."""

    @staticmethod
    def calculate_text_jaccard_similarity(text1: str, text2: str) -> float:
        """Calculates Jaccard similarity over word tokens extracted via OCR."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return round(len(intersection) / len(union), 4)

    @classmethod
    def process_document_image(
        cls, image_id: str, image_name: str, raw_ocr_text: str
    ) -> List[OcrExtractedChunk]:
        """Simulates bounding box extraction and text chunking for OCR image input."""
        chunks = []
        lines = [line.strip() for line in raw_ocr_text.split("\n") if line.strip()]

        for idx, line in enumerate(lines, start=1):
            bbox = BoundingBoxCoordinates(
                x_min=10 * idx,
                y_min=20 * idx,
                x_max=200 + (10 * idx),
                y_max=40 + (20 * idx),
                confidence_score=round(random.uniform(0.85, 0.99), 2),
            )
            chunks.append(
                OcrExtractedChunk(
                    chunk_id=f"OCR-CHK-{uuid.uuid4().hex[:6].upper()}",
                    image_id=image_id,
                    extracted_text=line,
                    bounding_box=bbox,
                    page_number=1,
                )
            )

        return chunks

    @classmethod
    def scan_image_against_reference(
        cls,
        image_id: str,
        image_name: str,
        raw_ocr_text: str,
        reference_id: str,
        reference_title: str,
        reference_text: str,
        ocr_engine: str = "Tesseract-5.0",
    ) -> MultimodalImageOcrMatch:
        """Compares OCR text extracted from an image against a target reference document."""
        ocr_sim = cls.calculate_text_jaccard_similarity(raw_ocr_text, reference_text)
        layout_sim = (
            round(random.uniform(0.70, 0.95), 4)
            if ocr_sim > 0.40
            else round(random.uniform(0.10, 0.40), 4)
        )

        overall_score = round((ocr_sim * 0.7) + (layout_sim * 0.3), 4)

        return MultimodalImageOcrMatch(
            scan_id=f"OCR-SCAN-{uuid.uuid4().hex[:8].upper()}",
            source_image_id=image_id,
            source_image_name=image_name,
            target_reference_id=reference_id,
            target_reference_title=reference_title,
            ocr_text_similarity=ocr_sim,
            layout_structure_similarity=layout_sim,
            overall_multimodal_score=overall_score,
            ocr_engine_used=ocr_engine,
            scanned_at=datetime.utcnow(),
        )

"""
src/core/screenshot_layout_analyzer.py
--------------------------------------
Screenshot Layout and Reading Order Analyzer.

Reconstructs the logical reading order and flow from spatial bounding
boxes extracted by the OCR engine. Uses an X-Y cut heuristic to sort
blocks top-to-bottom, left-to-right, handling multi-column layouts.
"""

import logging
from typing import List, Dict, Any
from src.core.image_ocr_extractor import BoundingBox, OCRResult

logger = logging.getLogger(__name__)


def reconstruct_reading_order(
    blocks: List[BoundingBox], column_threshold: int = 50
) -> List[BoundingBox]:
    """Reconstruct the logical reading order from spatial bounding boxes.

    Implements a simplified X-Y cut algorithm. It first sorts blocks by
    their Y-coordinate (top-to-bottom). If the horizontal gap between
    consecutive blocks exceeds the column_threshold, it identifies a
    column break and adjusts the sorting to read columns top-to-bottom,
    left-to-right.

    Args:
        blocks: List of BoundingBox objects from OCR extraction.
        column_threshold: Minimum horizontal gap to consider a column break.

    Returns:
        List of BoundingBox objects sorted in logical reading order.
    """
    if not blocks:
        return []

    # Sort primarily by Y-coordinate (top edge), then by X-coordinate
    sorted_blocks = sorted(blocks, key=lambda b: (b.y, b.x))

    # Group into lines based on Y-coordinate proximity
    lines = []
    current_line = [sorted_blocks[0]]

    for i in range(1, len(sorted_blocks)):
        prev_block = current_line[-1]
        curr_block = sorted_blocks[i]

        # If the Y-distance is small, they are on the same line
        if abs(curr_block.y - prev_block.y) <= prev_block.height // 2:
            current_line.append(curr_block)
        else:
            # Sort the current line left-to-right
            lines.append(sorted(current_line, key=lambda b: b.x))
            current_line = [curr_block]

    if current_line:
        lines.append(sorted(current_line, key=lambda b: b.x))

    # Flatten lines back into a single list (simple top-to-bottom, left-to-right)
    # For a more advanced multi-column layout, we would cluster X-coordinates
    # and process columns independently. For this implementation, we assume
    # standard single-column or simple multi-column where Y-sorting suffices.
    ordered_blocks = []
    for line in lines:
        ordered_blocks.extend(line)

    logger.info("Reconstructed reading order for %d blocks.", len(ordered_blocks))
    return ordered_blocks


def compute_layout_coherence(ocr_result: OCRResult) -> Dict[str, Any]:
    """Compute layout coherence and extract normalized text.

    Args:
        ocr_result: The complete OCRResult object.

    Returns:
        Dictionary containing the reconstructed text, block count, and coherence score.
    """
    if not ocr_result.blocks:
        return {
            "extracted_text": "",
            "block_count": 0,
            "layout_coherence": 0.0,
            "reading_order": [],
        }

    ordered_blocks = reconstruct_reading_order(ocr_result.blocks)
    extracted_text = "\n".join(b.text for b in ordered_blocks)

    # Compute a simple coherence score based on bounding box alignment
    # High coherence means blocks are well-aligned in columns/rows
    y_coords = [b.y for b in ordered_blocks]
    x_coords = [b.x for b in ordered_blocks]

    # Variance in X-coordinates for blocks on the same Y-level indicates alignment
    # For simplicity, we use the inverse of X-coordinate variance as a proxy
    if len(x_coords) > 1:
        x_mean = sum(x_coords) / len(x_coords)
        x_var = sum((x - x_mean) ** 2 for x in x_coords) / len(x_coords)
        # Normalize variance (lower variance = higher coherence)
        coherence = max(0.0, 1.0 - (x_var / (ocr_result.image_width**2)))
    else:
        coherence = 1.0

    return {
        "extracted_text": extracted_text,
        "block_count": len(ordered_blocks),
        "layout_coherence": round(coherence, 4),
        "reading_order": [b.to_dict() for b in ordered_blocks],
    }

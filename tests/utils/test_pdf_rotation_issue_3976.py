"""
tests/utils/test_pdf_rotation_issue_3976.py
-------------------------------------------
Unit tests for Issue #3976: Handling PDF rotation in pdf_highlighter.py
to prevent misaligned highlight boxes on rotated scanned/standard pages.
"""

from __future__ import annotations

import os
import fitz
import pytest

from src.utils.pdf_highlighter import (
    highlight_pdf_matches,
    apply_highlight_with_popup_note,
    transform_rect_for_rotation,
)


@pytest.fixture
def multi_rotation_pdf_bytes() -> bytes:
    """Creates a 4-page PDF with 0, 90, 180, and 270 degree rotated pages."""
    doc = fitz.open()
    angles = [0, 90, 180, 270]
    
    for angle in angles:
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (80, 120),
            f"Rotated section containing distinct text for angle {angle} degrees.",
            fontsize=12,
        )
        if angle != 0:
            page.set_rotation(angle)
            
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_transform_rect_for_rotation_zero_angle():
    """Verify that rotation=0 leaves bounding rectangle unmodified."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    rect = fitz.Rect(50, 50, 200, 100)
    
    transformed = transform_rect_for_rotation(rect, page)
    assert transformed == rect
    doc.close()


def test_transform_rect_for_rotation_90_180_270():
    """Verify transform_rect_for_rotation transforms coordinates for 90, 180, 270 degree pages."""
    for angle in [90, 180, 270]:
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        page.set_rotation(angle)
        
        # Rect in unrotated coordinates outside rotated page.rect (800x600 for 90/270)
        rect_unrotated = fitz.Rect(100, 700, 200, 750)
        transformed = transform_rect_for_rotation(rect_unrotated, page)
        
        assert page.rect.contains(transformed)
        assert isinstance(transformed, fitz.Rect)
        doc.close()


def test_transform_rect_for_rotation_tuple_and_list_input():
    """Verify that transform_rect_for_rotation accepts tuple and list inputs."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.set_rotation(90)
    
    rect_tuple = (100, 700, 200, 750)
    transformed_tuple = transform_rect_for_rotation(rect_tuple, page)
    assert isinstance(transformed_tuple, fitz.Rect)
    assert page.rect.contains(transformed_tuple)
    
    rect_list = [100, 700, 200, 750]
    transformed_list = transform_rect_for_rotation(rect_list, page)
    assert isinstance(transformed_list, fitz.Rect)
    assert page.rect.contains(transformed_list)
    doc.close()


def test_highlight_pdf_matches_on_rotated_pages(multi_rotation_pdf_bytes: bytes):
    """Verify that highlight_pdf_matches properly applies highlights across rotated pages (0, 90, 180, 270)."""
    matching_phrases = [
        "Rotated section containing distinct text for angle 0 degrees.",
        "Rotated section containing distinct text for angle 90 degrees.",
        "Rotated section containing distinct text for angle 180 degrees.",
        "Rotated section containing distinct text for angle 270 degrees.",
    ]
    
    res_bytes = highlight_pdf_matches(
        pdf_bytes=multi_rotation_pdf_bytes,
        matching_phrases=matching_phrases,
        source_doc="Rotated_Source.pdf",
        similarity=0.95,
    )
    
    assert isinstance(res_bytes, bytes)
    assert len(res_bytes) > 0
    
    with fitz.open(stream=res_bytes, filetype="pdf") as doc:
        assert doc.page_count == 4
        for i, page in enumerate(doc):
            annots = list(page.annots())
            assert len(annots) >= 1, f"Expected highlights on rotated page {i} with rotation {page.rotation}"
            annot = annots[0]
            assert annot.info["title"] == "Plagiarism Match"
            assert "Rotated_Source.pdf" in annot.info["content"]
            assert "95.0%" in annot.info["content"]


def test_apply_highlight_with_popup_note_rotated_pages(tmp_path, multi_rotation_pdf_bytes: bytes):
    """Verify apply_highlight_with_popup_note applies annotations on rotated PDF pages."""
    input_pdf = str(tmp_path / "multi_rot_input.pdf")
    output_pdf = str(tmp_path / "multi_rot_output.pdf")
    
    with open(input_pdf, "wb") as f:
        f.write(multi_rotation_pdf_bytes)
        
    # Match coordinates for each page (page_num, rect)
    match_coords = [
        (0, fitz.Rect(80, 110, 300, 140)),
        (1, fitz.Rect(80, 110, 300, 140)),
        (2, fitz.Rect(80, 110, 300, 140)),
        (3, fitz.Rect(80, 110, 300, 140)),
    ]
    
    apply_highlight_with_popup_note(
        input_pdf_path=input_pdf,
        output_pdf_path=output_pdf,
        match_coordinates=match_coords,
        source_doc="RefDoc.pdf",
        similarity=0.88,
    )
    
    assert os.path.exists(output_pdf)
    with fitz.open(output_pdf) as doc:
        assert doc.page_count == 4
        for i, page in enumerate(doc):
            annots = list(page.annots())
            assert len(annots) == 1, f"Page {i} should have 1 annotation"
            assert annots[0].info["title"] == "Plagiarism Match"
            assert "88.0%" in annots[0].info["content"]

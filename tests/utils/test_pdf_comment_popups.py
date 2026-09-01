import os
import fitz
import pytest
from src.utils.pdf_highlighter import highlight_pdf_matches, apply_highlight_with_popup_note

def test_apply_highlight_with_popup_note(tmp_path):
    """
    Verify that apply_highlight_with_popup_note attaches the expected popup note info
    with content formatted as 'Matched with {source_doc} ({similarity:.1%})' and title 'Plagiarism Match'.
    """
    input_pdf = "tests/fixtures/clean.pdf"
    output_pdf = str(tmp_path / "annotated.pdf")
    
    source_doc = "Assignment1_Copy.docx"
    similarity = 0.854

    # Coordinates payload: (page_num, fitz.Rect)
    rect = fitz.Rect(100, 100, 200, 120)
    match_coordinates = [(0, rect)]

    apply_highlight_with_popup_note(
        input_pdf_path=input_pdf,
        output_pdf_path=output_pdf,
        match_coordinates=match_coordinates,
        source_doc=source_doc,
        similarity=similarity,
    )

    assert os.path.exists(output_pdf)

    # Open generated PDF and verify annotation info
    with fitz.open(output_pdf) as doc:
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) > 0
        info = annots[0].info
        assert info["title"] == "Plagiarism Match"
        assert info["content"] == "Matched with Assignment1_Copy.docx (85.4%)"


def test_highlight_pdf_matches_popup_info():
    """
    Verify that highlight_pdf_matches populates set_info when source_doc and similarity are provided.
    """
    with open("tests/fixtures/clean.pdf", "rb") as f:
        pdf_bytes = f.read()

    source_doc = "Original_Essay.pdf"
    similarity = 0.92

    res_bytes = highlight_pdf_matches(
        pdf_bytes,
        matching_phrases=["Plagiarism detection"],
        source_doc=source_doc,
        similarity=similarity,
    )
    assert isinstance(res_bytes, bytes)
    assert len(res_bytes) > 0

    with fitz.open(stream=res_bytes, filetype="pdf") as doc:
        has_popup = False
        for page in doc:
            for annot in page.annots():
                info = annot.info
                if info.get("title") == "Plagiarism Match":
                    assert "Original_Essay.pdf" in info.get("content", "")
                    assert "92.0%" in info.get("content", "")
                    has_popup = True
        assert has_popup

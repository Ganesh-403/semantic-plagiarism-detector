import fitz
import pytest
from src.utils.pdf_highlighter import highlight_pdf_matches

def test_multi_color_highlights():
    """
    Assert that score >= 0.9 results in Red stroke (1, 0, 0),
    0.7 <= score < 0.9 results in Orange stroke (1, 0.5, 0),
    and score < 0.7 results in Yellow stroke (1, 1, 0).
    """
    with open("tests/fixtures/clean.pdf", "rb") as f:
        pdf_bytes = f.read()

    # 1. High severity (score >= 0.9) -> Red
    res_red = highlight_pdf_matches(pdf_bytes, matching_phrases=[("Plagiarism detection", 0.95)])
    with fitz.open(stream=res_red, filetype="pdf") as doc:
        annots = list(doc[0].annots())
        assert len(annots) > 0
        stroke = annots[0].colors["stroke"]
        assert list(stroke) == [1.0, 0.0, 0.0]

    # 2. Medium severity (score 0.75) -> Orange
    res_orange = highlight_pdf_matches(pdf_bytes, matching_phrases=[("Plagiarism detection", 0.75)])
    with fitz.open(stream=res_orange, filetype="pdf") as doc:
        annots = list(doc[0].annots())
        assert len(annots) > 0
        stroke = annots[0].colors["stroke"]
        assert list(stroke) == [1.0, 0.5, 0.0]

    # 3. Low severity (score 0.50) -> Yellow
    res_yellow = highlight_pdf_matches(pdf_bytes, matching_phrases=[("Plagiarism detection", 0.50)])
    with fitz.open(stream=res_yellow, filetype="pdf") as doc:
        annots = list(doc[0].annots())
        assert len(annots) > 0
        stroke = annots[0].colors["stroke"]
        assert list(stroke) == [1.0, 1.0, 0.0]

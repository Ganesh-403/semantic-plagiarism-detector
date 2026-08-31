import fitz
import pytest
from src.utils.pdf_highlighter import highlight_pdf_matches as highlight_hl
from src.utils.pdf_report import highlight_pdf_matches as highlight_rep


def _create_unicode_pdf() -> bytes:
    """Helper to generate an in-memory PDF containing non-ASCII unicode text with accents and umlauts."""
    doc = fitz.open()
    page = doc.new_page()
    
    # Text with Spanish accents and German umlauts
    sample_text = (
        "El cambio climático en España es un problema crítico.\n"
        "Übermäßige Erwärmung führt zu Dürren in München und Köln.\n"
        "L'été à Paris et la préservation de la biodiversité."
    )
    
    page.insert_text((50, 72), sample_text, fontsize=12)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_highlight_pdf_matches_unicode_spanish():
    """Verify highlighting phrases with Spanish accents (e.g. 'El cambio climático en España')."""
    pdf_bytes = _create_unicode_pdf()
    matching_phrase = "El cambio climático en España"

    # Test pdf_highlighter
    highlighted_hl = highlight_hl(pdf_bytes, [matching_phrase])
    assert isinstance(highlighted_hl, bytes)
    assert len(highlighted_hl) > 0

    with fitz.open(stream=highlighted_hl, filetype="pdf") as doc:
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) >= 1
        assert annots[0].type[0] == 8  # 8 is Highlight annotation in PDF / PyMuPDF spec

    # Test pdf_report
    highlighted_rep = highlight_rep(pdf_bytes, [matching_phrase])
    assert isinstance(highlighted_rep, bytes)
    assert len(highlighted_rep) > 0

    with fitz.open(stream=highlighted_rep, filetype="pdf") as doc:
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) >= 1
        assert annots[0].type[0] == 8


def test_highlight_pdf_matches_unicode_umlauts_and_accents():
    """Verify highlighting phrases with German umlauts and French accents."""
    pdf_bytes = _create_unicode_pdf()
    phrases = [
        "Übermäßige Erwärmung",
        "préservation de la biodiversité",
    ]

    # Test pdf_highlighter with multiple non-ASCII phrases
    highlighted_hl = highlight_hl(pdf_bytes, phrases)
    with fitz.open(stream=highlighted_hl, filetype="pdf") as doc:
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) >= 2
        for annot in annots:
            assert annot.type[0] == 8

    # Test pdf_report with multiple non-ASCII phrases
    highlighted_rep = highlight_rep(pdf_bytes, phrases)
    with fitz.open(stream=highlighted_rep, filetype="pdf") as doc:
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) >= 2
        for annot in annots:
            assert annot.type[0] == 8

"""
src/utils/pdf_highlighter.py
----------------------------
Highlights overlapping phrases/sentences in a PDF file using PyMuPDF (fitz).
"""

from typing import List, Optional

import fitz  # PyMuPDF


def highlight_pdf_matches(
    pdf_bytes: bytes,
    matching_phrases: Optional[list[str]] = None,
    password: Optional[str] = None,
    source_doc: Optional[str] = None,
    similarity: Optional[float] = None,
) -> bytes:
    """Open a PDF in-memory, search for matching phrases, and apply yellow highlight annotations with popup notes."""
    if not pdf_bytes:
        return b""

    # Open PDF stream with PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Authenticate if encrypted
    if doc.is_encrypted:
        if password:
            doc.authenticate(password)
        else:
            return pdf_bytes

    if not matching_phrases:
        # Fallback: if no specific phrases provided, return unmodified PDF
        return pdf_bytes

    s_doc = source_doc if source_doc else "Source Document"
    s_sim = similarity if similarity is not None else 1.0

    # Iterate through pages and highlight matched text
    for page in doc:
        for phrase in matching_phrases:
            phrase_clean = phrase.strip()
            # Ignore ultra-short tokens to avoid over-highlighting single words
            if len(phrase_clean) > 8:
                matches = page.search_for(phrase_clean)
                for rect in matches:
                    annot = page.add_highlight_annot(rect)
                    annot.set_info(
                        content=f"Matched with {s_doc} ({s_sim:.1%})",
                        title="Plagiarism Match",
                    )
                    annot.set_colors(stroke=(1, 1, 0))  # Bright Yellow
                    annot.update()

    # Return modified PDF bytes
    return doc.write()


def apply_highlight_with_popup_note(
    input_pdf_path: str,
    output_pdf_path: str,
    match_coordinates: list,
    source_doc: str,
    similarity: float,
) -> None:
    """
    Applies yellow highlights to specified coordinates on a PDF page and attaches
    a hoverable popup comment card detailing the source document match metrics.
    """
    doc = fitz.open(input_pdf_path)

    for page_num, rect in match_coordinates:
        page = doc[page_num]
        annot = page.add_highlight_annot(rect)
        annot.set_info(
            content=f"Matched with {source_doc} ({similarity:.1%})",
            title="Plagiarism Match",
        )
        annot.set_colors(stroke=[1, 0.9, 0])
        annot.update()

    doc.save(output_pdf_path, garbage=3, deflate=True)
    doc.close()

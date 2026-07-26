"""
src/utils/pdf_highlighter.py
----------------------------
Highlights overlapping phrases/sentences in a PDF file using PyMuPDF (fitz).
"""

from typing import List, Optional

import fitz  # PyMuPDF


def highlight_pdf_matches(
    pdf_bytes: bytes,
    matching_phrases: Optional[List[str]] = None,
    password: Optional[str] = None,
) -> bytes:
    """Open a PDF in-memory, search for matching phrases, and apply yellow highlight annotations."""
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

    # Iterate through pages and highlight matched text
    for page in doc:
        for phrase in matching_phrases:
            phrase_clean = phrase.strip()
            # Ignore ultra-short tokens to avoid over-highlighting single words
            if len(phrase_clean) > 8:
                matches = page.search_for(phrase_clean)
                for rect in matches:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 1, 0))  # Bright Yellow
                    annot.update()

    # Return modified PDF bytes
    return doc.write()
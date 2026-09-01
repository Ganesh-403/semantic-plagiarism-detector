"""
src/utils/pdf_highlighter.py
----------------------------
Highlights overlapping phrases/sentences in a PDF file using PyMuPDF (fitz).
"""

import logging
from typing import List, Optional

import fitz  # PyMuPDF

from src.errors import PDFEncryptedError

logger = logging.getLogger(__name__)

__all__ = ["highlight_pdf_matches", "PDFEncryptedError", "get_word_ngrams"]


def get_word_ngrams(text: str, n: int = 6) -> list[str]:
    """Splits a block of text into overlapping n-gram phrases."""
    words = text.split()
    if len(words) <= n:
        return [text] if text.strip() else []

    ngrams = []
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i : i + n])
        ngrams.append(phrase)
    return ngrams


def highlight_pdf_matches(
    pdf_bytes: bytes,
    matching_phrases: Optional[list[str]] = None,
    password: Optional[str] = None,
    source_doc: Optional[str] = None,
    similarity: Optional[float] = None,
) -> bytes:
    """Open a PDF in-memory, search for matching phrases, and apply yellow highlight annotations."""
    if not pdf_bytes:
        return b""

    # Open PDF stream with PyMuPDF context manager
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        # Authenticate if encrypted
        if doc.is_encrypted:
            authenticated = False
            if password:
                authenticated = bool(doc.authenticate(password))
            else:
                # Try empty password in case PDF only has an owner password set
                authenticated = bool(doc.authenticate(""))

            if not authenticated:
                logger.warning("PDF is encrypted and password was not provided or invalid.")
                raise PDFEncryptedError(
                    "PDF is encrypted and password was not provided or invalid."
                )

        if not matching_phrases:
            # Fallback: if no specific phrases provided, return unmodified PDF
            return pdf_bytes

        s_doc = source_doc if source_doc else "Source Document"
        s_sim = similarity if similarity is not None else 1.0

        # Iterate through pages and highlight matched text
        for page in doc:
            for phrase in matching_phrases:
                phrase_clean = phrase.strip()
                if not phrase_clean:
                    continue

                # Generate 6-word sliding windows to counter localized paraphrasing
                sub_phrases = get_word_ngrams(phrase_clean, n=6)
                for sub_phrase in sub_phrases:
                    sub_clean = sub_phrase.strip()
                    if len(sub_clean) > 8:
                        matches = page.search_for(sub_clean)
                        for rect in matches:
                            annot = page.add_highlight_annot(rect)
                            annot.set_info(
                                content=f"Matched with {s_doc} ({s_sim:.1%})",
                                title="Plagiarism Match",
                            )
                            annot.set_colors(stroke=(1, 1, 0))  # Bright Yellow
                            annot.update()

        # Return modified PDF bytes with compression and garbage collection
        return doc.write(deflate=True, garbage=3)


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
    with fitz.open(input_pdf_path) as doc:
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

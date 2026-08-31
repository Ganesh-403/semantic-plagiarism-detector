"""
src/utils/pdf_highlighter.py
----------------------------
Highlights overlapping phrases/sentences in a PDF file using PyMuPDF (fitz).
"""

from typing import List, Optional, Tuple, Union

import fitz  # PyMuPDF


def get_word_ngrams(text: str, n: int = 6) -> list[str]:
    """Splits a block of text into overlapping n-gram phrases."""
    words = text.split()
    if len(words) <= n:
        return [text] if text.strip() else []

    ngrams = []
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i:i+n])
        ngrams.append(phrase)
    return ngrams


def highlight_pdf_matches(
    pdf_bytes: bytes,
    matching_phrases: Optional[Union[list[str], list[Tuple[str, float]]]] = None,
    password: Optional[str] = None,
    source_doc: Optional[str] = None,
    similarity: Optional[float] = None,
) -> bytes:
    """Open a PDF in-memory, search for matching phrases, and apply color-coded highlight annotations."""
    if not pdf_bytes:
        return b""

    # Open PDF stream with PyMuPDF context manager
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
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

        # Iterate through pages and highlight matched text
        for page in doc:
            for item in matching_phrases:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    phrase, score = item[0], float(item[1])
                else:
                    phrase = str(item)
                    score = similarity if similarity is not None else 0.5

                phrase_clean = str(phrase).strip()
                if not phrase_clean:
                    continue

                sub_phrases = get_word_ngrams(phrase_clean, n=6)
                for sub_phrase in sub_phrases:
                    sub_clean = sub_phrase.strip()
                    if len(sub_clean) > 8:
                        matches = page.search_for(sub_clean)
                        for rect in matches:
                            annot = page.add_highlight_annot(rect)

                            # --- DYNAMIC COLOR SCHEME TRIGGER ---
                            if score >= 0.9:
                                stroke_color = (1, 0, 0)    # Red for high severity (>= 90%)
                            elif score >= 0.7:
                                stroke_color = (1, 0.5, 0)  # Orange for medium severity (70-89%)
                            else:
                                stroke_color = (1, 1, 0)    # Yellow for baseline matches (< 70%)

                            annot.set_info(
                                content=f"Matched with {s_doc} ({score:.1%})",
                                title="Plagiarism Match",
                            )
                            annot.set_colors(stroke=stroke_color)
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
    Applies highlights to specified coordinates on a PDF page and attaches
    a hoverable popup comment card detailing the source document match metrics.
    """
    with fitz.open(input_pdf_path) as doc:
        for page_num, rect in match_coordinates:
            page = doc[page_num]
            annot = page.add_highlight_annot(rect)

            if similarity >= 0.9:
                stroke_color = [1, 0, 0]
            elif similarity >= 0.7:
                stroke_color = [1, 0.5, 0]
            else:
                stroke_color = [1, 1, 0]

            annot.set_info(
                content=f"Matched with {source_doc} ({similarity:.1%})",
                title="Plagiarism Match",
            )
            annot.set_colors(stroke=stroke_color)
            annot.update()

        doc.save(output_pdf_path, garbage=3, deflate=True)

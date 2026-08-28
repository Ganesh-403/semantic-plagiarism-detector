# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

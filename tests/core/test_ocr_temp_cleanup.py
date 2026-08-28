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
test_ocr_temp_cleanup.py
------------------------
Tests ensuring that temporary artifacts created during OCR processing (PNGs, txt files)
are strictly managed via Python's `tempfile.TemporaryDirectory()` context manager, and are
automatically purged even if Tesseract crashes or raises exceptions mid-execution.
"""

import os
from unittest.mock import MagicMock, patch

from src.core.document_parser import _ocr_pdf_page, extract_text_from_image


def test_ocr_pdf_page_cleans_up_temp_dir_on_success():
    """Verify _ocr_pdf_page executes inside managed_ocr_temp_dir and leaves no leftover files."""
    captured_temp_dirs = []

    def mock_image_to_string(img, **kwargs):
        # Record the active tempdir during pytesseract execution
        captured_temp_dirs.append(os.environ.get("TMPDIR"))
        # Create dummy file inside active tempdir to simulate Tesseract temp file
        tmp_file = os.path.join(os.environ.get("TMPDIR"), "tess_temp_123.txt")
        with open(tmp_file, "w") as f:
            f.write("Extracted OCR text")
        return "Extracted OCR text"

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page
    mock_pixmap = MagicMock()
    mock_pixmap.samples = b"\xff" * 3
    mock_pixmap.width = 1
    mock_pixmap.height = 1
    mock_page.get_pixmap.return_value = mock_pixmap

    mock_fitz = MagicMock()
    mock_fitz.open.return_value.__enter__.return_value = mock_doc
    mock_pytesseract = MagicMock()
    mock_pytesseract.TesseractNotFoundError = type(
        "TesseractNotFoundError", (Exception,), {}
    )
    mock_pytesseract.image_to_string = mock_image_to_string

    with patch.dict(
        "sys.modules",
        {
            "fitz": mock_fitz,
            "pytesseract": mock_pytesseract,
            "PIL": MagicMock(),
            "PIL.Image": MagicMock(),
        },
    ):
        result = _ocr_pdf_page(b"%PDF-1.4", 0)
        assert result == "Extracted OCR text"

    assert len(captured_temp_dirs) == 1
    active_temp_dir = captured_temp_dirs[0]
    assert active_temp_dir is not None
    # Verify the temporary directory was deleted upon completion
    assert not os.path.exists(active_temp_dir)


def test_ocr_pdf_page_cleans_up_temp_dir_on_tesseract_crash():
    """Verify _ocr_pdf_page cleans up temporary directory if Tesseract crashes mid-execution."""
    captured_temp_dirs = []

    def crashing_image_to_string(img, **kwargs):
        temp_dir = os.environ.get("TMPDIR")
        captured_temp_dirs.append(temp_dir)
        # Create temp files that would normally leak during a crash
        png_file = os.path.join(temp_dir, "tess_input_crash.png")
        txt_file = os.path.join(temp_dir, "tess_output_crash.txt")
        with open(png_file, "wb") as f:
            f.write(b"PNG header data")
        with open(txt_file, "w") as f:
            f.write("Partial OCR output")
        # Simulate Tesseract crash / exception
        raise RuntimeError("Tesseract process crashed mid-execution!")

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page
    mock_pixmap = MagicMock()
    mock_pixmap.samples = b"\x00" * 3
    mock_pixmap.width = 1
    mock_pixmap.height = 1
    mock_page.get_pixmap.return_value = mock_pixmap

    mock_fitz = MagicMock()
    mock_fitz.open.return_value.__enter__.return_value = mock_doc
    mock_pytesseract = MagicMock()
    mock_pytesseract.TesseractNotFoundError = type(
        "TesseractNotFoundError", (Exception,), {}
    )
    mock_pytesseract.image_to_string = crashing_image_to_string

    with patch.dict(
        "sys.modules",
        {
            "fitz": mock_fitz,
            "pytesseract": mock_pytesseract,
            "PIL": MagicMock(),
            "PIL.Image": MagicMock(),
        },
    ):
        result = _ocr_pdf_page(b"%PDF-1.4", 0)
        assert "[OCR extraction failed for page 0]" in result

    assert len(captured_temp_dirs) == 1
    crashed_temp_dir = captured_temp_dirs[0]
    assert crashed_temp_dir is not None
    # Verify the temporary directory and all crash files were deleted
    assert not os.path.exists(crashed_temp_dir)


def test_extract_text_from_image_cleans_up_temp_dir_on_crash():
    """Verify extract_text_from_image cleans up temporary files if pytesseract crashes."""
    captured_temp_dirs = []

    def crashing_image_to_string(img, **kwargs):
        temp_dir = os.environ.get("TMPDIR")
        captured_temp_dirs.append(temp_dir)
        # Create temp files that would leak on crash
        with open(os.path.join(temp_dir, "tess_crash.png"), "w") as f:
            f.write("temp img")
        raise RuntimeError("Tesseract crash during image OCR")

    mock_img = MagicMock()
    mock_image = MagicMock()
    mock_image.open.return_value = mock_img
    mock_pytesseract = MagicMock()
    mock_pytesseract.TesseractNotFoundError = type(
        "TesseractNotFoundError", (Exception,), {}
    )
    mock_pytesseract.image_to_string = crashing_image_to_string

    with patch.dict(
        "sys.modules",
        {
            "pytesseract": mock_pytesseract,
            "PIL": MagicMock(),
            "PIL.Image": mock_image,
        },
    ):
        result = extract_text_from_image(b"fake image bytes")
        assert "[OCR extraction failed for the file]" in result

    assert len(captured_temp_dirs) == 1
    crashed_dir = captured_temp_dirs[0]
    assert not os.path.exists(crashed_dir)

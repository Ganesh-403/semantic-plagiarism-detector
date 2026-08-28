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

from unittest.mock import patch

from src.pdf_router import (
    process_pdf_page,  # Adjust import based on your project structure
)


@patch("src.pdf_router.pytesseract.image_to_string")
def test_ocr_fallback_heuristic(mock_image_to_string):
    """
    Validate that MIN_NATIVE_WORDS_PER_PAGE correctly triggers or skips OCR:
    - Pages with native word counts below the threshold trigger OCR.
    - Pages with native word counts at or above the threshold skip OCR.
    """
    # 1. Test case: Page with 7 native words (below threshold, should trigger OCR)
    low_word_page = "one two three four five six seven"

    # Mock return value for pytesseract OCR
    mock_image_to_string.return_value = "ocr extracted text"

    # Process page through router
    process_pdf_page(low_word_page, min_threshold=8)

    # Assert pytesseract was called
    assert (
        mock_image_to_string.called
    ), "Expected OCR to be triggered for low native word count page."

    # Reset mock state
    mock_image_to_string.reset_mock()

    # 2. Test case: Page with 9 native words (above threshold, should skip OCR)
    high_word_page = "one two three four five six seven eight nine"

    process_pdf_page(high_word_page, min_threshold=8)

    # Assert pytesseract was NOT called
    assert (
        not mock_image_to_string.called
    ), "Expected OCR to be skipped for sufficient native word count page."

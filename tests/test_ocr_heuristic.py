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
    assert mock_image_to_string.called, (
        "Expected OCR to be triggered for low native word count page."
    )

    # Reset mock state
    mock_image_to_string.reset_mock()

    # 2. Test case: Page with 9 native words (above threshold, should skip OCR)
    high_word_page = "one two three four five six seven eight nine"

    process_pdf_page(high_word_page, min_threshold=8)

    # Assert pytesseract was NOT called
    assert not mock_image_to_string.called, (
        "Expected OCR to be skipped for sufficient native word count page."
    )

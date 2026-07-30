"""Tests for zero-width unicode character sanitizer (Issue #609)."""

import logging
from src.core.document_parser import (
    ZERO_WIDTH_CHARS_PATTERN,
    extract_text,
    sanitize_zero_width_characters,
)


def test_sanitize_zero_width_characters_removes_hidden_spaces():
    """Verify that zero-width unicode spaces are removed from text."""
    dirty_text = "Plagiarism\u200B Detection\u200C Test\u200D Code\ufeff"
    clean = sanitize_zero_width_characters(dirty_text)
    assert clean == "Plagiarism Detection Test Code"
    assert not ZERO_WIDTH_CHARS_PATTERN.search(clean)


def test_sanitize_zero_width_characters_preserves_clean_text():
    """Verify that normal text without zero-width characters is returned unchanged."""
    clean_text = "This is a normal assignment text without hidden characters."
    result = sanitize_zero_width_characters(clean_text)
    assert result == clean_text


def test_sanitize_zero_width_characters_logs_warning(caplog):
    """Verify that a security warning is logged when zero-width characters are detected."""
    dirty_text = "Hidden\u200BZero\u200BWidth"
    with caplog.at_level(logging.WARNING):
        result = sanitize_zero_width_characters(dirty_text, filename="essay.txt")

    assert result == "HiddenZeroWidth"
    assert "Security warning: Found and stripped 2 zero-width unicode character(s)" in caplog.text
    assert "essay.txt" in caplog.text


def test_extract_text_sanitizes_zero_width_spaces(caplog):
    """Verify that extract_text automatically strips zero-width characters from uploaded file data."""
    raw_content = "Student\u200B Essay\u200C Content with hidden unicode spaces.".encode("utf-8")
    with caplog.at_level(logging.WARNING):
        extracted = extract_text(raw_content, "sample.txt")

    assert "Student Essay Content with hidden unicode spaces." in extracted
    assert "\u200B" not in extracted
    assert "\u200C" not in extracted
    assert "Security warning: Found and stripped 2 zero-width unicode character(s)" in caplog.text

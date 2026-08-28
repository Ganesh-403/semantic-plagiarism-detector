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

"""Tests for zero-width unicode character sanitizer (Issue #609)."""

import logging

from src.core.parsers.cleaners import (
    ZERO_WIDTH_CHARS_PATTERN,
    sanitize_zero_width_characters,
)
from src.core.parsers.dispatch import extract_text


def test_sanitize_zero_width_characters_removes_hidden_spaces():
    """Verify that zero-width unicode spaces are removed from text."""
    dirty_text = "Plagiarism\u200b Detection\u200c Test\u200d Code\ufeff"
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
    dirty_text = "Hidden\u200bZero\u200bWidth"
    with caplog.at_level(logging.WARNING):
        result = sanitize_zero_width_characters(dirty_text, filename="essay.txt")

    assert result == "HiddenZeroWidth"
    assert (
        "Security warning: Found and stripped 2 zero-width unicode character(s)"
        in caplog.text
    )
    assert "essay.txt" in caplog.text


def test_extract_text_sanitizes_zero_width_spaces():
    """Verify that extract_text automatically strips zero-width characters from uploaded file data."""
    raw_content = (
        "Student\u200b Essay\u200c Content with hidden unicode spaces.".encode("utf-8")
    )
    extracted = extract_text(raw_content, "sample.txt")

    assert "Student Essay Content with hidden unicode spaces." in extracted
    assert "\u200b" not in extracted
    assert "\u200c" not in extracted


def test_sanitize_zero_width_characters_all_variations():
    """Verify that a test string containing ZWSP (\\u200b), ZWNJ (\\u200c), ZWJ (\\u200d),
    BOM (\\ufeff), and Word Joiner (\\u2060) is stripped of all zero-width characters while
    leaving the surrounding text intact (Issue #2699)."""
    # Create test string containing \u200b, \u200c, \u200d, \ufeff, and \u2060
    dirty_text = (
        "The\u200b quick\u200c brown\u200d fox\ufeff jumps\u2060 over the lazy dog."
    )
    cleaned = sanitize_zero_width_characters(dirty_text)

    # Assert surrounding text is intact
    assert cleaned == "The quick brown fox jumps over the lazy dog."

    # Assert none of the zero-width character variations remain
    for char in ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]:
        assert char not in cleaned


def test_sanitize_zero_width_characters_consecutive_and_word_splitting():
    """Verify consecutive zero-width characters inside words and at boundaries are cleanly stripped."""
    # Embedded inside a single word
    embedded_word = "P\u200bl\u200ca\u200dg\ufeffi\u2060arism"
    assert sanitize_zero_width_characters(embedded_word) == "Plagiarism"

    # Consecutive cluster at boundaries
    cluster_text = "\u200b\u200c\u200d\ufeff\u2060Header\u200b\u200c\u200d\ufeff\u2060 Body \u200b\u200c\u200d\ufeff\u2060Footer\u200b\u200c\u200d\ufeff\u2060"
    assert sanitize_zero_width_characters(cluster_text) == "Header Body Footer"


import pytest


@pytest.mark.parametrize(
    "char_code, char_name",
    [
        ("\u200b", "Zero-Width Space (ZWSP)"),
        ("\u200c", "Zero-Width Non-Joiner (ZWNJ)"),
        ("\u200d", "Zero-Width Joiner (ZWJ)"),
        ("\ufeff", "Byte Order Mark / Zero-Width No-Break Space (BOM)"),
        ("\u2060", "Word Joiner"),
        ("\u200e", "Left-to-Right Mark"),
        ("\u200f", "Right-to-Left Mark"),
    ],
)
def test_sanitize_zero_width_characters_individual_variations(char_code, char_name):
    """Verify each individual zero-width character variation is detected and stripped."""
    dirty_text = f"prefix{char_code}suffix"
    cleaned = sanitize_zero_width_characters(dirty_text)
    assert cleaned == "prefixsuffix"
    assert char_code not in cleaned

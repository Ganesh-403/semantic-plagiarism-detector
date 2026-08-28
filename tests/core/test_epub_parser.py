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

"""Tests for EPUB document parsing and extraction (Issue #2730)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.document_parser import extract_text, extract_text_from_epub


def test_extract_text_from_epub_clean_chapters():
    """Verify extract_text_from_epub parses XML/HTML structures and extracts clean text."""
    mock_item1 = MagicMock()
    mock_item1.get_type.return_value = 9
    mock_item1.get_content.return_value = b"<html><body><h1>Chapter 1</h1><p>This is the first chapter text.</p></body></html>"

    mock_item2 = MagicMock()
    mock_item2.get_type.return_value = 9
    mock_item2.get_content.return_value = b"<html><body><h2>Chapter 2</h2><p>Second chapter with <em>formatted</em> text.</p></body></html>"

    # Item that is not a document (e.g. image or stylesheet)
    mock_item_other = MagicMock()
    mock_item_other.get_type.return_value = 1
    mock_item_other.get_content.return_value = b"body { font-size: 12px; }"

    mock_book = MagicMock()
    mock_book.get_items.return_value = [mock_item1, mock_item_other, mock_item2]

    with patch("ebooklib.epub.read_epub", return_value=mock_book):
        extracted = extract_text_from_epub(b"dummy_epub_content")

    assert "Chapter 1 This is the first chapter text." in extracted
    assert "Chapter 2 Second chapter with formatted text." in extracted
    assert "font-size" not in extracted


def test_extract_text_from_epub_handles_invalid_or_corrupt_files():
    """Verify extract_text_from_epub gracefully handles corrupted input without raising unhandled exceptions."""
    with patch(
        "ebooklib.epub.read_epub", side_effect=ValueError("Corrupted EPUB archive")
    ):
        extracted = extract_text_from_epub(b"corrupted_bytes")
        assert extracted == ""

    with patch("ebooklib.epub.read_epub", side_effect=OSError("Read error")):
        extracted = extract_text_from_epub(b"corrupted_bytes")
        assert extracted == ""


def test_extract_text_pipeline_epub_dispatch():
    """Verify top-level extract_text function dispatches .epub files to extract_text_from_epub."""
    mock_item = MagicMock()
    mock_item.get_type.return_value = 9
    mock_item.get_content.return_value = (
        b"<div><p>Testing EPUB pipeline extraction dispatch.</p></div>"
    )

    mock_book = MagicMock()
    mock_book.get_items.return_value = [mock_item]

    with patch("ebooklib.epub.read_epub", return_value=mock_book):
        result = extract_text(b"mock_epub", "book.epub")

    assert "Testing EPUB pipeline extraction dispatch." in result

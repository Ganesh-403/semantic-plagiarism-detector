"""
tests/core/test_text_chunking_min_words.py
------------------------------------------
Comprehensive unit tests for the dynamic min_words filtering logic
in the text chunking utilities (Issue #2912).

Verifies that small chunks (e.g., page numbers, headers) are discarded
early in the chunking loop to optimize performance.
"""

from src.core.text_chunking import (
    chunk_by_sentences,
    chunk_text,
    count_words,
)


class TestCountWords:
    """Test suite for the word counting utility."""

    def test_counts_standard_words(self):
        """Verify standard alphanumeric words are counted."""
        assert count_words("The quick brown fox") == 4

    def test_ignores_punctuation(self):
        """Verify punctuation is not counted as words."""
        assert count_words("Hello, world! How are you?") == 5

    def test_handles_empty_string(self):
        """Verify empty strings return 0."""
        assert count_words("") == 0
        assert count_words("   ") == 0

    def test_handles_none_input(self):
        """Verify None input returns 0."""
        assert count_words(None) == 0

    def test_counts_numbers_as_words(self):
        """Verify numeric sequences are counted as words."""
        assert count_words("Page 123 of 456") == 4


class TestChunkTextDynamicMinWords:
    """Test suite for dynamic min_words filtering in chunk_text (Issue #2912)."""

    def test_discards_small_chunks_early(self):
        """Verify chunks below min_words threshold are discarded."""
        # Create text with a very short segment that would form a chunk
        text = (
            "Short. "
            + "This is a much longer sentence that should definitely be included in the final output. "
            * 10
        )

        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0, min_words=5)

        # The "Short." chunk (1 word) should be discarded
        assert all(count_words(c.text) >= 5 for c in chunks)
        assert all("Short." not in c.text for c in chunks)

    def test_keeps_chunks_meeting_threshold(self):
        """Verify chunks meeting the min_words threshold are retained."""
        text = "This is a valid sentence with enough words. " * 5

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0, min_words=5)

        assert len(chunks) > 0
        assert all(count_words(c.text) >= 5 for c in chunks)

    def test_min_words_zero_keeps_all(self):
        """Verify min_words=0 retains all chunks regardless of size."""
        text = "A. B. C. D. E."

        chunks = chunk_text(text, chunk_size=2, chunk_overlap=0, min_words=0)

        # Should keep even 1-word chunks
        assert len(chunks) >= 5

    def test_page_numbers_discarded(self):
        """Verify isolated page numbers (common in PDFs) are discarded."""
        # Simulate PDF extraction with page numbers
        text = (
            "1\n"
            + "This is the actual content of the page with many words. " * 20
            + "\n2\n"
        )

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0, min_words=5)

        # Page numbers "1" and "2" should not appear as standalone chunks
        assert all("1" not in c.text for c in chunks)
        assert all("2" not in c.text for c in chunks)

    def test_headers_discarded_if_too_short(self):
        """Verify short headers are discarded if they fall below min_words."""
        text = "Chapter 1\n" + "This is the main body text with many words. " * 20

        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0, min_words=5)

        # "Chapter 1" is only 2 words, should be discarded
        assert all("Chapter 1" not in c.text for c in chunks)

    def test_overlap_does_not_create_invalid_chunks(self):
        """Verify overlapping windows don't generate invalid small chunks."""
        text = "Word " * 100

        chunks = chunk_text(text, chunk_size=50, chunk_overlap=25, min_words=5)

        assert all(count_words(c.text) >= 5 for c in chunks)

    def test_entire_text_too_small_returns_empty(self):
        """Verify if the entire text is below min_words, an empty list is returned."""
        text = "Too short."

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0, min_words=5)

        assert chunks == []

    def test_sentence_alignment_preserves_valid_chunks(self):
        """Verify sentence alignment doesn't accidentally discard valid chunks."""
        # Create a sentence that spans across the chunk_size boundary
        text = (
            "This is a very long sentence that will definitely cross the boundary. " * 5
        )

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0, min_words=5)

        # All resulting chunks should still meet the min_words requirement
        assert all(count_words(c.text) >= 5 for c in chunks)


class TestChunkBySentencesMinWords:
    """Test suite for min_words filtering in chunk_by_sentences."""

    def test_discards_small_sentence_groups(self):
        """Verify sentence groups below min_words are discarded."""
        text = (
            "Short. This is a much longer sentence with many words that should be kept."
        )

        chunks = chunk_by_sentences(text, max_chunk_size=100, min_words=5)

        # "Short." should be discarded
        assert all(count_words(c.text) >= 5 for c in chunks)

    def test_combines_short_sentences_to_meet_threshold(self):
        """Verify short sentences are combined to meet the min_words threshold."""
        text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."

        chunks = chunk_by_sentences(text, max_chunk_size=1000, min_words=5)

        # Should combine sentences until each chunk has >= 5 words
        assert all(count_words(c.text) >= 5 for c in chunks)

    def test_empty_text_returns_empty_list(self):
        """Verify empty text returns an empty list."""
        assert chunk_by_sentences("", min_words=5) == []
        assert chunk_by_sentences(None, min_words=5) == []

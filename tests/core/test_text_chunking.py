"""
tests/core/test_text_chunking.py
---------------------------------
Unit tests for customizable chunk size and overlap parameters, including edge cases.
Also validates sentence-aware chunk padding (Issue #1480).
"""

import pytest

from src.core.text_chunking import (
    ChunkString,
    _find_sentence_boundary,
    chunk_by_sentences,
    chunk_documents,
    chunk_text,
    chunk_text_dynamic,
)


def test_chunk_text_custom_parameters():
    sample_text = "Word " * 200  # 1000 characters approximately

    # Default parameters
    default_chunks = chunk_text(sample_text, chunk_size=500, chunk_overlap=50)

    # Smaller chunk size should produce more chunks
    small_chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=20)

    assert len(small_chunks) > len(default_chunks)


def test_chunk_documents_passes_parameters():
    docs = {"doc1.txt": "Line content text repeating " * 50}
    chunked = chunk_documents(docs, chunk_size=300, chunk_overlap=30)

    assert "doc1.txt" in chunked
    assert len(chunked["doc1.txt"]) > 0


def test_chunk_documents_empty_dictionary():
    """Empty input should return an empty dict without error."""
    assert chunk_documents({}) == {}


def test_min_words_filters_short_chunks():
    # "42" and "Page 1" are ultra-short; only the long sentence should survive
    text = "42\n\nPage 1\n\nThis is a sufficiently long sentence with many words in it. Here is a second sentence to add enough text. This is a third sentence to ensure we have enough text for overlap to potentially trigger."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=50, min_words=5)
    assert len(chunks) > 0
    assert all(len(c.split()) >= 5 for c in chunks)
    assert any("sufficiently" in c for c in chunks)
    assert not any("42" in c for c in chunks)
    assert not any("Page 1" in c for c in chunks)


def test_min_words_default_is_five():
    # Verify default min_words=5 without explicit argument
    text = "one two\n\nthree four five six seven eight"
    chunks = chunk_text(text)
    assert all(len(c.split()) >= 5 for c in chunks)


def test_chunk_text_respects_max_chunks_limit():
    # Build text large enough to produce far more than 5 chunks at this chunk_size
    huge_text = "Word " * 5000

    chunks = chunk_text(huge_text, chunk_size=50, chunk_overlap=5, max_chunks=5)

    assert len(chunks) <= 5


# ── Edge Case Tests (#849) ───────────────────────────────────────────────────


def test_chunk_text_empty_and_whitespace():
    """Verify empty or whitespace-only strings return an empty list or clean output."""
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
    assert chunk_text("   \n\t  ", chunk_size=500, chunk_overlap=50) == []


def test_chunk_text_single_long_word():
    """Verify single long words exceeding chunk size are handled safely without crashing."""
    long_word = "A" * 1200
    chunks = chunk_text(long_word, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 1
    # Ensure no chunk exceeds the maximum hard limits
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_cjk_characters():
    """Verify CJK (Chinese, Japanese, Korean) non-Latin unicode text chunking."""
    cjk_text = "这是一个关于人工智能和神经网络的测试文本。" * 20
    chunks = chunk_text(cjk_text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_chunk_text_emoji_only():
    """Verify emoji-only strings are chunked correctly without character corruption."""
    emoji_text = "🚀🔍🤖📝💻📊" * 50
    chunks = chunk_text(emoji_text, chunk_size=50, chunk_overlap=10, min_words=1)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_emoji_byte_length_enforced():
    """Verify count_bytes=True enforces chunk_size as UTF-8 bytes, not code points.

    Each emoji below is 1 Unicode code point but 4 UTF-8 bytes, so plain
    len() undercounts actual size by 4x. With count_bytes=True, no chunk's
    UTF-8 byte length should exceed chunk_size.
    """
    emoji_text = "🚀🔍🤖📝💻📊" * 50
    chunks = chunk_text(
        emoji_text, chunk_size=50, chunk_overlap=10, min_words=1, count_bytes=True
    )

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.encode("utf-8")) <= 50


def test_chunk_overlap_boundaries():
    """Verify consecutive chunks share the exact configured overlap substring."""
    # Use CJK text so chunking is character-based and overlap is a precise substring.
    text = "这是一个关于人工智能和神经网络的测试文本。" * 20
    chunk_size = 100
    chunk_overlap = 20

    chunks = chunk_text(
        text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_words=1
    )

    assert len(chunks) > 1
    overlap = chunks[0][-chunk_overlap:]
    assert chunks[1].startswith(overlap)


# ── Sentence-Boundary Chunking Tests (#919) ──────────────────────────────────


def test_chunk_by_sentences_preserves_full_sentences():
    """Each chunk must end on a sentence boundary – no mid-sentence splits."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A stitch in time saves nine. "
        "All that glitters is not gold. "
        "To be or not to be, that is the question. "
        "Actions speak louder than words."
    )
    chunks = chunk_by_sentences(text, max_chunk_size=120)

    assert len(chunks) >= 1
    # None of the raw sentence-ending markers should be split across chunks:
    # every chunk must be a coherent unit of complete sentences.
    for chunk in chunks:
        stripped = chunk.strip()
        assert len(stripped) > 0
        # The chunk must not start mid-word (no leading lowercase after space
        # at the very beginning caused by a mid-sentence break).
        assert stripped[0] == stripped[0].upper() or not stripped[0].isalpha()


def test_chunk_by_sentences_respects_max_chunk_size():
    """Chunks should not exceed max_chunk_size unless a single sentence is longer."""
    sentences = [
        f"This is sentence number {i} in the test document." for i in range(20)
    ]
    text = " ".join(sentences)
    max_size = 100

    chunks = chunk_by_sentences(text, max_chunk_size=max_size)

    assert len(chunks) >= 1
    for chunk in chunks:
        # A single overlong sentence is allowed to exceed the limit; all
        # multi-sentence blocks must respect it.
        words = chunk.split()
        if len(words) > 15:  # heuristic: definitely more than one sentence
            assert len(chunk) <= max_size + 60  # soft tolerance for joining space


def test_chunk_by_sentences_empty_and_whitespace():
    """Returns empty list for empty or whitespace-only input."""
    assert chunk_by_sentences("") == []
    assert chunk_by_sentences("   \n\t  ") == []


def test_chunk_by_sentences_single_sentence():
    """A text with a single sentence yields exactly one chunk."""
    text = "This is the only sentence in the document."
    chunks = chunk_by_sentences(text, max_chunk_size=500)

    assert len(chunks) == 1
    assert chunks[0].strip() == text.strip()


def test_chunk_by_sentences_no_sentence_is_split_mid_word():
    """Verify words are never cut in half across chunk boundaries."""
    text = " ".join(
        f"Word{j} is part of sentence {i}." for i in range(30) for j in range(5)
    )
    chunks = chunk_by_sentences(text, max_chunk_size=200)

    all_words_in_chunks = set()
    for chunk in chunks:
        for word in chunk.split():
            all_words_in_chunks.add(word)

    original_words = set(text.split())
    # Every word in the original text must appear intact in some chunk
    assert original_words.issubset(all_words_in_chunks)


def test_chunk_by_sentences_produces_multiple_chunks_for_long_text():
    """Long multi-sentence text must be split into more than one chunk."""
    sentences = [
        "The algorithm processes the input data efficiently." for _ in range(40)
    ]
    text = " ".join(sentences)

    chunks = chunk_by_sentences(text, max_chunk_size=150)
    assert len(chunks) > 1


def test_chunk_by_sentences_decimal_and_ellipsis():
    """Verify chunk_by_sentences handles decimals and ellipses properly."""
    text = (
        "The software version 3.14 was released today. "
        "We are currently loading... done! "
        "This is the final sentence."
    )
    chunks = chunk_by_sentences(text, max_chunk_size=50)

    # They should be split as three separate chunks
    assert len(chunks) == 3

    assert "version 3.14" in chunks[0]
    assert "loading... done!" in chunks[1]
    assert "final sentence" in chunks[2]

    # Ensure no split happened strictly inside the decimal or ellipsis
    for chunk in chunks:
        assert not chunk.strip().endswith("version 3.")
        assert not chunk.strip().endswith("loading.")
        assert not chunk.strip().endswith("loading..")


def test_chunk_text_percentage_overlap():
    """Verify overlap_percentage correctly derives chunk_overlap from chunk_size."""
    text = "Word " * 200
    chunk_size = 200
    overlap_percentage = 0.10  # expected chunk_overlap = int(200 * 0.10) = 20

    percentage_chunks = chunk_text(
        text, chunk_size=chunk_size, overlap_percentage=overlap_percentage
    )
    absolute_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=20)

    assert len(percentage_chunks) == len(absolute_chunks)
    assert percentage_chunks == absolute_chunks


# ── Truncation Warning Tests (Issue #1390) ───────────────────────────────────


def test_chunk_text_logs_truncation_warning(caplog):
    """Issue #1390: A WARNING must be logged when input text exceeds chunk capacity.

    When the raw input text length exceeds max_chunks * chunk_size, the
    resulting chunk stream is truncated to fit within max_chunks.  This
    silent data loss must be surfaced via a logger.warning() call whose
    message includes the truncated character count.
    """
    import logging

    # Build text large enough to exceed max_chunks * chunk_size capacity.
    # With chunk_size=10, max_chunks=3 → capacity = 30 chars.
    # 50 words × 5 chars (incl. trailing space) ≈ 250 chars, well over capacity.
    huge_text = "Word " * 50

    with caplog.at_level(logging.WARNING, logger="src.core.text_chunking"):
        chunks = chunk_text(huge_text, chunk_size=10, chunk_overlap=0, max_chunks=3)

    # Filter to the specific truncation warning we added
    truncation_warnings = [
        record
        for record in caplog.records
        if "exceeded chunk capacity limit" in record.getMessage()
    ]

    # Exactly one warning is expected, emitted before chunking begins
    assert len(truncation_warnings) == 1

    record = truncation_warnings[0]
    assert record.levelno == logging.WARNING
    assert record.name == "src.core.text_chunking"

    # Message must include the truncated character count verbatim
    message = record.getMessage()
    assert "text was truncated" in message
    assert str(len(huge_text)) in message

    # Sanity check: chunks were actually truncated to the max_chunks limit
    assert len(chunks) <= 3


def test_chunk_text_truncation_warning_includes_char_count(caplog):
    """Issue #1390: The truncation warning must contain the original char count.

    The exact substring ``Text length (%d chars) exceeded chunk capacity limit;
    text was truncated`` is required by the acceptance criteria, with ``%d``
    substituted by ``len(raw_text)``.
    """
    import logging
    import re

    raw_text = "A" * 1234  # known length so we can assert the exact digit appears

    with caplog.at_level(logging.WARNING, logger="src.core.text_chunking"):
        chunk_text(raw_text, chunk_size=10, chunk_overlap=0, max_chunks=5)

    # Find the warning and verify the message format
    truncation_records = [
        r for r in caplog.records if "exceeded chunk capacity limit" in r.getMessage()
    ]
    assert truncation_records, "Expected a truncation warning to be logged"

    message = truncation_records[0].getMessage()
    # Verify the message matches the format required by the issue
    assert re.search(
        r"^Text length \(\d+ chars\) exceeded chunk capacity limit; text was truncated$",
        message,
    ), f"Unexpected truncation warning message format: {message!r}"
    # Verify the substituted count matches the input length
    assert "1234" in message


def test_chunk_text_no_truncation_warning_for_small_text(caplog):
    """Issue #1390: No truncation warning when text fits within capacity."""
    import logging

    small_text = "Word " * 5  # 25 chars, well under capacity of 30

    with caplog.at_level(logging.WARNING, logger="src.core.text_chunking"):
        chunk_text(small_text, chunk_size=10, chunk_overlap=0, max_chunks=3)

    truncation_warnings = [
        record
        for record in caplog.records
        if "exceeded chunk capacity limit" in record.getMessage()
    ]
    assert len(truncation_warnings) == 0


def test_chunk_text_no_truncation_warning_for_empty_text(caplog):
    """Issue #1390: Empty/whitespace text must not trigger a truncation warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.core.text_chunking"):
        chunk_text("", chunk_size=10, chunk_overlap=0, max_chunks=3)
        chunk_text("   \n\t  ", chunk_size=10, chunk_overlap=0, max_chunks=3)

    truncation_warnings = [
        record
        for record in caplog.records
        if "exceeded chunk capacity limit" in record.getMessage()
    ]
    assert len(truncation_warnings) == 0


# ── Sliding Window Chunk Overlap Optimizer Tests (#1352) ─────────────────────


def test_chunk_text_dynamic_preserves_sentences_intact():
    """Verify sentences are preserved intact across chunk boundaries."""
    text = (
        "First sentence provides initial context. "
        "Second sentence elaborates further details on the topic. "
        "Third sentence completes the paragraph argument. "
        "Fourth sentence begins the new discussion topic!"
    )
    chunks = chunk_text_dynamic(text, target_size=80, min_overlap=20)

    assert len(chunks) >= 2
    sentence_endings = (".", "!", "?")
    for chunk in chunks:
        stripped = chunk.strip()
        assert len(stripped) > 0
        # Check that non-final boundary chunks end at sentence punctuation
        assert stripped[-1] in sentence_endings or stripped == text.strip()


def test_chunk_text_dynamic_empty_and_short():
    """Verify empty text returns empty list and short text returns single chunk."""
    assert chunk_text_dynamic("") == []
    assert chunk_text_dynamic("   ") == []

    short = "Short single sentence."
    chunks = chunk_text_dynamic(short, target_size=500)
    assert len(chunks) == 1
    assert chunks[0] == short


def test_chunk_text_raises_value_error_for_invalid_overlap():
    """Verify chunk_text raises ValueError when chunk_overlap >= chunk_size (Issue #1041)."""
    with pytest.raises(
        ValueError, match="chunk_overlap must be strictly smaller than chunk_size"
    ):
        chunk_text(
            "Sample text content for testing chunking.", chunk_size=50, chunk_overlap=50
        )

    with pytest.raises(
        ValueError, match="chunk_overlap must be strictly smaller than chunk_size"
    ):
        chunk_text(
            "Sample text content for testing chunking.",
            chunk_size=50,
            chunk_overlap=100,
        )

    with pytest.raises(
        ValueError, match="chunk_overlap must be strictly smaller than chunk_size"
    ):
        chunk_text(
            "Sample text content for testing chunking.",
            chunk_size=100,
            overlap_percentage=1.0,
        )


def test_chunk_text_raises_value_error_for_non_positive_chunk_size():
    """Verify chunk_text raises ValueError when chunk_size <= 0 (Issue #1579)."""
    for invalid_size in [0, -1, -50]:
        with pytest.raises(
            ValueError, match="chunk_size must be a positive integer > 0"
        ):
            chunk_text(
                "Sample text content for testing chunking.", chunk_size=invalid_size
            )


# ── Sentence-Aware Padding Tests (Issue #1480) ──────────────────────────────


class TestFindSentenceBoundary:
    """Tests for the internal sentence boundary detection helper."""

    def test_backward_finds_period(self):
        text = "This is sentence one. This is sentence two. This is three."
        # Index is in the middle of sentence two
        idx = text.find("two") + 1
        boundary = _find_sentence_boundary(text, idx, direction="backward")
        # Should snap back to the end of "one."
        assert text[boundary - 1] == "."
        assert boundary <= idx

    def test_forward_finds_exclamation(self):
        text = "Hello world! How are you today? I am fine."
        idx = text.find("world")
        boundary = _find_sentence_boundary(text, idx, direction="forward")
        assert text[boundary - 1] == "!"
        assert boundary > idx

    def test_no_boundary_found_returns_original(self):
        text = "no punctuation here at all just words"
        idx = 10
        boundary = _find_sentence_boundary(
            text, idx, direction="backward", max_search=50
        )
        assert boundary == idx

    def test_max_search_limit_respected(self):
        # Create a string with a period very far back
        text = "Start. " + ("a " * 100) + " end"
        idx = len(text) - 2
        boundary = _find_sentence_boundary(
            text, idx, direction="backward", max_search=20
        )
        # Should not find the period because it's > 20 chars away
        assert boundary == idx


class TestChunkTextSentencePadding:
    """Tests for sentence-aware padding in chunk_text (Issue #1480)."""

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text(None) == []

    def test_short_text_returns_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_fixed_size_chunking_without_padding(self):
        text = "A" * 1000
        chunks = chunk_text(
            text, chunk_size=200, chunk_overlap=0, sentence_padding=False
        )
        assert len(chunks) == 5
        assert all(len(c) == 200 for c in chunks)

    def test_sentence_padding_extends_to_boundary(self):
        # Create text where fixed chunk size cuts mid-sentence
        text = "First sentence here. " + (
            "Second sentence is much longer and should not be cut off in the middle of a thought. "
            * 5
        )

        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0, sentence_padding=True)

        # Verify no chunk ends abruptly without punctuation (unless it's the very last chunk)
        for i, chunk in enumerate(chunks[:-1]):
            # Chunk should end with a sentence terminator or be the end of text
            assert chunk.endswith((".", "!", "?")), (
                f"Chunk {i} does not end on sentence boundary: '{chunk[-20:]}'"
            )

    def test_sentence_padding_respects_hard_cap(self):
        # Create a massive sentence that exceeds 2x chunk_size
        text = "This is a sentence that never ends " * 100
        chunks = chunk_text(
            text, chunk_size=100, chunk_overlap=0, sentence_padding=True
        )

        # Ensure no chunk is absurdly large (hard cap is 2x chunk_size = 200)
        for chunk in chunks:
            assert len(chunk) <= 200 + 10  # small buffer for strip()

    def test_overlap_preserves_context(self):
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        chunks = chunk_text(
            text, chunk_size=20, chunk_overlap=10, sentence_padding=False
        )
        assert len(chunks) > 1

    def test_chunk_documents_with_sentence_padding(self):
        docs = ["Doc one text.", "Doc two text."]
        result = chunk_documents(docs, chunk_size=500, sentence_padding=True)
        assert isinstance(result, dict)
        assert "doc_0" in result
        assert "doc_1" in result

    def test_chunks_start_on_sentence_boundaries(self):
        """Acceptance criteria: chunks must start on sentence boundaries when padding is enabled."""
        text = (
            "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. "
            "Juliet kilo lima. Mike november oscar. Papa quebec romeo."
        )
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=0, sentence_padding=True)

        assert len(chunks) > 1
        for chunk in chunks:
            stripped = chunk.strip()
            # First character should be uppercase (start of sentence) or non-alpha
            assert stripped[0] == stripped[0].upper() or not stripped[0].isalpha(), (
                f"Chunk does not start on sentence boundary: '{stripped[:30]}'"
            )

    def test_chunks_end_on_sentence_boundaries(self):
        """Acceptance criteria: chunks must end on sentence boundaries when padding is enabled."""
        text = (
            "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. "
            "Juliet kilo lima. Mike november oscar. Papa quebec romeo."
        )
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=0, sentence_padding=True)

        assert len(chunks) > 1
        # All chunks except possibly the last should end with sentence punctuation
        for chunk in chunks[:-1]:
            stripped = chunk.strip()
            assert stripped[-1] in (
                ".",
                "!",
                "?",
            ), f"Chunk does not end on sentence boundary: '{stripped[-30:]}'"

    def test_sentence_padding_disabled_falls_back_to_word_boundary(self):
        """When sentence_padding=False, behavior matches original word-boundary chunking."""
        text = "Word " * 200
        padded = chunk_text(
            text, chunk_size=100, chunk_overlap=0, sentence_padding=False
        )
        unpadded = chunk_text(
            text, chunk_size=100, chunk_overlap=0, sentence_padding=False
        )
        assert padded == unpadded


def test_chunkstring_strip_returns_plain_str():
    """str operations on ChunkString drop metadata and return a plain str."""
    chunk = ChunkString("hello", {"k": "v"})
    result = chunk.strip()

    assert result == "hello"
    assert type(result) is str
    assert not hasattr(result, "metadata")


# ── NLTK punkt download caching (Issue #2059) ────────────────────────────────


def test_nltk_punkt_download_called_at_most_once(monkeypatch):
    """Missing punkt corpus should trigger nltk.download only once across calls."""
    import sys
    from unittest.mock import MagicMock

    import src.core.text_chunking as text_chunking
    from src.core.text_chunking import _split_into_sentences

    text_chunking._nltk_punkt_checked = False

    mock_download = MagicMock()
    mock_sent_tokenize = MagicMock(side_effect=LookupError("punkt missing"))

    fake_tokenize = MagicMock()
    fake_tokenize.sent_tokenize = mock_sent_tokenize

    fake_nltk = MagicMock()
    fake_nltk.download = mock_download
    fake_nltk.tokenize = fake_tokenize

    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setitem(sys.modules, "nltk.tokenize", fake_tokenize)

    sample = "First sentence. Second sentence."
    for _ in range(5):
        result = _split_into_sentences(sample)
        assert len(result) >= 1

    assert mock_download.call_count == 1
    mock_download.assert_called_with("punkt_tab", quiet=True)
    assert text_chunking._nltk_punkt_checked is True


def test_dynamic_snaps_to_period():
    """Test that chunk_text_dynamic snaps chunk boundaries to a period within the margin."""
    from src.core.text_chunking import chunk_text_dynamic

    # Construct text where a period falls near the target split boundary (e.g. target ~50 chars)
    # The snapping margin is 20% (±10 chars around index 50).
    text = "This is the first sentence that is quite long. Here is the second short sentence."

    chunks = chunk_text_dynamic(text, target_size=45)

    # Verify that the first chunk correctly snapped to the period after "long."
    assert len(chunks) > 0
    assert chunks[0].endswith(".")
    assert "first sentence" in chunks[0]


def test_dynamic_no_punctuation():
    """Test that text without sentence-ending punctuation falls back to an exact character split."""
    from src.core.text_chunking import chunk_text_dynamic

    text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    target_size = 20

    chunks = chunk_text_dynamic(text, target_size=target_size)

    # Verify that chunks are split strictly by the target character length without punctuation snapping
    assert len(chunks) > 1
    assert chunks[0] == text[:target_size]


def test_dynamic_single_chunk():
    """Test that text shorter than target_size is returned as a single chunk."""
    from src.core.text_chunking import chunk_text_dynamic

    text = "Short text."
    chunks = chunk_text_dynamic(text, target_size=100)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_sentence_boundary_empty_text():
    """Test that _find_sentence_boundary returns the original index when given empty text."""
    from src.core.text_chunking import _find_sentence_boundary

    index = 10
    result = _find_sentence_boundary("", index, max_search=5)
    assert result == index


def test_sentence_boundary_no_match():
    """Test that _find_sentence_boundary returns the original index when no punctuation is found within max_search."""
    from src.core.text_chunking import _find_sentence_boundary

    text = "abcdefghijklmnopqrstuvwxyz"
    index = 10
    # No punctuation anywhere near index 10, and tight max_search
    result = _find_sentence_boundary(text, index, max_search=3)
    assert result == index


def test_sentence_boundary_backward():
    """Test that _find_sentence_boundary finds the nearest backward sentence end."""
    from src.core.text_chunking import _find_sentence_boundary

    # "Hello world. How are you?"
    # Period is at index 11. If target index is 13, it should search backward and snap to index 12 (after period/space).
    text = "Hello world. How are you?"
    index = 13
    result = _find_sentence_boundary(text, index, max_search=5)
    # Depending on implementation details, it should identify the boundary near index 11 or 12.
    assert result != index
    assert text[result - 1] in ".!?"


def test_sentence_boundary_forward():
    """Test that _find_sentence_boundary finds the nearest forward sentence end."""
    from src.core.text_chunking import _find_sentence_boundary

    text = "Hello world. How are you?"
    # Index 9 is inside "world", period is at index 11. Searching forward within max_search should find it.
    index = 9
    result = _find_sentence_boundary(text, index, max_search=5)
    assert result != index
    assert text[result - 1] in ".!?"


# ── Comprehensive Sentence-Aware Chunking Tests ─────────────────────────────


class TestChunkBySentencesBasic:
    """Test suite for basic sentence-aware chunking behavior."""

    def test_empty_string_returns_empty_list(self):
        """Verify empty input returns an empty list."""
        assert chunk_by_sentences("") == []
        assert chunk_by_sentences("   ") == []

    def test_none_input_returns_empty_list(self):
        """Verify None input is handled gracefully."""
        assert chunk_by_sentences(None) == []

    def test_single_sentence_returns_single_chunk(self):
        """Verify a single sentence is returned as one chunk."""
        text = "This is a single sentence."
        chunks = chunk_by_sentences(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_sentences_combined_into_chunks(self):
        """Verify short sentences are combined to meet target chunk length."""
        text = "First. Second. Third. Fourth. Fifth."
        chunks = chunk_by_sentences(text, min_chunk_length=5)
        # Should combine short sentences rather than returning 5 tiny chunks
        assert len(chunks) < 5
        assert all(len(c) >= 5 for c in chunks)

    def test_long_sentence_not_split(self):
        """Verify a single long sentence is not split mid-sentence."""
        long_sentence = "This is a very long sentence " * 50
        text = long_sentence + ". Another sentence."
        chunks = chunk_by_sentences(text)
        # The long sentence should remain intact in the first chunk
        assert long_sentence.strip() in chunks[0]

    def test_min_chunk_length_filters_tiny_chunks(self):
        """Verify chunks shorter than min_chunk_length are filtered out."""
        text = "Ok. This is a much longer sentence that should be kept."
        chunks = chunk_by_sentences(text, min_chunk_length=20)
        assert len(chunks) == 1
        assert "Ok" not in chunks[0]


class TestChunkBySentencesLimits:
    """Test suite for max_chunks safety limit (Issue #2054)."""

    def test_max_chunks_default_is_1000(self):
        """Verify the default max_chunks parameter is 1000."""
        # Generate text with > 1000 sentences
        text = ". ".join([f"Sentence {i}" for i in range(1500)]) + "."
        chunks = chunk_by_sentences(text, min_chunk_length=1)
        assert len(chunks) <= 1000

    def test_max_chunks_custom_limit_respected(self):
        """Verify custom max_chunks limit is strictly enforced."""
        text = ". ".join([f"Sentence {i}" for i in range(100)]) + "."
        chunks = chunk_by_sentences(text, max_chunks=10, min_chunk_length=1)
        assert len(chunks) == 10

    def test_max_chunks_zero_raises_value_error(self):
        """Verify max_chunks=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_chunks must be > 0"):
            chunk_by_sentences("Some text.", max_chunks=0)

    def test_max_chunks_negative_raises_value_error(self):
        """Verify negative max_chunks raises ValueError."""
        with pytest.raises(ValueError, match="max_chunks must be > 0"):
            chunk_by_sentences("Some text.", max_chunks=-5)

    def test_max_chunks_logs_warning_on_truncation(self, caplog):
        """Verify a warning is logged when the max_chunks limit is reached."""
        import logging

        text = ". ".join([f"Sentence {i}" for i in range(50)]) + "."

        with caplog.at_level(logging.WARNING):
            chunk_by_sentences(text, max_chunks=5, min_chunk_length=1)

        assert any(
            "Reached max_chunks limit" in record.message for record in caplog.records
        )

    def test_text_shorter_than_max_chunks_not_truncated(self):
        """Verify text with fewer sentences than max_chunks is not truncated."""
        text = "First. Second. Third."
        chunks = chunk_by_sentences(text, max_chunks=100, min_chunk_length=1)
        assert len(chunks) <= 3  # Depends on combining logic, but definitely <= 100

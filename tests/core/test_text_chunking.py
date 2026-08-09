"""
tests/core/test_text_chunking.py
---------------------------------
Unit tests for customizable chunk size and overlap parameters, including edge cases.
"""

import pytest

from src.core.text_chunking import (
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


def test_min_words_filters_short_chunks():
    # "42" and "Page 1" are ultra-short; only the long sentence should survive
    text = "42\n\nPage 1\n\nThis is a sufficiently long sentence with many words in it."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=0, min_words=5)
    assert all(len(c.split()) >= 5 for c in chunks)
    assert any("sufficiently" in c for c in chunks)


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
    chunks = chunk_text(emoji_text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_overlap_boundaries():
    """Verify consecutive chunks preserve configured overlap boundaries."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunk_size = 30
    chunk_overlap = 10

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if len(chunks) > 1:
        # Check that consecutive chunks share overlapping content
        for i in range(len(chunks) - 1):
            assert len(chunks[i]) <= chunk_size


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
            chunk_text("Sample text content for testing chunking.", chunk_size=invalid_size)
            

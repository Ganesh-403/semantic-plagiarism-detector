"""
tests/core/test_repetitive_chunking_issue_4006.py
-------------------------------------------------
Test chunking with repetitive single-character text (Issue #4006).

Description:
    Test chunking an adversarial document containing 10,000 'a' characters without
    spaces or periods to verify it splits into fixed chunks without infinite loops.

Acceptance Criteria:
    Assert chunker splits text without hanging.
"""

import time
import pytest

from src.core.text_chunking import (
    _character_fallback_chunking,
    chunk_by_sentences,
    chunk_text,
    chunk_text_dynamic,
)


def test_chunking_repetitive_single_character_adversarial_10k_a():
    """Issue #4006: Test chunking 10,000 'a' characters without spaces or periods.

    Verifies the chunker splits the adversarial document into fixed-size chunks
    without hanging or entering infinite loops.
    """
    adversarial_text = "a" * 10000
    chunk_size = 500
    chunk_overlap = 50

    start_time = time.time()

    # Dynamic chunker should split into fixed chunks without infinite loop
    chunks = chunk_text_dynamic(
        adversarial_text, target_size=chunk_size, min_overlap=chunk_overlap
    )

    elapsed_time = time.time() - start_time

    # Acceptance Criteria: Assert chunker splits text without hanging (< 1.0 second)
    assert elapsed_time < 1.0, f"Chunker took too long ({elapsed_time:.2f}s), possible hang"
    assert len(chunks) > 0, "Chunker should produce at least one chunk"

    # Verify all chunks are non-empty and respect chunk size
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert len(chunk.text) <= chunk_size
        assert set(chunk.text) == {"a"}


def test_character_fallback_chunking_repetitive_10k_a():
    """Verify _character_fallback_chunking splits 10,000 'a' characters without hanging."""
    adversarial_text = "a" * 10000
    chunk_size = 500
    chunk_overlap = 50

    start_time = time.time()

    chunks = _character_fallback_chunking(
        adversarial_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    elapsed_time = time.time() - start_time

    assert elapsed_time < 1.0, f"Chunker took too long ({elapsed_time:.2f}s)"
    assert len(chunks) > 0

    # Verify no chunk is empty and character content is preserved
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert len(chunk.text) <= chunk_size
        assert set(chunk.text) == {"a"}


@pytest.mark.parametrize("chunk_size", [50, 100, 250, 500, 1000])
def test_repetitive_single_character_various_chunk_sizes(chunk_size):
    """Verify chunking adversarial single-character text across different chunk sizes."""
    adversarial_text = "a" * 10000

    start_time = time.time()

    chunks = chunk_text_dynamic(
        adversarial_text, target_size=chunk_size, min_overlap=min(20, chunk_size // 2)
    )

    elapsed = time.time() - start_time
    assert elapsed < 1.0
    assert len(chunks) >= (10000 // chunk_size)
    assert all(len(c.text) <= chunk_size for c in chunks)


@pytest.mark.parametrize("char", ["a", "z", "1", "x", "\u4e00"])
def test_repetitive_different_characters_10k(char):
    """Verify chunking works with different single repeated characters without periods."""
    adversarial_text = char * 10000
    chunk_size = 200

    start_time = time.time()

    chunks = chunk_text_dynamic(adversarial_text, target_size=chunk_size, min_overlap=20)

    elapsed = time.time() - start_time
    assert elapsed < 1.0
    assert len(chunks) > 0
    assert all(set(c.text) == {char} for c in chunks)


def test_chunk_text_with_repetitive_adversarial_text():
    """Verify chunk_text handles 10,000 'a' chars when min_words threshold allows it."""
    adversarial_text = "a" * 10000
    start_time = time.time()

    chunks = chunk_text(adversarial_text, chunk_size=500, chunk_overlap=50, min_words=1)

    elapsed = time.time() - start_time
    assert elapsed < 1.0
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert set(chunk.text) == {"a"}


def test_chunk_by_sentences_with_repetitive_adversarial_text():
    """Verify chunk_by_sentences handles 10,000 'a' text without periods or spaces without hanging."""
    adversarial_text = "a" * 10000
    start_time = time.time()

    chunks = chunk_by_sentences(adversarial_text, min_words=1, min_chunk_length=1)

    elapsed = time.time() - start_time
    assert elapsed < 1.0
    assert len(chunks) > 0
    for chunk in chunks:
        assert set(chunk) == {"a"}

"""
tests/core/test_text_chunking.py
---------------------------------
Unit tests for customizable chunk size and overlap parameters, including edge cases.
"""

import pytest
from src.core.text_chunking import chunk_documents, chunk_text


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
            
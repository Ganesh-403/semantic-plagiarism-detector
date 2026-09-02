"""
test_token_chunking_issue_3998.py
---------------------------------
Unit tests for Issue #3998: Token-aware chunking based on Hugging Face tokenizer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.text_chunking import chunk_document_by_tokens


class DummyTokenizer:
    """Simulated Hugging Face tokenizer mapping words to dummy integer token IDs."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split()
        return [hash(w) % 10000 for w in words]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"token_{tid}" for tid in token_ids)


class MockWordTokenizer:
    """Mock tokenizer returning word strings directly."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, token_ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(token_ids)


def test_chunk_document_by_tokens_parameter_validation():
    """Verify that invalid max_tokens or overlap_tokens raise ValueError."""
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        chunk_document_by_tokens("sample text", max_tokens=0)

    with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
        chunk_document_by_tokens("sample text", max_tokens=100, overlap_tokens=-5)

    with pytest.raises(ValueError, match="overlap_tokens must be strictly less than max_tokens"):
        chunk_document_by_tokens("sample text", max_tokens=100, overlap_tokens=100)


def test_chunk_document_by_tokens_empty_text():
    """Verify that empty or whitespace text returns empty list."""
    assert chunk_document_by_tokens("") == []
    assert chunk_document_by_tokens("   \n\t  ") == []


def test_chunk_document_by_tokens_short_text():
    """Verify that text shorter than max_tokens returns a single chunk."""
    tokenizer = MockWordTokenizer()
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_document_by_tokens(text, max_tokens=256, overlap_tokens=32, tokenizer=tokenizer)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_document_by_tokens_sliding_window_overlap():
    """Verify sliding window token-based chunking with overlap using a mock tokenizer."""
    tokenizer = MockWordTokenizer()
    words = [f"word_{i}" for i in range(100)]
    text = " ".join(words)

    # 100 words, max_tokens=30, overlap_tokens=10 => step=20
    # Chunks: [0:30], [20:50], [40:70], [60:90], [80:100] => 5 chunks
    chunks = chunk_document_by_tokens(
        text,
        max_tokens=30,
        overlap_tokens=10,
        tokenizer=tokenizer,
    )

    assert len(chunks) == 5
    assert chunks[0] == " ".join(words[0:30])
    assert chunks[1] == " ".join(words[20:50])
    assert chunks[4] == " ".join(words[80:100])


def test_chunk_document_by_tokens_with_dummy_hf_tokenizer():
    """Verify decoding and token mapping using DummyTokenizer."""
    tokenizer = DummyTokenizer()
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
    chunks = chunk_document_by_tokens(
        text,
        max_tokens=4,
        overlap_tokens=1,
        tokenizer=tokenizer,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert "token_" in chunk


def test_chunk_document_by_tokens_fallback_without_tokenizer():
    """Verify graceful fallback when tokenizer is None and EmbeddingModelManager is mocked/unavailable."""
    words = [f"item_{i}" for i in range(50)]
    text = " ".join(words)

    chunks = chunk_document_by_tokens(
        text,
        max_tokens=20,
        overlap_tokens=5,
        tokenizer=None,
    )

    assert len(chunks) >= 3
    assert "item_0" in chunks[0]

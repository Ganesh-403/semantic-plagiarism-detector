"""Tests for min_chunk_length on chunk_document (Issue #4001)."""

from src.core.text_chunking import (
    ChunkString,
    _merge_undersized_trailing_chunk,
    chunk_document,
)


def test_merge_undersized_trailing_chunk():
    chunks = [
        ChunkString(text="A" * 80),
        ChunkString(text="Page 12"),
    ]
    merged = _merge_undersized_trailing_chunk(chunks, min_chunk_length=40)
    assert len(merged) == 1
    assert merged[0].text.endswith("Page 12")
    assert "A" * 80 in merged[0].text


def test_merge_keeps_trailing_chunk_when_long_enough():
    chunks = [
        ChunkString(text="A" * 80),
        ChunkString(text="B" * 50),
    ]
    merged = _merge_undersized_trailing_chunk(chunks, min_chunk_length=40)
    assert len(merged) == 2


def test_chunk_document_merges_short_trailer():
    body = (
        "Semantic plagiarism detection compares meaning across student essays. "
        "It uses embeddings so paraphrased text still surfaces as similar. "
    ) * 8
    text = body + "\n\nPage 12"
    chunks = chunk_document(text, chunk_size=120, chunk_overlap=10, min_words=1)
    assert chunks
    assert all(len(c.text) >= 40 or len(chunks) == 1 for c in chunks)
    assert not any(c.text.strip() == "Page 12" for c in chunks)

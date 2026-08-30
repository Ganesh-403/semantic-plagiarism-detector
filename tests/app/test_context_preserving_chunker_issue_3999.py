"""Tests for paragraph-aware ContextPreservingChunker (Issue #3999)."""

from app.components.advanced_analytics import ContextPreservingChunker


def test_prefers_paragraph_boundaries_within_chunk_size():
    para1 = "First paragraph stays intact. It has two sentences."
    para2 = "Second paragraph is separate. It also has two sentences."
    para3 = "Third paragraph closes the essay."
    text = f"{para1}\n\n{para2}\n\n{para3}"

    chunker = ContextPreservingChunker(chunk_size=80, overlap_size=10)
    chunks = chunker.chunk_with_context(text, min_chunk_size=1)

    assert len(chunks) >= 2
    assert all(meta.get("split_at") == "paragraph" for _, meta in chunks)
    # Chunks should not glue paragraphs with a single space across \n\n.
    joined = "\n\n".join(c for c, _ in chunks)
    assert para1 in joined
    assert para2 in joined


def test_oversized_paragraph_falls_back_to_sentences():
    long_para = " ".join(f"Sentence number {i} is fairly long here." for i in range(40))
    short_para = "Tiny closer."
    text = f"{long_para}\n\n{short_para}"

    chunker = ContextPreservingChunker(chunk_size=120, overlap_size=10)
    chunks = chunker.chunk_with_context(text, min_chunk_size=1)

    assert len(chunks) >= 2
    assert any("Sentence number" in c for c, _ in chunks)


def test_single_paragraph_still_chunks_by_sentence():
    text = "One. Two. Three. Four. Five."
    chunker = ContextPreservingChunker(chunk_size=20, overlap_size=5)
    chunks = chunker.chunk_with_context(text, min_chunk_size=1)
    assert len(chunks) >= 1
    assert all("split_at" not in meta for _, meta in chunks)

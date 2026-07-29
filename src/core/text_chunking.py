"""
src/core/text_chunking.py
-------------------------
Utilities for splitting raw extracted document text into processable chunks.
"""

from typing import Dict, List


class ChunkString(str):
    def __new__(cls, value, metadata=None):
        obj = super().__new__(cls, value)
        obj.metadata = metadata or {}
        return obj


def _character_fallback_chunking(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """Fallback character-based chunking for non-space or single-token texts (CJK, emojis, long words)."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk:
            chunks.append(ChunkString(chunk))
        if end >= len(text):
            break
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_words: int = 5,
) -> List[str]:
    """
    Splits text into chunks of a target character length with overlapping boundaries.

    Args:
        text: The input text to chunk.
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        min_words: Minimum word count for a chunk to be included. Chunks with
            fewer words are filtered out to reduce noise from headers/page numbers.

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    word_headings = getattr(text, "word_headings", None)

    words = text.split()
    chunks = []
    current_chunk_with_indices = []
    current_length = 0

    for i, word in enumerate(words):
        word_len = len(word) + 1  # include space
        if current_length + word_len > chunk_size and current_chunk_with_indices:
            chunk_str = " ".join(w for w, _ in current_chunk_with_indices)

            metadata = {}
            if word_headings:
                first_word_idx = current_chunk_with_indices[0][1]
                if first_word_idx < len(word_headings) and word_headings[first_word_idx] is not None:
                    metadata["section_title"] = word_headings[first_word_idx]

            if len(chunk_str.split()) >= min_words:
                chunks.append(ChunkString(chunk_str, metadata=metadata))

            # Retain overlap words from the end of the previous chunk
            overlap_words = []
            overlap_len = 0
            for w, idx in reversed(current_chunk_with_indices):
                if overlap_len + len(w) + 1 <= chunk_overlap:
                    overlap_words.insert(0, (w, idx))
                    overlap_len += len(w) + 1
                else:
                    break
            current_chunk_with_indices = overlap_words + [(word, i)]
            current_length = sum(len(w) + 1 for w, _ in current_chunk_with_indices)
        else:
            current_chunk_with_indices.append((word, i))
            current_length += word_len

    if current_chunk_with_indices:
        chunk_str = " ".join(w for w, _ in current_chunk_with_indices)
        metadata = {}
        if word_headings:
            first_word_idx = current_chunk_with_indices[0][1]
            if first_word_idx < len(word_headings) and word_headings[first_word_idx] is not None:
                metadata["section_title"] = word_headings[first_word_idx]
        if len(chunk_str.split()) >= min_words:
            chunks.append(ChunkString(chunk_str, metadata=metadata))

    # Fallback to character-based chunking if no valid word chunks were formed
    # (e.g. single long words, CJK text without spaces, or emoji sequences)
    if not chunks:
        chunks = _character_fallback_chunking(text, chunk_size, chunk_overlap)

    return chunks


# Alias for backward compatibility with src/core/__init__.py
chunk_document = chunk_text


def chunk_documents(
    documents: Dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_words: int = 5,
) -> Dict[str, List[str]]:
    """
    Splits a dictionary of document raw texts into chunks respecting customizable
    chunk size and overlap parameters.
    """
    chunked_docs = {}
    for doc_name, text in documents.items():
        chunked_docs[doc_name] = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_words=min_words,
        )
    return chunked_docs

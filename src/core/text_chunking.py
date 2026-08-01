"""
src/core/text_chunking.py
-------------------------
Utilities for splitting raw extracted document text into processable chunks.

Two strategies are available:

* ``chunk_text``          – fixed character-count chunking with word-boundary
  awareness (original behaviour, preserved for backward compatibility).
* ``chunk_by_sentences``  – sentence-boundary-aware chunking that groups whole
  sentences into blocks up to *max_chunk_size* characters, ensuring no sentence
  is split mid-word or mid-clause.
"""

import re
from typing import Dict, List


# ── Sentence splitting helper ─────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """Return a list of sentences from *text*.

    Tries NLTK ``sent_tokenize`` first.  Falls back to a regex-based splitter
    if NLTK data is unavailable so the function works in restricted environments
    (e.g. CI containers without the punkt corpus downloaded).
    """
    try:
        import nltk
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            if sentences:
                return sentences
        except LookupError:
            # punkt_tab / punkt corpus not downloaded – trigger download once
            try:
                nltk.download("punkt_tab", quiet=True)
                from nltk.tokenize import sent_tokenize
                return sent_tokenize(text)
            except Exception:
                pass
    except ImportError:
        pass

    # Regex fallback: split on sentence-ending punctuation followed by
    # whitespace and an uppercase letter (covers English prose well).
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── ChunkString ───────────────────────────────────────────────────────────────


class ChunkString(str):
    def __new__(cls, value, metadata=None):
        obj = super().__new__(cls, value)
        obj.metadata = metadata or {}
        return obj


# ── Character-level fallback (CJK / emoji / long-word texts) ─────────────────


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


# ── Fixed-size word-boundary chunking (original) ─────────────────────────────


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

    if overlap_percentage is not None:
        chunk_overlap = int(chunk_size * overlap_percentage)

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


# ── Sentence-boundary-aware chunking (Issue #919) ────────────────────────────


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500,
    min_sentences: int = 1,
    min_words: int = 3,
) -> List[str]:
    """Group full sentences into chunk blocks without splitting mid-sentence.

    Sentences are detected via NLTK ``sent_tokenize`` (with a regex fallback
    when NLTK data is unavailable).  Consecutive sentences are accumulated into
    a block until adding the next sentence would exceed *max_chunk_size*
    characters.  When a single sentence is already longer than *max_chunk_size*
    it is emitted as its own chunk rather than being dropped.

    Args:
        text: Raw document text to chunk.
        max_chunk_size: Maximum number of characters per chunk (soft limit –
            a single long sentence may exceed it rather than be discarded).
        min_sentences: Minimum number of sentences required before a block is
            emitted.  Trailing sentences that do not satisfy this minimum are
            still emitted to avoid data loss.
        min_words: Minimum word count to include a chunk.  Filters out
            degenerate fragments such as lone page numbers or headers.

    Returns:
        List of :class:`ChunkString` objects, each containing one or more
        complete sentences.
    """
    if not text or not text.strip():
        return []

    sentences = _split_into_sentences(text.strip())
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_length: int = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # +1 accounts for the joining space between sentences
        added_length = len(sentence) + (1 if current_sentences else 0)

        if current_sentences and current_length + added_length > max_chunk_size:
            # Flush the current block if it meets the minimum sentence count
            block = " ".join(current_sentences)
            if len(current_sentences) >= min_sentences and len(block.split()) >= min_words:
                chunks.append(ChunkString(block))
            elif current_sentences:
                # Below min_sentences threshold – still emit to avoid data loss
                if len(block.split()) >= min_words:
                    chunks.append(ChunkString(block))
            current_sentences = [sentence]
            current_length = len(sentence)
        else:
            current_sentences.append(sentence)
            current_length += added_length

    # Flush the remaining sentences
    if current_sentences:
        block = " ".join(current_sentences)
        if len(block.split()) >= min_words:
            chunks.append(ChunkString(block))

    return chunks


# ── Multi-document helpers ────────────────────────────────────────────────────


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


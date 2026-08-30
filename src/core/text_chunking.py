"""
src/core/text_chunking.py
-------------------------
Utilities for splitting documents into overlapping text chunks optimized
for semantic embedding models.

Provides strategies for chunking text by sentence boundaries, character
limits, and word counts, with overlap support to preserve context across
chunk boundaries.

Two strategies are available:

* ``chunk_text``         – fixed character-count chunking with word-boundary
  awareness and optional sentence-aware padding (Issue #1480).
* ``chunk_by_sentences``  – sentence-boundary-aware chunking that groups whole
  sentences into blocks up to *max_chunk_size* characters, ensuring no sentence
  is split mid-word or mid-clause.

Recent Additions:
- Issue #1480: Added sentence-aware padding to chunk_text() and chunk_documents().
  Chunks now extend to the nearest sentence boundary to prevent cutting
  off semantic context mid-sentence, improving embedding quality.
- Issue #2912: Ensure min_words filtering applies dynamically inside the
  chunking loop to skip expensive padding logic for clearly invalid chunks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

try:
    import nltk  # type: ignore
except ImportError:
    nltk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 50

# Track whether we've already attempted to download the NLTK punkt corpus
_nltk_punkt_checked = False

# Regex pattern to split text into sentences while preserving punctuation.
# Matches periods, exclamation marks, and question marks followed by whitespace
# and an uppercase letter, or at the end of the string. Also matches standard CJK terminators.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$|(?<=[。！？])")

# Regex pattern to identify sentence boundaries.
# Matches '.', '!', or '?' followed by a space and an uppercase letter,
# or followed by the end of the string. Also matches standard CJK terminators.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"([.!?])\s+(?=[A-Z])|([.!?])$|([。！？])")

# Regex pattern to count words (alphanumeric sequences)
_WORD_COUNT_PATTERN = re.compile(r"\b\w+\b")


# ── Helper Functions ──────────────────────────────────────────────────────────


def _chunking_text(text: str) -> str:
    """Return plain text from either a string or a structured DOCX result."""
    structured_text = getattr(text, "text", None)
    return structured_text if isinstance(structured_text, str) else text


def count_words(text: str) -> int:
    """Count the number of words in a text string.

    Args:
        text: The input text string.

    Returns:
        The number of words (alphanumeric sequences) in the text.
    """
    if not text:
        return 0
    return len(_WORD_COUNT_PATTERN.findall(text))


def _split_into_sentences(text: str) -> list[str]:
    """Return a list of sentences from *text*.

    Tries NLTK ``sent_tokenize`` first.  Falls back to a regex-based splitter
    if NLTK data is unavailable so the function works in restricted environments
    (e.g. CI containers without the punkt corpus downloaded).
    """
    if nltk is not None:
        try:
            from nltk.tokenize import sent_tokenize  # type: ignore

            sentences = sent_tokenize(text)
            if sentences:
                return sentences
        except LookupError:
            # punkt_tab / punkt corpus not downloaded – trigger download once
            try:
                nltk.download("punkt_tab", quiet=True)
                from nltk.tokenize import sent_tokenize  # type: ignore

                return sent_tokenize(text)
            except Exception:
                pass

    # Regex fallback: split on sentence-ending punctuation followed by
    # whitespace and an uppercase letter (covers English prose well).
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _align_to_sentence_boundary(
    raw_chunk: str, full_text: str, start_idx: int, end_idx: int
) -> str:
    """Align a raw character window to the nearest sentence boundaries.

    Attempts to extend the chunk to the end of the current sentence if the
    raw window cuts off mid-sentence. This preserves semantic integrity.

    Args:
        raw_chunk: The raw character window extracted from the text.
        full_text: The complete original text document.
        start_idx: The starting character index of the raw chunk.
        end_idx: The ending character index of the raw chunk.

    Returns:
        The aligned chunk string, extended to sentence boundaries where possible.
    """
    # If we're at the end of the document, no alignment needed
    if end_idx >= len(full_text):
        return raw_chunk

    # Check if the raw chunk ends with sentence-ending punctuation
    if re.search(r"[.!?]\s*$", raw_chunk):
        return raw_chunk

    # The chunk cuts off mid-sentence. Try to extend it to the next sentence end.
    # Look ahead in the full_text for the next sentence terminator.
    remaining_text = full_text[end_idx:]
    next_sentence_end = re.search(r"[.!?](?:\s+|$)", remaining_text)

    if next_sentence_end:
        # Extend the chunk to include the rest of the sentence
        extension_length = next_sentence_end.end()
        extended_chunk = raw_chunk + remaining_text[:extension_length]
        return extended_chunk

    # No sentence terminator found in the remaining text (end of document)
    return raw_chunk


# ── ChunkString ───────────────────────────────────────────────────────────────


@dataclass
class ChunkString:
    """Structured text chunk with optional metadata.

    The payload is stored explicitly in ``text`` rather than by subclassing
    ``str``, which makes the type easier for static analyzers, serializers,
    and C-extension boundaries to handle safely.
    """

    text: str
    metadata: dict = field(default_factory=dict)


# ── Character-level fallback (CJK / emoji / long-word texts) ─────────────────


def _is_low_surrogate(char: str) -> bool:
    """Return whether *char* is a UTF-16 low-surrogate code point."""
    return 0xDC00 <= ord(char) <= 0xDFFF


def _is_high_surrogate(char: str) -> bool:
    """Return whether *char* is a UTF-16 high-surrogate code point."""
    return 0xD800 <= ord(char) <= 0xDBFF


def _safe_chunk_start(text: str, index: int) -> int:
    """Move *index* back when it points into a UTF-16 surrogate pair.

    Python normally represents Unicode characters as complete code points, but
    strings containing explicitly encoded UTF-16 surrogate pairs can still be
    encountered at API boundaries. Starting a chunk on a low surrogate would
    split that pair and can make downstream UTF-8 encoding fail.
    """
    if 0 < index < len(text) and _is_low_surrogate(text[index]):
        if _is_high_surrogate(text[index - 1]):
            return index - 1
    return index


def _safe_chunk_end(text: str, index: int) -> int:
    """Move *index* back so a chunk never ends between surrogate code points."""
    if 0 < index < len(text):
        if _is_low_surrogate(text[index]) and _is_high_surrogate(text[index - 1]):
            return index - 1
    return index


def _find_length_capped_end(
    text: str, start: int, limit: int, count_bytes: bool
) -> int:
    """Return a Unicode-safe end index within the requested size limit.

    The returned slice never ends between the two code points of an explicit
    UTF-16 surrogate pair. When *count_bytes* is true, the limit is enforced
    using UTF-8 bytes without ever encoding an isolated surrogate.
    """
    n = len(text)
    start = _safe_chunk_start(text, start)

    if not count_bytes:
        end = _safe_chunk_end(text, min(start + limit, n))
        if end == start and start < n:
            if _is_high_surrogate(text[start]) and start + 1 < n and _is_low_surrogate(text[start + 1]):
                return start + 2
            if not _is_low_surrogate(text[start]):
                return start + 1
        return end

    end = start
    byte_total = 0
    while end < n:
        char = text[end]

        # Treat an explicit surrogate pair as one logical character. This
        # avoids attempting to UTF-8 encode either half independently.
        if _is_high_surrogate(char) and end + 1 < n and _is_low_surrogate(text[end + 1]):
            codepoint = chr(
                0x10000
                + ((ord(char) - 0xD800) << 10)
                + (ord(text[end + 1]) - 0xDC00)
            )
            char_bytes = len(codepoint.encode("utf-8"))
            width = 2
        elif _is_low_surrogate(char):
            # An already-isolated surrogate cannot be safely encoded as UTF-8.
            # Keep it out of normal chunks rather than producing invalid bytes.
            break
        else:
            char_bytes = len(char.encode("utf-8"))
            width = 1

        if byte_total + char_bytes > limit:
            break
        byte_total += char_bytes
        end += width

    if end == start and start < n:
        # A single character/pair larger than the requested byte limit still
        # needs to make progress. Include the complete Unicode unit.
        if _is_high_surrogate(text[start]) and start + 1 < n and _is_low_surrogate(text[start + 1]):
            end = start + 2
        elif not _is_low_surrogate(text[start]):
            end = start + 1

    return _safe_chunk_end(text, end)


def _character_fallback_chunking(
    text: str, chunk_size: int, chunk_overlap: int, count_bytes: bool = False
) -> list[ChunkString]:
    """Fallback character-based chunking for non-space or single-token texts (CJK, emojis, long words)."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(text), step):
        start = _safe_chunk_start(text, start)
        end = _find_length_capped_end(text, start, chunk_size, count_bytes)
        chunk = text[start:end]
        if chunk:
            chunks.append(ChunkString(text=chunk))
        if end >= len(text):
            break
    return chunks


# ── Sentence boundary search helper (Issue #1480) ────────────────────────────


def _find_sentence_boundary(
    text: str,
    index: int,
    direction: str = "backward",
    max_search: int = 150,
) -> int:
    """Find the nearest sentence boundary relative to the given index.

    Args:
        text: The full document text.
        index: The starting index to search from.
        direction: 'backward' to search left, 'forward' to search right.
        max_search: Maximum number of characters to search before giving up.

    Returns:
        The index of the nearest sentence boundary, or the original index
        if no boundary is found within max_search.
    """
    if not text or index < 0 or index >= len(text):
        return index

    if direction == "backward":
        start_idx = max(0, index - max_search)
        search_space = text[start_idx:index]

        # Find the last occurrence of a sentence boundary in the search space
        matches = list(_SENTENCE_BOUNDARY_PATTERN.finditer(search_space))
        if matches:
            last_match = matches[-1]
            # Return the index immediately after the punctuation
            return start_idx + last_match.end()

    elif direction == "forward":
        end_idx = min(len(text), index + max_search)
        search_space = text[index:end_idx]

        matches = list(_SENTENCE_BOUNDARY_PATTERN.finditer(search_space))
        if matches:
            first_match = matches[0]
            # Return the index immediately after the punctuation
            return index + first_match.end()

    # Fallback to original index if no boundary found
    return index


# ── Fixed-size chunking with sentence-aware padding (Issue #1480 & #2912) ───


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_words: int = 10,
    overlap_percentage: float | None = None,
    max_chunks: int = 1000,
    sentence_padding: bool = True,
    count_bytes: bool = False,
    separator: str = " ",
) -> list[ChunkString]:
    """Split text into chunks of a target character length with overlapping boundaries.

    When *sentence_padding* is enabled (default), chunk start and end boundaries
    are extended to the nearest sentence terminator to preserve semantic context.
    This prevents embeddings from being computed over truncated sentences.

    Performance Note (Issue #2912):
        The min_words check is performed immediately after extracting the
        raw character window, before any expensive sentence boundary alignment
        or padding logic is executed. This prevents wasting CPU cycles on
        formatting chunks that will ultimately be discarded.

    Args:
        text: The input text to chunk.
        chunk_size: Target length per chunk (see *count_bytes* for units).
        chunk_overlap: Number of characters to overlap between chunks.
        min_words: Minimum word count for a chunk to be included. Chunks with
            fewer words are filtered out to reduce noise from headers/page numbers.
        overlap_percentage: If provided, overrides *chunk_overlap* as a fraction
            of *chunk_size*.
        max_chunks: Maximum number of chunks to generate. Chunking stops once
            this limit is reached, and a warning is logged, to avoid memory
            spikes on extremely large documents.
        sentence_padding: If True, extends chunk boundaries to the nearest
            sentence terminator to preserve semantic context (Issue #1480).
        count_bytes: If True, *chunk_size* is enforced using UTF-8 byte
            length (len(text.encode('utf-8'))) instead of Unicode code
            points. This gives accurate, strict size enforcement for
            multi-byte characters such as emoji (Issue #2435).
        separator: The character used to join sentences/words. Defaults to space.

    Returns:
        List of chunk strings.
    """
    structured_headings = getattr(text, "headings", None)
    text = _chunking_text(text)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer > 0")

    if overlap_percentage is not None:
        chunk_overlap = int(chunk_size * overlap_percentage)

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    if not text or not text.strip():
        return []

    # Enforce minimum chunk size to prevent infinite loops
    if chunk_size < MIN_CHUNK_SIZE:
        logger.warning(
            "chunk_size %d is too small. Forcing to %d.",
            chunk_size,
            MIN_CHUNK_SIZE,
        )
        chunk_size = MIN_CHUNK_SIZE

    # ── Issue #1390 ───────────────────────────────────────────────────────
    max_chunk_capacity = max_chunks * chunk_size
    if len(text) > max_chunk_capacity:
        logger.warning(
            "Text length (%d chars) exceeded chunk capacity limit; text was truncated",
            len(text),
        )

    text = text.strip()
    text_len = len(text)
    chunks: list[str] = []
    start = 0

    while start < text_len:
        end = _find_length_capped_end(text, start, chunk_size, count_bytes)

        # Extract the raw character window
        raw_chunk = text[start:end]

        # Issue #2912: DYNAMIC MIN_WORDS CHECK
        # Evaluate the word count immediately on the raw window.
        # If it doesn't meet the threshold, skip expensive sentence padding
        # and move to the next window. This optimizes performance by avoiding
        # regex sentence splitting and formatting for invalid chunks.
        raw_word_count = count_words(raw_chunk)

        if raw_word_count < min_words:
            # Chunk is too small. If we're at the end of the text, break.
            if end >= text_len:
                break

            # Otherwise, advance the window and try again.
            step = max(1, chunk_size - chunk_overlap)
            start += step
            continue

        # The chunk meets the minimum word count. Now apply sentence boundary
        # alignment if enabled.
        if sentence_padding:
            # Adjust end to the nearest forward sentence boundary
            if end < text_len:
                end = _find_sentence_boundary(
                    text, end, direction="forward", max_search=100
                )
                # Hard cap to prevent chunks from growing too large for embedding models
                max_allowed_end = _find_length_capped_end(
                    text, start, chunk_size * 2, count_bytes
                )
                if end > max_allowed_end:
                    end = max_allowed_end

            chunk = text[start:end].strip()

            # Final verification after sentence alignment
            final_word_count = count_words(chunk)
            if final_word_count >= min_words:
                chunks.append(ChunkString(text=chunk))
        else:
            # Original word-boundary path (sentence_padding=False)
            word_headings = structured_headings
            words = raw_chunk.split()

            if len(words) >= min_words:
                chunk_str = separator.join(words)
                metadata = {}
                if word_headings:
                    # Approximate heading lookup based on start index
                    # Note: This is a simplified approximation for the non-padding path
                    metadata["section_title"] = None

                chunks.append(ChunkString(text=chunk_str, metadata=metadata))

        if len(chunks) >= max_chunks:
            logger.warning(
                "[text_chunking] Document exceeded max_chunks limit "
                "(%d); truncating remaining chunks.",
                max_chunks,
            )
            return chunks

        if end >= text_len:
            break

        # Calculate next start position with overlap
        next_start = end - chunk_overlap

        # Apply sentence padding to the start of the next chunk
        if sentence_padding and next_start > 0:
            next_start = _find_sentence_boundary(
                text, next_start, direction="backward", max_search=50
            )

        # Prevent infinite loops if sentence padding doesn't advance the pointer
        if next_start <= start:
            next_start = start + chunk_size - chunk_overlap

        start = next_start

    # Fallback to character-based chunking if no valid chunks were formed
    if not chunks:
        chunks = _character_fallback_chunking(
            text, chunk_size, chunk_overlap, count_bytes=count_bytes
        )

    logger.info(
        "chunk_text: Generated %d chunks from %d characters (min_words=%d).",
        len(chunks),
        text_len,
        min_words,
    )
    return [c for c in chunks if len(c.text.split()) >= min_words]


def _merge_undersized_trailing_chunk(
    chunks: list[ChunkString],
    min_chunk_length: int,
) -> list[ChunkString]:
    """Merge a too-short final chunk into the previous one (Issue #4001)."""
    if min_chunk_length <= 0 or len(chunks) < 2:
        return chunks

    last = chunks[-1]
    if len(last.text) >= min_chunk_length:
        return chunks

    prev = chunks[-2]
    merged = ChunkString(
        text=f"{prev.text} {last.text}".strip(),
        metadata=dict(prev.metadata),
    )
    return [*chunks[:-2], merged]


def chunk_document(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_words: int = 10,
    overlap_percentage: float | None = None,
    max_chunks: int = 1000,
    sentence_padding: bool = True,
    count_bytes: bool = False,
    separator: str = " ",
    min_chunk_length: int = 40,
) -> list[ChunkString]:
    """Chunk a single document and fold undersized trailing fragments back.

    ``min_chunk_length`` drops noisy trailer chunks like ``\"Page 12\"`` by
    merging them into the previous chunk when they are shorter than the
    threshold (default 40 characters).
    """
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_words=min_words,
        overlap_percentage=overlap_percentage,
        max_chunks=max_chunks,
        sentence_padding=sentence_padding,
        count_bytes=count_bytes,
        separator=separator,
    )
    return _merge_undersized_trailing_chunk(chunks, min_chunk_length)


# ── Sentence-boundary-aware chunking (Issue #919 & #2054) ────────────────────────────


def chunk_by_sentences(
    text: str,
    max_chunks: int = 1000,
    min_chunk_length: int = 10,
    max_chunk_size: int = 1000,
    min_words: int = 10,
) -> list[str]:
    """Split text into chunks based on natural sentence boundaries.

    Groups consecutive sentences together until the max_chunk_size limit
    is reached, ensuring each resulting chunk meets the min_words threshold.
    This function provides a semantic-aware chunking strategy that respects
    natural sentence boundaries, preventing the mid-sentence truncation
    that can occur with fixed-character chunking.

    Args:
        text: The input text to be chunked.
        max_chunks: Maximum number of chunks to return. Acts as a safety
                   limit to prevent memory exhaustion on extremely large
                   documents. Defaults to 1000. When the limit is reached,
                   the function breaks the loop and logs a warning.
        min_chunk_length: Minimum character length for a chunk to be included.
                         Prevents the creation of tiny, semantically meaningless
                         chunks from fragmented sentences. Defaults to 10.
        max_chunk_size: Maximum number of characters per chunk.
        min_words: Minimum number of words required per chunk.

    Returns:
        A list of string chunks, split by sentence boundaries, respecting
        the max_chunks limit and min_chunk_length threshold.

    Raises:
        ValueError: If max_chunks is less than or equal to 0.

    Examples:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunk_by_sentences(text, max_chunks=2)
        >>> len(chunks)
        2
    """
    if not text or not isinstance(text, str):
        return []

    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be > 0, got {max_chunks}")

    text = text.strip()
    if not text:
        return []

    # Split text into individual sentences
    raw_sentences = _SENTENCE_SPLIT_PATTERN.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk_sentences: list[str] = []
    current_chunk_length = 0

    # Target length for combining short sentences into a single chunk
    # This prevents creating hundreds of 1-word chunks
    target_chunk_length = min(max_chunk_size, 500)

    for sentence in sentences:
        sentence_length = len(sentence) + 1  # +1 for space

        # Check if adding this sentence would exceed target chunk length
        if (
            current_chunk_length + sentence_length > target_chunk_length
            and current_chunk_sentences
        ):
            # Finalize the current chunk
            chunk_text_val = " ".join(current_chunk_sentences)

            # Issue #2912: Apply min_words filter
            if (
                count_words(chunk_text_val) >= min_words
                and len(chunk_text_val) >= min_chunk_length
            ):
                chunks.append(chunk_text_val)

                # Safety limit check (Issue #2054)
                if len(chunks) >= max_chunks:
                    logger.warning(
                        "chunk_by_sentences: Reached max_chunks limit (%d). "
                        "Truncating remaining text to prevent memory exhaustion.",
                        max_chunks,
                    )
                    break

            current_chunk_sentences = []
            current_chunk_length = 0

        current_chunk_sentences.append(sentence)
        current_chunk_length += sentence_length

    # Don't forget the last chunk if we didn't hit the limit
    if current_chunk_sentences and len(chunks) < max_chunks:
        chunk_text_val = " ".join(current_chunk_sentences)
        if (
            count_words(chunk_text_val) >= min_words
            and len(chunk_text_val) >= min_chunk_length
        ):
            chunks.append(chunk_text_val)

    return chunks


# ── Sliding Window Chunk Overlap Optimizer (Issue #1352) ─────────────────────


def chunk_text_dynamic(
    text: str,
    target_size: int = 500,
    min_overlap: int = 50,
    max_chunks: int = 1000,
) -> list[ChunkString]:
    """Dynamically split text into sliding window chunks while preserving sentence boundaries.

    Window boundaries are shifted to the nearest sentence end punctuation ('.', '!', '?')
    when a punctuation mark occurs within 20% of target_size.

    Args:
        text: Raw document text to chunk.
        target_size: Target character length per chunk (default: 500).
        min_overlap: Minimum character overlap between consecutive chunks (default: 50).

    Returns:
        List of ChunkString objects representing sentence-boundary-optimized text chunks.
    """
    if not text or not text.strip():
        return []

    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0")

    clean_src = text.strip()
    n_total = len(clean_src)

    if n_total <= target_size:
        return [ChunkString(text=clean_src)]

    margin = int(target_size * 0.20)
    chunks: list[ChunkString] = []
    start = 0

    sentence_punct = {".", "!", "?"}

    while start < n_total:
        target_end = min(n_total, start + target_size)

        if target_end >= n_total:
            actual_end = n_total
        else:
            # Search for sentence ending punctuation within [target_end - margin, target_end + margin]
            min_search = max(start + min_overlap, target_end - margin)
            max_search = min(n_total, target_end + margin)

            candidate_indices = [
                idx
                for idx in range(min_search, max_search)
                if clean_src[idx] in sentence_punct
            ]

            if candidate_indices:
                # Pick sentence ending punctuation closest to target_end
                best_idx = min(
                    candidate_indices, key=lambda idx: abs((idx + 1) - target_end)
                )
                actual_end = best_idx + 1
            else:
                actual_end = target_end

        chunk_content = clean_src[start:actual_end].strip()
        if chunk_content:
            chunks.append(ChunkString(text=chunk_content))

            if len(chunks) >= max_chunks:
                logger.warning(
                    "Maximum chunk limit reached in chunk_text_dynamic: %d",
                    max_chunks,
                )
                break

        if actual_end >= n_total:
            break

        next_start = actual_end - min_overlap
        if next_start <= start:
            next_start = start + max(1, target_size - min_overlap)
        start = next_start

    return chunks


# ── Multi-document helpers ────────────────────────────────────────────────────


def chunk_documents(
    documents: dict[str, str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_words: int = 10,
    sentence_padding: bool = True,
) -> dict[str, list[str]]:
    """Splits a dictionary of document raw texts into chunks respecting customizable
    chunk size and overlap parameters.

    Args:
        documents: Dictionary mapping document name to raw text.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        min_words: Minimum word count for a chunk to be included.
        sentence_padding: If True, extends chunk boundaries to the nearest
            sentence terminator to preserve semantic context (Issue #1480).

    Returns:
        Dictionary mapping document name to list of chunks.
    """
    chunked_docs = {}
    for doc_name, text in documents.items():
        chunked_docs[doc_name] = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_words=min_words,
            sentence_padding=sentence_padding,
        )
    return chunked_docs

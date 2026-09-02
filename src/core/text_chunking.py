"""
src/core/text_chunking.py
-------------------------
Utilities for splitting documents into overlapping text chunks optimized
for semantic embedding models.

Provides strategies for chunking text by sentence boundaries, character
limits, and word counts, with overlap support to preserve context across
chunk boundaries.

Two strategies are available:

* ``chunk_text``        – fixed character-count chunking with word-boundary
  awareness and optional sentence-aware padding (Issue #1480).
* ``chunk_by_sentences`` – sentence-boundary-aware chunking that groups whole
  sentences into blocks up to *max_chunk_size* characters, ensuring no sentence
  is split mid-word or mid-clause.

Recent Additions:
- Issue #1480: Added sentence-aware padding to chunk_text() and chunk_documents().
  Chunks now extend to the nearest sentence boundary to prevent cutting
  off semantic context mid-sentence, improving embedding quality.
- Issue #2912: Ensure min_words filtering applies dynamically inside the
  chunking loop to skip expensive padding logic for clearly invalid chunks.
- Issue #3997: Enhanced citation masking before sentence splitting to preserve
  author initial periods (e.g., 'J. Doe') and prevent premature splits.
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


# ── Citation Masking Helpers (Issue #3997) ────────────────────────────────────

def mask_citation_initials(text: str) -> str:
    """
    Masks periods in author initials to prevent premature sentence splitting.
    Uses a length-preserving null byte (\x00) to maintain index alignment for
    chunking algorithms that rely on absolute string positions.
    """
    # \b      : word boundary
    # ([A-Z]) : capture a single uppercase letter (Group 1)
    # \.      : match the literal period
    # (?=\s|\(): lookahead to ensure it's followed by a space or opening parenthesis
    return re.sub(r'\b([A-Z])\.(?=\s|\()', lambda m: m.group(1) + '\x00', text)

def unmask_citation_initials(text: str) -> str:
    """Restores the masked periods back to their original state."""
    return text.replace('\x00', '.')


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
    text = mask_citation_initials(text)

    if nltk is not None:
        try:
            from nltk.tokenize import sent_tokenize  # type: ignore

            sentences = sent_tokenize(text)
            if sentences:
                return [unmask_citation_initials(s) for s in sentences]
        except LookupError:
            # punkt_tab / punkt corpus not downloaded – trigger download once
            try:
                nltk.download("punkt_tab", quiet=True)
                from nltk.tokenize import sent_tokenize  # type: ignore

                sentences = sent_tokenize(text)
                return [unmask_citation_initials(s) for s in sentences]
            except Exception:
                pass

    # Regex fallback: split on sentence-ending punctuation followed by
    # whitespace and an uppercase letter (covers English prose well).
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", text.strip())
    return [unmask_citation_initials(p.strip()) for p in parts if p.strip()]


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


# ── Chunk & ChunkString ────────────────────────────────────────────────────────
from typing import Optional


@dataclass
class Chunk:
    """Structured text chunk with position and section metadata (#4002).

    Attributes:
        text: Raw text content of the chunk.
        metadata: Additional arbitrary metadata key-value pairs.
        page_number: Optional 1-based page number where chunk originated.
        char_start: Starting character offset in the source document.
        char_end: Ending character offset in the source document.
        section_title: Optional title/heading of the section containing this chunk.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    page_number: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    section_title: Optional[str] = None

    def __post_init__(self):
        # Synchronize metadata dictionary with primary fields
        if self.page_number is not None and "page_number" not in self.metadata:
            self.metadata["page_number"] = self.page_number
        elif "page_number" in self.metadata and self.page_number is None:
            self.page_number = self.metadata["page_number"]

        if self.char_start != 0 and "char_start" not in self.metadata:
            self.metadata["char_start"] = self.char_start
        elif "char_start" in self.metadata and self.char_start == 0:
            self.char_start = self.metadata["char_start"]

        if self.char_end != 0 and "char_end" not in self.metadata:
            self.metadata["char_end"] = self.char_end
        elif "char_end" in self.metadata and self.char_end == 0:
            self.char_end = self.metadata["char_end"]

        if self.section_title is not None and "section_title" not in self.metadata:
            self.metadata["section_title"] = self.section_title
        elif "section_title" in self.metadata and self.section_title is None:
            self.section_title = self.metadata["section_title"]


ChunkString = Chunk


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
            chunks.append(
                Chunk(
                    text=chunk,
                    char_start=start,
                    char_end=end,
                )
            )
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


# ── Paragraph boundary search helper (Issue #3999) ───────────────────────────


def _find_paragraph_boundary(
    text: str,
    index: int,
    direction: str = "forward",
    max_search: int = 150,
) -> int | None:
    """Find the nearest paragraph boundary (double newline) relative to *index*.

    Splitting at a paragraph boundary preserves semantic completeness better
    than splitting at an arbitrary sentence boundary, since paragraphs are
    already-authored units of meaning.

    Args:
        text: The full document text.
        index: The starting index to search from.
        direction: 'forward' to search right, 'backward' to search left.
        max_search: Maximum number of characters to search before giving up.

    Returns:
        The index immediately after the paragraph break (forward) or
        immediately before it (backward), or None if no paragraph boundary
        is found within max_search.
    """
    if not text or index < 0 or index > len(text):
        return None

    if direction == "forward":
        end_idx = min(len(text), index + max_search)
        search_space = text[index:end_idx]
        pos = search_space.find("\n\n")
        if pos == -1:
            return None
        return index + pos + 2

    start_idx = max(0, index - max_search)
    search_space = text[start_idx:index]
    pos = search_space.rfind("\n\n")
    if pos == -1:
        return None
    return start_idx + pos


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

    if isinstance(text, str):
        text = mask_citation_initials(text)

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
    chunks: list[ChunkString] = []
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
            # Adjust end to the nearest forward boundary. Prefer a paragraph
            # break (Issue #3999) over a plain sentence boundary when one is
            # available within the chunk size target range, since it
            # preserves semantic completeness better than an arbitrary
            # sentence split.
            if end < text_len:
                max_allowed_end = _find_length_capped_end(
                    text, start, chunk_size * 2, count_bytes
                )

                paragraph_end = _find_paragraph_boundary(
                    text, end, direction="forward", max_search=100
                )
                if paragraph_end is not None and paragraph_end <= max_allowed_end:
                    end = paragraph_end
                else:
                    end = _find_sentence_boundary(
                        text, end, direction="forward", max_search=100
                    )

                # Hard cap to prevent chunks from growing too large for embedding models
                if end > max_allowed_end:
                    end = max_allowed_end

            chunk = text[start:end].strip()
            chunk = unmask_citation_initials(chunk)

            # Final verification after sentence alignment
            final_word_count = count_words(chunk)
            if final_word_count >= min_words:
                chunks.append(
                    Chunk(
                        text=chunk,
                        char_start=start,
                        char_end=end,
                        section_title=None,
                    )
                )
        else:
            # Original word-boundary path (sentence_padding=False)
            word_headings = structured_headings
            words = raw_chunk.split()

            if len(words) >= min_words:
                chunk_str = separator.join(words)
                chunk_str = unmask_citation_initials(chunk_str)
                metadata = {}
                section_title = None
                if word_headings:
                    # Approximate heading lookup based on start index
                    metadata["section_title"] = None

                chunks.append(
                    Chunk(
                        text=chunk_str,
                        char_start=start,
                        char_end=end,
                        section_title=section_title,
                        metadata=metadata,
                    )
                )

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

        # Apply padding to the start of the next chunk, preferring a
        # paragraph boundary (Issue #3999) over a sentence boundary.
        if sentence_padding and next_start > 0:
            paragraph_start = _find_paragraph_boundary(
                text, next_start, direction="backward", max_search=50
            )
            if paragraph_start is not None:
                next_start = paragraph_start
            else:
                next_start = _find_sentence_boundary(
                    text, next_start, direction="backward", max_search=50
                )

        # Prevent infinite loops if sentence padding doesn't advance the pointer
        if next_start <= start:
            next_start = start + chunk_size - chunk_overlap

        start = next_start

    # Fallback to character-based chunking if no valid chunks were formed
    if not chunks:
        fallback_chunks = _character_fallback_chunking(
            text, chunk_size, chunk_overlap, count_bytes=count_bytes
        )
        chunks = [
            ChunkString(text=unmask_citation_initials(c.text), metadata=c.metadata) 
            for c in fallback_chunks
        ]

    logger.info(
        "chunk_text: Generated %d chunks from %d characters (min_words=%d).",
        len(chunks),
        text_len,
        min_words,
    )
    return [c for c in chunks if len(c.text.split()) >= min_words]


# Alias for backward compatibility with src/core/__init__.py
chunk_document = chunk_text


# ── Token-aware Chunking (Issue #3998) ───────────────────────────────────────


def chunk_document_by_tokens(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    tokenizer: Optional[Any] = None,
    min_words: int = 0,
) -> list[str]:
    """Splits a document text string into overlapping chunks based on Hugging Face token count.

    Acceptance Criteria (Issue #3998):
    Implement chunk_document_by_tokens(text: str, max_tokens: int = 256, overlap_tokens: int = 32, tokenizer=None)
    using the model's tokenizer.

    Args:
        text: Input raw text string to chunk.
        max_tokens: Maximum number of tokens allowed per chunk (default: 256).
        overlap_tokens: Number of overlapping tokens between consecutive chunks (default: 32).
        tokenizer: Hugging Face tokenizer instance or object with `encode` and `decode` methods.
            If None, attempts to fetch the tokenizer from the active embedding model manager or
            falls back to a default word-tokenization encoder.
        min_words: Optional minimum word count for included chunks (default: 0).

    Returns:
        List of text chunks as strings.

    Raises:
        ValueError: If max_tokens <= 0 or overlap_tokens >= max_tokens or overlap_tokens < 0.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be strictly less than max_tokens")

    raw_text = _chunking_text(text)
    if not raw_text or not raw_text.strip():
        return []

    clean_text = raw_text.strip()

    # Resolve tokenizer if not provided
    if tokenizer is None:
        try:
            from src.core.embedding_model import EmbeddingModelManager

            model = EmbeddingModelManager.get_instance().get_model()
            tokenizer = getattr(model, "tokenizer", None)
        except Exception as exc:
            logger.debug("Could not automatically retrieve model tokenizer: %s", exc)
            tokenizer = None

    # Tokenize input text
    token_ids: list[Any] = []
    if tokenizer is not None:
        if hasattr(tokenizer, "encode"):
            try:
                token_ids = tokenizer.encode(clean_text, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(clean_text)
        elif callable(tokenizer):
            token_ids = tokenizer(clean_text)
        elif hasattr(tokenizer, "tokenize"):
            token_ids = tokenizer.tokenize(clean_text)

    # Fallback to word-based tokens if no tokenizer available or tokenization produced empty results
    if not token_ids:
        words = clean_text.split()
        if not words:
            return []
        token_ids = words
        is_word_fallback = True
    else:
        is_word_fallback = False

    total_tokens = len(token_ids)
    if total_tokens <= max_tokens:
        if min_words > 0 and count_words(clean_text) < min_words:
            return []
        return [clean_text]

    step = max_tokens - overlap_tokens
    chunks: list[str] = []
    start = 0

    while start < total_tokens:
        end = min(total_tokens, start + max_tokens)
        chunk_token_ids = token_ids[start:end]

        if is_word_fallback:
            chunk_str = " ".join(chunk_token_ids).strip()
        else:
            if hasattr(tokenizer, "decode"):
                try:
                    chunk_str = tokenizer.decode(
                        chunk_token_ids, skip_special_tokens=True
                    ).strip()
                except TypeError:
                    chunk_str = tokenizer.decode(chunk_token_ids).strip()
            elif hasattr(tokenizer, "convert_tokens_to_string"):
                chunk_str = tokenizer.convert_tokens_to_string(
                    chunk_token_ids
                ).strip()
            else:
                chunk_str = " ".join(str(t) for t in chunk_token_ids).strip()

        if chunk_str:
            if min_words <= 0 or count_words(chunk_str) >= min_words:
                chunks.append(chunk_str)

        if end >= total_tokens:
            break

        start += step

    return chunks


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
        
    text = mask_citation_initials(text)

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
            chunk_text_val = unmask_citation_initials(chunk_text_val)

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
        chunk_text_val = unmask_citation_initials(chunk_text_val)
        
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

    clean_src = mask_citation_initials(text.strip())
    n_total = len(clean_src)

    if n_total <= target_size:
        return [ChunkString(text=unmask_citation_initials(clean_src))]

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
        chunk_content = unmask_citation_initials(chunk_content)
        
        if chunk_content:
            chunks.append(
                Chunk(
                    text=chunk_content,
                    char_start=start,
                    char_end=actual_end,
                )
            )

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

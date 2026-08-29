"""
src/core/cross_lingual.py
-------------------------
Cross-lingual alignment and back-translation pipeline for detecting
translated plagiarism.

Provides functions to detect the source language of text chunks,
translate them back to the primary corpus language (English), and
verify semantic fidelity using embedding comparisons.

The original source text is never replaced. Only ``embedding_text`` is
translated to English so FAISS vectors for different languages share the same
semantic space.

Recent Additions (Issue #1956, #3696):
- Implemented detect_chunk_language() using lightweight heuristics and thread-safe LRU cache.
- Implemented back_translate_chunk() with SQLite cache integration.
- Implemented verify_semantic_fidelity() for translation quality checks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Callable, Iterable, Optional

import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect_langs

from src.core.translator import (
    translate_text,
    translate_text_batch,
    translate_text_secondary,
)
from src.db.translation_cache import get_cached_translation, save_translation

logger = logging.getLogger(__name__)

# langdetect is non-deterministic by default. A fixed seed makes tests and
# production behaviour repeatable.
DetectorFactory.seed = 0

ENGLISH_CODES = {"en"}
MIN_DETECTION_CHARACTERS = 20

# Target language for back-translation (primary corpus language)
TARGET_LANGUAGE = "en"

# ── Lightweight Language Detection Heuristics (Issue #1956, #2222, #3696) ────────────

# Regex patterns for common stop words and character ranges.
# Avoids heavy dependencies like langdetect for fast chunk-level detection.
_LANGUAGE_HEURISTICS = {
    "es": re.compile(
        r"\b(el|la|los|las|de|del|en|que|es|por|con|para|se|su)\b",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(le|la|les|de|des|un|une|et|est|en|que|qui|dans|ce|il|au|aux)\b",
        re.IGNORECASE,
    ),
    "de": re.compile(
        r"\b(der|die|das|und|ist|von|zu|den|mit|sich|des|auf|für|ein|eine)\b",
        re.IGNORECASE,
    ),
    "it": re.compile(
        r"\b(il|la|le|di|e|che|un|una|in|per|con|da|si|del)\b",
        re.IGNORECASE,
    ),
    "pt": re.compile(
        r"\b(o|a|os|as|de|do|da|em|um|uma|e|que|para|por|com)\b",
        re.IGNORECASE,
    ),
    "zh": re.compile(r"[\u4e00-\u9fff]"),  # CJK Unified Ideographs
    "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"),  # Hiragana/Katakana
    "ar": re.compile(r"[\u0600-\u06ff]"),  # Arabic script
    "hi": re.compile(r"[\u0900-\u097f]"),  # Devanagari script (Hindi)
}

# Thread-safe LRU Cache for Language Detection
_DETECT_LANG_CACHE_LOCK = RLock()
_DETECT_LANG_CACHE: OrderedDict[str, str] = OrderedDict()
_DETECT_LANG_CACHE_MAXSIZE = 1024


def detect_chunk_language(text: str) -> str:
    """Detect the likely language of a text chunk using lightweight heuristics.

    This function uses regex matching against common stop words and Unicode
    character ranges to identify the language without requiring heavy NLP
    models or external API calls. Results are cached using a thread-safe
    LRU cache to speed up multi-lingual processing.

    Args:
        text: The input text chunk to analyze.

    Returns:
        ISO 639-1 language code (e.g., 'es', 'fr', 'zh') or 'en' as default.
    """
    if not text or not isinstance(text, str):
        return TARGET_LANGUAGE

    # Generate SHA-256 hash for cache key
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Check cache first
    with _DETECT_LANG_CACHE_LOCK:
        if text_hash in _DETECT_LANG_CACHE:
            _DETECT_LANG_CACHE.move_to_end(text_hash)
            return _DETECT_LANG_CACHE[text_hash]

    detected_lang = TARGET_LANGUAGE
    cjk_found = False

    # Check for script-based languages first as they are highly distinctive
    for lang, pattern in _LANGUAGE_HEURISTICS.items():
        if lang in ("zh", "ja", "ar", "hi"):
            if pattern.search(text):
                detected_lang = lang
                cjk_found = True
                break

    if not cjk_found:
        # Count stop word matches for European languages
        matches: dict[str, int] = {}
        words = text.lower().split()
        total_words = len(words)

        if total_words >= 3:
            for lang, pattern in _LANGUAGE_HEURISTICS.items():
                if lang in ("zh", "ja", "ar", "hi"):
                    continue
                count = len(pattern.findall(text))
                matches[lang] = count

            # Find the language with the highest stop word density
            if matches:
                best_lang = max(matches, key=matches.get)
                # Require at least 15% of words to be stop words to avoid false positives
                if matches[best_lang] / total_words > 0.15:
                    detected_lang = best_lang

    # Update cache securely
    with _DETECT_LANG_CACHE_LOCK:
        if text_hash in _DETECT_LANG_CACHE:
            _DETECT_LANG_CACHE.move_to_end(text_hash)
        else:
            if len(_DETECT_LANG_CACHE) >= _DETECT_LANG_CACHE_MAXSIZE:
                _DETECT_LANG_CACHE.popitem(last=False)
        _DETECT_LANG_CACHE[text_hash] = detected_lang

    return detected_lang


# ── Untranslatable Content Detection (Issue #3693) ──────────────────────────

# Content that is expected to remain identical after back-translation
# (numbers, URLs, code, acronyms, file paths, ...). For such chunks the
# "suspiciously identical" warning is misleading and should be suppressed.
_UNTRANSLATABLE_PATTERNS = (
    r"^[\d.,:\-/\\+%$€£¥₹# ]+$",          # pure numbers, dates, currency
    r"^https?://\S+$",                    # URLs
    r"^[\w.+-]+@[\w-]+\.[\w.-]+$",        # email addresses
    r"^[A-Z](?:\.?[A-Z]){1,5}\.?$",        # short acronyms (API, HTTP, U.S.A.)
    r"^[\w./\\-]+\.[a-zA-Z]{1,5}$",       # file paths / extensions
    r"^[a-z][a-zA-Z0-9]*(?:_[a-z][a-zA-Z0-9]*)*$",  # snake_case identifiers
    r"^[a-z][a-zA-Z0-9]*(?:\.[a-z][a-zA-Z0-9]*)+$", # dotted / camelCase identifiers
)
_UNTRANSLATABLE_RE = [re.compile(p, re.IGNORECASE) for p in _UNTRANSLATABLE_PATTERNS]


def _is_untranslatable_content(text: str) -> bool:
    """Return ``True`` when *text* is expected to survive back-translation unchanged.

    Handles numbers, dates, currency, URLs, email addresses, acronyms, file
    paths/extensions and code-like identifiers (camelCase, snake_case, dotted).
    Also returns ``True`` for text whose alphabetic ratio is below 30% — a
    strong signal of machine-formatted rather than natural-language content.

    Used by :func:`back_translate_chunk` to suppress the noisy
    "suspiciously identical" warning for Issue #3693.
    """
    if not text or not isinstance(text, str):
        return True

    stripped = text.strip()
    if not stripped:
        return True

    for pattern in _UNTRANSLATABLE_RE:
        if pattern.match(stripped):
            return True

    # Predominantly non-letter characters => code, symbols, or numbers.
    letter_count = len(re.findall(r"[a-zA-Z]", stripped))
    total_count = len(stripped)
    if total_count > 0 and letter_count / total_count < 0.3:
        return True

    return False


# ── Back-Translation with SQLite Cache (Issue #1956) ─────────────────────────


def back_translate_chunk(
    text: str,
    source_lang: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    """Translate a text chunk back to the target language (English).

    This function first checks the SQLite translation cache to avoid
    redundant API/model calls. If not cached, it calls the real translation
    service via translate_text() and saves the result to cache.

    Args:
        text: The source text chunk to translate.
        source_lang: Optional source language code. If None, auto-detected.
        use_cache: Whether to check and update the SQLite cache.

    Returns:
        The back-translated text string in the target language.
        Falls back to original text if translation fails.
    """
    if not text or not isinstance(text, str):
        return ""

    if source_lang is None:
        source_lang = detect_chunk_language(text)

    # If already in target language, no translation needed
    if source_lang == TARGET_LANGUAGE:
        return text

    # Check cache first
    if use_cache:
        cached = get_cached_translation(text, source_lang, TARGET_LANGUAGE)
        if cached:
            logger.debug(
                "Cache hit for translation: %s -> %s", source_lang, TARGET_LANGUAGE
            )
            return cached

    # Perform real translation using translate_text() (Issue #2219)
    logger.info(
        "Translating chunk from %s to %s (Cache miss, executing translation).",
        source_lang,
        TARGET_LANGUAGE,
    )

    try:
        translated_text = translate_text(
            text,
            target_lang=TARGET_LANGUAGE,
            source_lang=source_lang,
        )
        
        # Validate translation result
        if not translated_text or not isinstance(translated_text, str):
            logger.warning(
                "Translation returned empty or invalid result for %s -> %s. "
                "Falling back to original text.",
                source_lang,
                TARGET_LANGUAGE,
            )
            return text
            
        translated_text = translated_text.strip()
        
        # Check if translation is suspiciously identical to source
        # (could indicate translation service failure). Skip the warning for
        # content that is expected to remain unchanged (Issue #3693).
        if (
            translated_text == text.strip()
            and source_lang != TARGET_LANGUAGE
            and not _is_untranslatable_content(text.strip())
        ):
            logger.warning(
                "Translation service returned identical text for %s -> %s. "
                "This may indicate a translation failure.",
                source_lang,
                TARGET_LANGUAGE,
            )
        
    except Exception as exc:
        logger.error(
            "Translation failed for %s -> %s: %s. Attempting fallback.",
            source_lang,
            TARGET_LANGUAGE,
            exc,
        )
        # Route to secondary translator if enabled (resilience layer)
        if os.getenv("SECONDARY_TRANSLATOR_ENABLED", "false").lower() == "true":
            try:
                logger.info("Routing translation request to secondary offline model...")
                translated_text = translate_text_secondary(
                    text,
                    target_lang=TARGET_LANGUAGE,
                    source_lang=source_lang,
                )
                if translated_text:
                    if use_cache:
                        save_translation(text, source_lang, TARGET_LANGUAGE, translated_text)
                    return translated_text
            except Exception as sec_exc:
                logger.error("Secondary offline model translation also failed: %s", sec_exc)
        return text

    # Save to cache
    if use_cache:
        save_translation(text, source_lang, TARGET_LANGUAGE, translated_text)

    return translated_text


def back_translate_chunks(
    chunks: list[str],
    source_lang: Optional[str] = None,
) -> list[str]:
    """Translate a list of document text chunks back to English.

    Uncached chunks are batched into groups of 10 to minimize translation
    API round-trips.

    Args:
        chunks: List of source text chunks to translate.
        source_lang: Optional source language code. If None, auto-detected per chunk.

    Returns:
        List of translated text strings matching original index sequence.
    """
    if not chunks:
        return []

    # Initialize results list with placeholders
    results: list[Optional[str]] = [None] * len(chunks)

    # Store indices and text of uncached chunks grouped by source language
    # structure: {lang: [(original_index, chunk_text), ...]}
    uncached_by_lang: dict[str, list[tuple[int, str]]] = {}

    for idx, chunk in enumerate(chunks):
        if not chunk or not isinstance(chunk, str):
            results[idx] = ""
            continue

        # Determine source language
        lang = source_lang if source_lang is not None else detect_chunk_language(chunk)

        # If it's already English, no translation needed
        if lang == TARGET_LANGUAGE:
            results[idx] = chunk
            continue

        # Check SQLite cache
        cached = get_cached_translation(chunk, lang, TARGET_LANGUAGE)
        if cached:
            results[idx] = cached
        else:
            if lang not in uncached_by_lang:
                uncached_by_lang[lang] = []
            uncached_by_lang[lang].append((idx, chunk))

    # Process uncached chunks by language in batches of 10
    BATCH_SIZE = 10

    for lang, items in uncached_by_lang.items():
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            batch_texts = [item[1] for item in batch]

            try:
                translated_batch = translate_text_batch(
                    batch_texts,
                    target_lang=TARGET_LANGUAGE,
                    source_lang=lang,
                )
            except Exception as exc:
                logger.error(
                    "Batch translation failed for %s -> %s: %s. Falling back to individual translations.",
                    lang,
                    TARGET_LANGUAGE,
                    exc,
                )
                translated_batch = []
                for text in batch_texts:
                    try:
                        translated_batch.append(
                            translate_text(text, target_lang=TARGET_LANGUAGE, source_lang=lang)
                        )
                    except Exception:
                        translated_batch.append(text)

            # Assign results and save to cache
            for (idx, original_text), translated_text in zip(batch, translated_batch):
                if (
                    not translated_text
                    or not isinstance(translated_text, str)
                    or translated_text.lower().startswith("(translation error")
                ):
                    translated_text = original_text
                else:
                    translated_text = translated_text.strip()
                    save_translation(original_text, lang, TARGET_LANGUAGE, translated_text)

                results[idx] = translated_text

    # Any remaining None gets original chunk
    for idx, res in enumerate(results):
        if res is None:
            results[idx] = chunks[idx]

    return results


# ── Semantic Fidelity Verification (Issue #1956) ─────────────────────────────


def verify_semantic_fidelity(
    original_embedding: np.ndarray,
    translated_embedding: np.ndarray,
) -> float:
    """Verify that the back-translation preserved the original semantic meaning.

    Computes the cosine similarity between the original chunk's embedding
    and the back-translated chunk's embedding. A low score indicates the
    translation significantly altered the meaning (potential translation error
    or highly idiomatic text).

    Args:
        original_embedding: Embedding vector of the source text.
        translated_embedding: Embedding vector of the back-translated text.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    if original_embedding is None or translated_embedding is None:
        return 0.0

    if original_embedding.size == 0 or translated_embedding.size == 0:
        return 0.0

    # Ensure vectors are 1D and normalized
    vec_a = original_embedding.flatten()
    vec_b = translated_embedding.flatten()

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return float(np.clip(cosine_sim, 0.0, 1.0))


# ── In-Memory Translation Cache (Original) ───────────────────────────────────


class TranslationMemoryCache:
    """Thread-safe in-memory LRU cache for translated sentences.

    Cache keys are SHA-256 hashes of the source language, target language,
    and exact source sentence. Including both language codes prevents a
    translation produced for one language pair from being reused for another.

    A configurable *maxsize* (default 512) caps memory usage by evicting the
    least-recently-used entry whenever the cache is full (Issue #2221).
    """

    def __init__(self, maxsize: int = 512) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self._translations: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize
        self._lock = RLock()

    @staticmethod
    def _build_key(
        sentence: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        """Return a deterministic SHA-256 key for a translation request."""
        payload = (
            f"{_normalise_language_code(source_lang)}\0"
            f"{_normalise_language_code(target_lang)}\0"
            f"{sentence}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        sentence: str,
        *,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str | None:
        """Return a cached translation or ``None`` when absent.

        Moves the accessed entry to the most-recently-used position.
        """
        key = self._build_key(sentence, source_lang, target_lang)
        with self._lock:
            if key not in self._translations:
                return None
            self._translations.move_to_end(key)
            return self._translations[key]

    def set(
        self,
        sentence: str,
        translation: str,
        *,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> None:
        """Store a successful non-empty translation.

        Evicts the least-recently-used entry when the cache is at capacity.
        """
        normalized_translation = str(translation or "").strip()
        if not normalized_translation:
            return
        key = self._build_key(sentence, source_lang, target_lang)
        with self._lock:
            if key in self._translations:
                self._translations.move_to_end(key)
            else:
                if len(self._translations) >= self._maxsize:
                    self._translations.popitem(last=False)
            self._translations[key] = normalized_translation

    def clear(self) -> None:
        """Remove every cached translation."""
        with self._lock:
            self._translations.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._translations)


TRANSLATION_MEMORY_CACHE = TranslationMemoryCache()


# ── PreparedText Dataclass ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreparedText:
    """Original text plus the aligned text used to build an embedding."""

    original_text: str
    embedding_text: str
    detected_language: str
    translated: bool
    translation_failed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a backwards-compatible dictionary for existing callers."""
        return asdict(self)


# ── Language Detection Helpers ────────────────────────────────────────────────


def _normalise_language_code(language: str | None) -> str:
    code = (language or "unknown").strip().lower().replace("_", "-")
    return code.split("-", 1)[0] or "unknown"


def detect_language(text: str, min_confidence: float = 0.8) -> tuple[str, bool]:
    """Detect an ISO 639-1 language code and return a (detected_lang, is_confident) tuple.

    Empty, very short, numeric, or otherwise undetectable text defaults to
    "en" with is_confident=False instead of raising an exception.
    """
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) < MIN_DETECTION_CHARACTERS:
        return "en", False

    if not any(character.isalpha() for character in cleaned):
        return "en", False

    try:
        langs = detect_langs(cleaned)
        if not langs:
            return "en", False
        top_lang = langs[0]
        detected_lang = _normalise_language_code(top_lang.lang)
        confidence = top_lang.prob

        if confidence < min_confidence:
            logger.warning(
                "Low-confidence language detection (%.4f < %.2f) for text: %s. Defaulting to 'en'.",
                confidence,
                min_confidence,
                cleaned[:50],
            )
            return "en", False

        return detected_lang, True
    except (LangDetectException, ValueError, TypeError) as e:
        logger.warning("Language detection failed: %s. Defaulting to 'en'.", e)
        return "en", False


# ── Text Preparation Pipeline ────────────────────────────────────────────────


def prepare_text_for_embedding(
    text: str,
    *,
    detector: Callable[[str], str | tuple[str, bool]] | None = None,
    translator: Callable[..., str] | None = None,
    translation_cache: TranslationMemoryCache | None = None,
) -> dict[str, object]:
    """Prepare one source paragraph for English-aligned embedding.

    Parameters are injectable to make behaviour deterministic in tests and to
    avoid network translation calls during unit tests.

    The returned ``original_text`` always matches the input. When translation
    fails, ``embedding_text`` safely falls back to the original text.
    """
    original_text = str(text or "")
    if not original_text.strip():
        return PreparedText(
            original_text=original_text,
            embedding_text=original_text,
            detected_language="unknown",
            translated=False,
        ).to_dict()

    detector_fn = detector or detect_language
    translator_fn = translator or translate_text
    cache = (
        translation_cache if translation_cache is not None else TRANSLATION_MEMORY_CACHE
    )

    try:
        res = detector_fn(original_text)
        if isinstance(res, tuple):
            detected_lang, _ = res
        else:
            detected_lang = res
        language = _normalise_language_code(detected_lang)
    except (LangDetectException, ValueError, TypeError) as exc:
        logger.warning("Language detection failed, defaulting to 'en': %s", exc)
        language = "en"
    if language in ENGLISH_CODES or language == "unknown":
        return PreparedText(
            original_text=original_text,
            embedding_text=original_text,
            detected_language=language,
            translated=False,
        ).to_dict()

    target_language = "en"
    cached_translation = cache.get(
        original_text,
        source_lang=language,
        target_lang=target_language,
    )

    if cached_translation is not None:
        translated_text = cached_translation
    else:
        try:
            translated_text = translator_fn(
                original_text,
                target_lang=target_language,
                source_lang=language,
            )
        except TypeError:
            # Backward compatibility with the repository's previous
            # translator signature: translate_text(text, target_lang="en").
            try:
                translated_text = translator_fn(
                    original_text,
                    target_lang=target_language,
                )
            except (TypeError, ValueError, ConnectionError) as exc:
                logger.warning("Fallback translation call failed: %s", exc)
                translated_text = ""
        except (TypeError, ValueError, ConnectionError) as exc:
            logger.warning("Translation call failed: %s", exc)
            translated_text = ""
    translated_text = str(translated_text or "").strip()
    translation_failed = not translated_text or translated_text.lower().startswith(
        "(translation error"
    )

    if translation_failed:
        return PreparedText(
            original_text=original_text,
            embedding_text=original_text,
            detected_language=language,
            translated=False,
            translation_failed=True,
        ).to_dict()

    if cached_translation is None:
        cache.set(
            original_text,
            translated_text,
            source_lang=language,
            target_lang=target_language,
        )

    return PreparedText(
        original_text=original_text,
        embedding_text=translated_text,
        detected_language=language,
        translated=True,
    ).to_dict()


def prepare_chunks_for_embedding(
    chunks: Iterable[str],
) -> tuple[list[str], list[dict[str, object]]]:
    """Prepare a sequence of chunks while preserving original display text.

    Returns:
        ``(embedding_chunks, metadata)`` where ``embedding_chunks`` contains
        English-aligned text and ``metadata`` records language/translation state.
    """
    embedding_chunks: list[str] = []
    metadata: list[dict[str, object]] = []

    for chunk in chunks:
        prepared = prepare_text_for_embedding(chunk.text if hasattr(chunk, "text") else chunk)
        embedding_chunks.append(str(prepared["embedding_text"]))
        metadata.append(prepared)

    return embedding_chunks, metadata


def prepare_documents_for_embedding(
    chunked_documents: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, object]]]]:
    """Prepare every document's chunks for embedding without mutating originals."""
    translated_documents: dict[str, list[str]] = {}
    alignment_metadata: dict[str, list[dict[str, object]]] = {}

    for document_name, chunks in chunked_documents.items():
        embedding_chunks, metadata = prepare_chunks_for_embedding(chunks)
        translated_documents[document_name] = embedding_chunks
        alignment_metadata[document_name] = metadata

    return translated_documents, alignment_metadata

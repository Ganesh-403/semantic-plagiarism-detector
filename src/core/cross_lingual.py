"""Cross-lingual preprocessing for semantic plagiarism alignment.

The original source text is never replaced.  Only ``embedding_text`` is
translated to English so FAISS vectors for different languages share the same
semantic space.
"""

from __future__ import annotations

import hashlib
import logging
from threading import RLock
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from langdetect import DetectorFactory, LangDetectException, detect_langs

logger = logging.getLogger(__name__)

from src.core.translator import translate_text

# langdetect is non-deterministic by default. A fixed seed makes tests and
# production behaviour repeatable.
DetectorFactory.seed = 0

ENGLISH_CODES = {"en"}
MIN_DETECTION_CHARACTERS = 20



class TranslationMemoryCache:
    """Thread-safe in-memory cache for translated sentences.

    Cache keys are SHA-256 hashes of the source language, target language,
    and exact source sentence. Including both language codes prevents a
    translation produced for one language pair from being reused for another.
    """

    def __init__(self) -> None:
        self._translations: dict[str, str] = {}
        self._lock = RLock()

    @staticmethod
    def _build_key(
        sentence: str,
        source_lang: str,
        target_lang: str,
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
        source_lang: str,
        target_lang: str,
    ) -> str | None:
        """Return a cached translation or ``None`` when absent."""
        key = self._build_key(
            sentence,
            source_lang,
            target_lang,
        )
        with self._lock:
            return self._translations.get(key)

    def set(
        self,
        sentence: str,
        translation: str,
        *,
        source_lang: str,
        target_lang: str,
    ) -> None:
        """Store a successful non-empty translation."""
        normalized_translation = str(translation or "").strip()
        if not normalized_translation:
            return

        key = self._build_key(
            sentence,
            source_lang,
            target_lang,
        )
        with self._lock:
            self._translations[key] = normalized_translation

    def clear(self) -> None:
        """Remove every cached translation."""
        with self._lock:
            self._translations.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._translations)


TRANSLATION_MEMORY_CACHE = TranslationMemoryCache()


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
                cleaned[:50]
            )
            return "en", False

        return detected_lang, True
    except (LangDetectException, ValueError, TypeError) as e:
        logger.warning("Language detection failed: %s. Defaulting to 'en'.", e)
        return "en", False


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

    The returned ``original_text`` always matches the input.  When translation
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
        translation_cache
        if translation_cache is not None
        else TRANSLATION_MEMORY_CACHE
    )

    try:
        res = detector_fn(original_text)
        if isinstance(res, tuple):
            detected_lang, _ = res
        else:
            detected_lang = res
        language = _normalise_language_code(detected_lang)
    except Exception:
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
            # translator signature:
            # translate_text(text, target_lang="en").
            try:
                translated_text = translator_fn(
                    original_text,
                    target_lang=target_language,
                )
            except Exception:
                translated_text = ""
        except Exception:
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
        prepared = prepare_text_for_embedding(chunk)
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

+--- a/src/core/cross_lingual.py
+@@ -10,7 +10,7 @@ class CrossLingualTranslator:
+     def translate(self, text, source_lang, target_lang):
+-        translated_text = "Translated: " + text
++        translated_text, confidence = self._translate_with_confidence(text, source_lang, target_lang)
+         return translated_text
+ 
+-    def _translate(self, text, source_lang, target_lang):
++    def _translate_with_confidence(self, text, source_lang, target_lang):
+         # Placeholder for actual translation logic
+         return "Translated: " + text, 0.95
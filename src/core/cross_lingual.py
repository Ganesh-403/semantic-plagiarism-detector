"""Cross-lingual preprocessing for semantic plagiarism alignment.

The original source text is never replaced.  Only ``embedding_text`` is
translated to English so FAISS vectors for different languages share the same
semantic space.
"""

from __future__ import annotations

import logging
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

        if confidence < 0.7:
            logger.warning(
                "Low language detection confidence (%.2f) for input text snippet",
                confidence,
            )

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


def translate_to_english(
    text: str,
    *,
    detector: Callable[[str], str | tuple[str, bool]] | None = None,
    translator: Callable[..., str] | None = None,
) -> dict[str, str | float]:
    """Translates source text into English and returns confidence alongside metadata.

    Returns:
        dict: {
            "translated_text": str,
            "source_language": str,
            "confidence": float
        }
    """
    original_text = str(text or "")
    if not original_text.strip():
        return {
            "translated_text": original_text,
            "source_language": "en",
            "confidence": 1.0,
        }

    detector_fn = detector or detect_language
    translator_fn = translator or translate_text

    source_lang = "en"
    confidence = 1.0

    try:
        cleaned = " ".join(original_text.split())
        if len(cleaned) >= MIN_DETECTION_CHARACTERS and any(c.isalpha() for c in cleaned):
            langs = detect_langs(cleaned)
            if langs:
                top_lang = langs[0]
                source_lang = _normalise_language_code(top_lang.lang)
                confidence = float(top_lang.prob)
    except Exception:
        res = detector_fn(original_text)
        if isinstance(res, tuple):
            source_lang, _ = res
        else:
            source_lang = res
        source_lang = _normalise_language_code(source_lang)

    # If text is already in English, return confidence = 1.0
    if source_lang in ENGLISH_CODES or source_lang == "unknown":
        return {
            "translated_text": original_text,
            "source_language": "en" if source_lang in ENGLISH_CODES else source_lang,
            "confidence": 1.0,
        }

    # Perform translation for non-English source text
    try:
        translated_text = translator_fn(
            original_text,
            target_lang="en",
            source_lang=source_lang,
        )
    except TypeError:
        try:
            translated_text = translator_fn(original_text, target_lang="en")
        except Exception:
            translated_text = original_text
            confidence = 0.0
    except Exception:
        translated_text = original_text
        confidence = 0.0

    translated_text_str = str(translated_text or "").strip()
    if not translated_text_str or translated_text_str.lower().startswith("(translation error"):
        return {
            "translated_text": original_text,
            "source_language": source_lang,
            "confidence": 0.0,
        }

    return {
        "translated_text": translated_text_str,
        "source_language": source_lang,
        "confidence": round(confidence, 4),
    }


def prepare_text_for_embedding(
    text: str,
    *,
    detector: Callable[[str], str | tuple[str, bool]] | None = None,
    translator: Callable[..., str] | None = None,
) -> dict[str, object]:
    """Prepare one source paragraph for English-aligned embedding."""
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

    try:
        translated_text = translator_fn(
            original_text,
            target_lang="en",
            source_lang=language,
        )
    except TypeError:
        try:
            translated_text = translator_fn(original_text, target_lang="en")
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

    return PreparedText(
        original_text=original_text,
        embedding_text=translated_text,
        detected_language=language,
        translated=True,
    ).to_dict()


def prepare_chunks_for_embedding(
    chunks: Iterable[str],
) -> tuple[list[str], list[dict[str, object]]]:
    """Prepare a sequence of chunks while preserving original display text."""
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

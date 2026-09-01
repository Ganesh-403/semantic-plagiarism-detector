"""Language-aware similarity configuration for plagiarism detection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from src.core.cross_lingual import detect_language
from src.core.embedding_model import _get_model_name
from src.utils.language_utils import VALID_LANGUAGE_CODES


@dataclass(frozen=True)
class LanguageSimilarityPolicy:
    """Scoring policy selected for a source/target language pair."""

    source_language: str
    target_language: str
    same_language: bool
    cross_lingual: bool
    detection_confident: bool
    lexical_processing_available: bool
    embedding_compatible: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_document_language(text: str) -> tuple[str, bool]:
    """Detect a document language while safely handling low confidence."""
    language, confident = detect_language(text)

    if not confident or language not in VALID_LANGUAGE_CODES:
        return "unknown", False

    return language, True


def build_language_metadata(
    documents: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build language metadata once for every processed document."""
    metadata: dict[str, dict[str, Any]] = {}

    for document_name, text in documents.items():
        language, confident = detect_document_language(text)

        metadata[document_name] = {
            "language": language,
            "language_confident": confident,
        }

    return metadata


def get_language_pair_policy(
    source_language: str,
    target_language: str,
    *,
    detection_confident: bool = True,
    base_threshold: float = 0.75,
    embedding_model_name: str | None = None,
) -> LanguageSimilarityPolicy:
    """Return the configurable scoring policy for a language pair.

    Cross-language thresholds can be configured with
    ``PLAGIARISM_CROSS_LANGUAGE_THRESHOLD``. Same-language comparisons use
    the caller-provided base threshold.

    Unknown or low-confidence languages safely use the base threshold.
    """
    source = (source_language or "unknown").lower()
    target = (target_language or "unknown").lower()

    same_language = (
        source != "unknown"
        and target != "unknown"
        and source == target
    )
    cross_lingual = (
        source != "unknown"
        and target != "unknown"
        and source != target
    )

    model_name = embedding_model_name or _get_model_name()
    embedding_compatible = (
        not cross_lingual
        or "multilingual" in model_name.lower()
    )

    threshold = float(base_threshold)

    if cross_lingual:
        configured_threshold = os.getenv(
            "PLAGIARISM_CROSS_LANGUAGE_THRESHOLD"
        )
        if configured_threshold is not None:
            try:
                threshold = max(
                    0.0,
                    min(1.0, float(configured_threshold)),
                )
            except (TypeError, ValueError):
                threshold = float(base_threshold)

    lexical_processing_available = (
        same_language
        and source in VALID_LANGUAGE_CODES
    )

    return LanguageSimilarityPolicy(
        source_language=source,
        target_language=target,
        same_language=same_language,
        cross_lingual=cross_lingual,
        detection_confident=detection_confident,
        lexical_processing_available=lexical_processing_available,
        embedding_compatible=embedding_compatible,
        threshold=threshold,
    )
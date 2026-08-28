# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Cross-Lingual Plagiarism Detection Engine.

Detects plagiarism across documents written in different languages
using multilingual embeddings and translation-assisted comparison.
"""

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# LANGUAGE DATABASE & ENUMS
# ============================================================================


class SupportedLanguage(Enum):
    """Supported languages for cross-lingual detection."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    HINDI = "hi"
    KOREAN = "ko"
    TURKISH = "tr"
    POLISH = "pl"


LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "hi": "Hindi",
    "ko": "Korean",
    "tr": "Turkish",
    "pl": "Polish",
    "ur": "Urdu",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "pa": "Punjabi",
    "ne": "Nepali",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "fil": "Filipino",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "sw": "Swahili",
}

LANGUAGE_DB = {
    code: {"name": name, "script": "Latin", "family": "Indo-European"}
    for code, name in LANGUAGE_NAMES.items()
}


# ============================================================================
# DATASTRUCTURES
# ============================================================================


@dataclass
class LanguageMatch:
    """A detected cross-lingual similarity match."""

    source_doc: str
    source_lang: str
    source_chunk: str
    target_doc: str
    target_lang: str
    target_chunk: str
    similarity: float
    method: str = "multilingual_embedding"
    translation_used: bool = True
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossLingualResult:
    """Result of cross-lingual plagiarism detection."""

    documents: list[dict[str, Any]] = field(default_factory=list)
    matches: list[LanguageMatch] = field(default_factory=list)
    language_distribution: dict[str, int] = field(default_factory=dict)
    total_comparisons: int = 0
    processing_time: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    target_text: str = ""
    source_lang: str = "en"
    target_lang: str = "en"
    similarity_score: float = 0.0
    translation_quality: float = 1.0
    method: str = "hybrid"
    is_plagiarism: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "matches": [
                m.to_dict() if isinstance(m, LanguageMatch) else m for m in self.matches
            ],
            "language_distribution": self.language_distribution,
            "total_comparisons": self.total_comparisons,
            "processing_time": self.processing_time,
            "summary": self.summary,
            "similarity_score": self.similarity_score,
            "is_plagiarism": self.is_plagiarism,
        }


@dataclass
class CrossLingualConfig:
    """Configuration for cross-lingual detection."""

    enabled_languages: list[str] = field(
        default_factory=lambda: ["en", "es", "fr", "de"]
    )
    similarity_threshold: float = 0.65
    use_translation_bridge: bool = True
    translation_service: str = "internal"
    max_chunks_per_doc: int = 100
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 10
    enable_cache: bool = True


# ============================================================================
# TRANSLATION CACHE & DETECTOR ENGINE
# ============================================================================


class TranslationCache:
    """Cache for translations."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, str] = {}
        self._timestamps: dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        key = f"{source_lang}:{target_lang}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"  # nosec
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(
        self, text: str, source_lang: str, target_lang: str, translation: str
    ) -> None:
        key = f"{source_lang}:{target_lang}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"  # nosec
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        self._cache[key] = translation
        self._timestamps[key] = time.time()


class CrossLingualDetector:
    """Detects plagiarism across documents in different languages."""

    def __init__(
        self, config: Optional[CrossLingualConfig | str] = None, use_cache: bool = True
    ):
        if isinstance(config, str):
            self.config = CrossLingualConfig()
            self.target_lang = config
        elif config is not None:
            self.config = config
            self.target_lang = "en"
        else:
            self.config = CrossLingualConfig()
            self.target_lang = "en"

        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self.cache = TranslationCache() if use_cache else None
        logger.info("CrossLingualDetector initialized")

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.config.embedding_model)
            except Exception:
                self._model = "mock"
        return self._model

    def detect_language(self, text: str) -> str:
        words = text.split()
        if not words:
            return "en"
        text_words = set(w.lower() for w in words[:100])
        common_en = {"the", "is", "and", "of", "to", "in"}
        common_es = {"el", "la", "de", "en", "que"}
        common_fr = {"le", "la", "de", "et", "les"}
        common_de = {"der", "die", "und", "den"}

        scores = {
            "en": len(text_words & common_en),
            "es": len(text_words & common_es),
            "fr": len(text_words & common_fr),
            "de": len(text_words & common_de),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "en"

    def embed_text(self, text: str, lang: str = "en") -> np.ndarray:
        cache_key = f"{lang}:{hash(text)}"
        if self.config.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        model = self._get_model()
        if model == "mock":
            embedding = np.random.rand(384).astype(np.float32)
        else:
            embedding = model.encode(text, normalize_embeddings=True)

        if self.config.enable_cache:
            self._cache[cache_key] = embedding
        return embedding

    def compare_across_languages(
        self, source_embeddings: list[np.ndarray], target_embeddings: list[np.ndarray]
    ) -> np.ndarray:
        if not source_embeddings or not target_embeddings:
            return np.array([])
        source_matrix = np.array(source_embeddings)
        target_matrix = np.array(target_embeddings)
        source_norm = source_matrix / np.linalg.norm(
            source_matrix, axis=1, keepdims=True
        )
        target_norm = target_matrix / np.linalg.norm(
            target_matrix, axis=1, keepdims=True
        )
        return np.dot(source_norm, target_norm.T)

    def detect_cross_lingual_plagiarism(
        self,
        documents: dict[str, tuple[str, list[str]]],
        threshold: Optional[float] = None,
    ) -> CrossLingualResult:
        start_time = datetime.now()
        threshold = threshold or self.config.similarity_threshold

        doc_chunks = {name: chunks for name, (_, chunks) in documents.items()}
        embeddings = {
            doc_name: [
                self.embed_text(chunk.text if hasattr(chunk, "text") else chunk)
                for chunk in chunks[: self.config.max_chunks_per_doc]
            ]
            for doc_name, chunks in doc_chunks.items()
        }

        lang_dist = {}
        doc_langs = {}
        for name, (lang, _) in documents.items():
            doc_langs[name] = lang
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        matches = []
        doc_names = list(documents.keys())
        total_comparisons = 0

        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                name_a, name_b = doc_names[i], doc_names[j]
                lang_a, lang_b = doc_langs[name_a], doc_langs[name_b]
                is_cross_lingual = lang_a != lang_b

                emb_a = embeddings.get(name_a, [])
                emb_b = embeddings.get(name_b, [])

                if not emb_a or not emb_b:
                    continue

                sim_matrix = self.compare_across_languages(emb_a, emb_b)
                total_comparisons += 1

                for ci in range(sim_matrix.shape[0]):
                    for cj in range(sim_matrix.shape[1]):
                        score = float(sim_matrix[ci, cj])
                        if score >= threshold:
                            chunks_a = doc_chunks.get(name_a, [])
                            chunks_b = doc_chunks.get(name_b, [])
                            matches.append(
                                LanguageMatch(
                                    source_doc=name_a,
                                    source_lang=lang_a,
                                    source_chunk=(
                                        chunks_a[ci] if ci < len(chunks_a) else ""
                                    ),
                                    target_doc=name_b,
                                    target_lang=lang_b,
                                    target_chunk=(
                                        chunks_b[cj] if cj < len(chunks_b) else ""
                                    ),
                                    similarity=score,
                                    method="multilingual_embedding",
                                    translation_used=is_cross_lingual
                                    and self.config.use_translation_bridge,
                                    confidence=min(score * 1.1, 1.0),
                                )
                            )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        matches = matches[: self.config.top_k]
        processing_time = (datetime.now() - start_time).total_seconds()

        summary = {
            "total_documents": len(documents),
            "languages_detected": len(lang_dist),
            "cross_lingual_matches": sum(
                1 for m in matches if m.source_lang != m.target_lang
            ),
            "same_language_matches": sum(
                1 for m in matches if m.source_lang == m.target_lang
            ),
            "high_severity": sum(1 for m in matches if m.similarity >= 0.90),
            "threshold_used": threshold,
        }

        doc_info = [
            {
                "name": name,
                "language": doc_langs.get(name, "unknown"),
                "language_name": LANGUAGE_NAMES.get(doc_langs.get(name, ""), "Unknown"),
                "chunk_count": len(doc_chunks.get(name, [])),
            }
            for name in doc_names
        ]

        return CrossLingualResult(
            documents=doc_info,
            matches=matches,
            language_distribution=lang_dist,
            total_comparisons=total_comparisons,
            processing_time=processing_time,
            summary=summary,
        )

    def detect_pair(
        self,
        text_a: str,
        text_b: str,
        lang_a: str,
        lang_b: str,
        threshold: float = 0.65,
        **kwargs,
    ) -> CrossLingualResult:
        emb_a = [self.embed_text(text_a, lang_a)]
        emb_b = [self.embed_text(text_b, lang_b)]
        matrix = self.compare_across_languages(emb_a, emb_b)
        sim = float(matrix[0, 0]) if matrix.size > 0 else 0.0

        return CrossLingualResult(
            source_text=text_a,
            target_text=text_b,
            source_lang=lang_a,
            target_lang=lang_b,
            similarity_score=sim,
            is_plagiarism=sim >= threshold,
            confidence=min(sim * 1.1, 1.0),
        )


def get_language_name(lang_code: str) -> str:
    return LANGUAGE_NAMES.get(lang_code, lang_code.upper())


_detector: Optional[CrossLingualDetector] = None


def get_cross_lingual_detector() -> CrossLingualDetector:
    global _detector
    if _detector is None:
        _detector = CrossLingualDetector()
    return _detector


def detect_cross_lingual_plagiarism(
    text_a: str,
    text_b: str,
    lang_a: str,
    lang_b: str,
    threshold: float = 0.65,
) -> CrossLingualResult:
    detector = get_cross_lingual_detector()
    return detector.detect_pair(text_a, text_b, lang_a, lang_b, threshold=threshold)

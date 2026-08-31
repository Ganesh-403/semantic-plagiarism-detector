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
AI-Powered Plagiarism Scoring Engine.

Provides advanced plagiarism scoring using multiple detection metrics,
content fingerprinting, and ensemble scoring for accurate detection.
"""

import hashlib
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Severity levels for plagiarism detection."""

    CLEAN = "clean"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ScoringMethod(Enum):
    """Available scoring methods."""

    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    STRUCTURAL = "structural"
    STATISTICAL = "statistical"
    FINGERPRINT = "fingerprint"
    ENSEMBLE = "ensemble"


@dataclass
class ScoreComponent:
    """Individual scoring component result."""

    method: str
    score: float
    confidence: float
    details: dict[str, Any]
    weight: float = 1.0

    def weighted_score(self) -> float:
        return self.score * self.weight * self.confidence


@dataclass
class PlagiarismScore:
    """Complete plagiarism score for a document pair."""

    doc_a: str
    doc_b: str
    overall_score: float
    severity: SeverityLevel
    components: list[ScoreComponent]
    fingerprint_match: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "overall_score": self.overall_score,
            "severity": self.severity.value,
            "components": [asdict(c) for c in self.components],
            "fingerprint_match": self.fingerprint_match,
            "metadata": self.metadata,
        }


@dataclass
class ContentFingerprint:
    """Content fingerprint for near-duplicate detection."""

    doc_name: str
    shingles: set[str]
    minhash_signature: list[int]
    ngram_hash: str
    word_set_hash: str
    paragraph_hashes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "shingle_count": len(self.shingles),
            "minhash_signature": self.minhash_signature,
            "ngram_hash": self.ngram_hash,
            "word_set_hash": self.word_set_hash,
            "paragraph_count": len(self.paragraph_hashes),
        }


@dataclass
class ScoringConfig:
    """Configuration for the scoring engine."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.35,
            "lexical": 0.25,
            "structural": 0.15,
            "statistical": 0.15,
            "fingerprint": 0.10,
        }
    )
    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "clean": 0.20,
            "low": 0.40,
            "moderate": 0.60,
            "high": 0.80,
        }
    )
    shingle_size: int = 5
    minhash_num_perm: int = 128
    ngram_size: int = 3
    enable_fingerprint: bool = True
    fingerprint_threshold: float = 0.85


class ContentFingerprinter:
    """
    Content fingerprinting for near-duplicate detection.

    Uses shingling, MinHash, and n-gram hashing for fast
    duplicate detection without embeddings.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def create_fingerprint(self, text: str, doc_name: str = "") -> ContentFingerprint:
        """Create a content fingerprint from text."""
        words = text.lower().split()
        shingles = self._create_shingles(words)
        minhash = self._compute_minhash(shingles)
        ngram_hash = self._ngram_hash(text)
        word_hash = self._word_set_hash(words)
        para_hashes = self._paragraph_hashes(text)

        return ContentFingerprint(
            doc_name=doc_name,
            shingles=shingles,
            minhash_signature=minhash,
            ngram_hash=ngram_hash,
            word_set_hash=word_hash,
            paragraph_hashes=para_hashes,
        )

    def _create_shingles(self, words: list[str]) -> set[str]:
        """Create word-level shingles."""
        shingles = set()
        for i in range(len(words) - self.config.shingle_size + 1):
            shingle = " ".join(words[i : i + self.config.shingle_size])
            shingles.add(hashlib.md5(shingle.encode()).hexdigest()[:12])
        return shingles

    def _compute_minhash(self, shingles: set[str]) -> list[int]:
        """Compute MinHash signature."""
        signature = []
        for i in range(self.config.minhash_num_perm):
            min_val = float("inf")
            for shingle in shingles:
                hash_val = int(hashlib.md5(f"{shingle}_{i}".encode()).hexdigest(), 16)  # nosec
                min_val = min(min_val, hash_val)
            signature.append(min_val % (2**32))
        return signature

    def _ngram_hash(self, text: str) -> str:
        """Compute n-gram hash."""
        words = text.lower().split()
        ngrams = [
            " ".join(words[i : i + self.config.ngram_size])
            for i in range(len(words) - self.config.ngram_size + 1)
        ]
        combined = "|".join(sorted(ngrams)[:100])
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _word_set_hash(self, words: list[str]) -> str:
        """Compute word set hash."""
        word_set = sorted({w.lower() for w in words})
        return hashlib.md5(" ".join(word_set).encode()).hexdigest()[:16]  # nosec

    def _paragraph_hashes(self, text: str) -> list[str]:
        """Compute paragraph-level hashes."""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        return [hashlib.md5(p.encode()).hexdigest()[:12] for p in paragraphs[:50]]  # nosec

    def compare_fingerprints(
        self, fp1: ContentFingerprint, fp2: ContentFingerprint
    ) -> float:
        """Compare two fingerprints using Jaccard similarity."""
        if not fp1.shingles or not fp2.shingles:
            return 0.0
        intersection = fp1.shingles & fp2.shingles
        union = fp1.shingles | fp2.shingles
        return len(intersection) / len(union) if union else 0.0

    def detect_near_duplicates(
        self, fingerprints: List[ContentFingerprint], threshold: float = 0.85
    ) -> List[Tuple[str, str, float]]:
        """Detect near-duplicate document pairs."""
        duplicates = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = self.compare_fingerprints(fingerprints[i], fingerprints[j])
                if sim >= threshold:
                    duplicates.append(
                        (fingerprints[i].doc_name, fingerprints[j].doc_name, sim)
                    )
        return duplicates


class AIScoringEngine:
    """
    AI-powered plagiarism scoring engine with multiple detection metrics.

    Combines semantic, lexical, structural, statistical, and fingerprint
    analysis for comprehensive plagiarism scoring.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.fingerprinter = ContentFingerprinter(config)
        self._cache: dict[str, Any] = {}

    def _preprocess(self, text: str) -> list[str]:
        """Tokenize and preprocess text."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if len(w) > 2]

    def compute_semantic_score(self, text_a: str, text_b: str) -> ScoreComponent:
        """Compute semantic similarity score."""
        words_a = self._preprocess(text_a)
        words_b = self._preprocess(text_b)
        vocab_a = set(words_a)
        vocab_b = set(words_b)
        intersection = vocab_a & vocab_b
        union = vocab_a | vocab_b
        jaccard = len(intersection) / len(union) if union else 0.0
        word_freq_a = Counter(words_a)
        word_freq_b = Counter(words_b)
        cosine = sum(
            word_freq_a.get(w, 0) * word_freq_b.get(w, 0) for w in intersection
        )
        norm_a = sum(v**2 for v in word_freq_a.values()) ** 0.5
        norm_b = sum(v**2 for v in word_freq_b.values()) ** 0.5
        cosine = cosine / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
        score = 0.5 * jaccard + 0.5 * cosine
        return ScoreComponent(
            method="semantic",
            score=min(score, 1.0),
            confidence=0.85,
            details={
                "jaccard": jaccard,
                "cosine": cosine,
                "vocab_overlap": len(intersection) / len(union) if union else 0,
            },
        )

    def compute_lexical_score(self, text_a: str, text_b: str) -> ScoreComponent:
        """Compute lexical similarity score."""
        words_a = self._preprocess(text_a)
        words_b = self._preprocess(text_b)
        if not words_a or not words_b:
            return ScoreComponent(
                method="lexical", score=0.0, confidence=0.5, details={}
            )
        bg_a = set(zip(words_a, words_a[1:]))
        bg_b = set(zip(words_b, words_b[1:]))
        bg_sim = len(bg_a & bg_b) / len(bg_a | bg_b) if (bg_a | bg_b) else 0.0
        tg_a = (
            set(zip(words_a, words_a[1:], words_a[2:])) if len(words_a) > 2 else set()
        )
        tg_b = (
            set(zip(words_b, words_b[1:], words_b[2:])) if len(words_b) > 2 else set()
        )
        tg_sim = len(tg_a & tg_b) / len(tg_a | tg_b) if (tg_a | tg_b) else 0.0
        score = 0.6 * bg_sim + 0.4 * tg_sim
        return ScoreComponent(
            method="lexical",
            score=min(score, 1.0),
            confidence=0.80,
            details={"bigram_similarity": bg_sim, "trigram_similarity": tg_sim},
        )

    def compute_structural_score(self, text_a: str, text_b: str) -> ScoreComponent:
        """Compute structural similarity score."""
        paras_a = [p.strip() for p in text_a.split("\n\n") if p.strip()]
        paras_b = [p.strip() for p in text_b.split("\n\n") if p.strip()]
        para_ratio = (
            min(len(paras_a), len(paras_b)) / max(len(paras_a), len(paras_b))
            if max(len(paras_a), len(paras_b)) > 0
            else 0
        )
        sent_a = len(re.split(r"[.!?]+", text_a))
        sent_b = len(re.split(r"[.!?]+", text_b))
        sent_ratio = (
            min(sent_a, sent_b) / max(sent_a, sent_b) if max(sent_a, sent_b) > 0 else 0
        )
        len_ratio = (
            min(len(text_a), len(text_b)) / max(len(text_a), len(text_b))
            if max(len(text_a), len(text_b)) > 0
            else 0
        )
        score = 0.4 * para_ratio + 0.3 * sent_ratio + 0.3 * len_ratio
        return ScoreComponent(
            method="structural",
            score=min(score, 1.0),
            confidence=0.70,
            details={
                "paragraph_ratio": para_ratio,
                "sentence_ratio": sent_ratio,
                "length_ratio": len_ratio,
            },
        )

    def compute_statistical_score(self, text_a: str, text_b: str) -> ScoreComponent:
        """Compute statistical similarity score."""
        words_a = self._preprocess(text_a)
        words_b = self._preprocess(text_b)
        avg_len_a = np.mean([len(w) for w in words_a]) if words_a else 0
        avg_len_b = np.mean([len(w) for w in words_b]) if words_b else 0
        vocab_rich_a = len(set(words_a)) / len(words_a) if words_a else 0
        vocab_rich_b = len(set(words_b)) / len(words_b) if words_b else 0
        type_token_sim = 1 - abs(vocab_rich_a - vocab_rich_b)
        freq_a = Counter(words_a)
        freq_b = Counter(words_b)
        top_a = {w for w, _ in freq_a.most_common(20)}
        top_b = {w for w, _ in freq_b.most_common(20)}
        keyword_sim = len(top_a & top_b) / len(top_a | top_b) if (top_a | top_b) else 0
        score = 0.5 * type_token_sim + 0.5 * keyword_sim
        return ScoreComponent(
            method="statistical",
            score=min(score, 1.0),
            confidence=0.75,
            details={
                "type_token_similarity": type_token_sim,
                "keyword_overlap": keyword_sim,
            },
        )

    def compute_fingerprint_score(
        self, text_a: str, text_b: str, doc_a: str = "", doc_b: str = ""
    ) -> ScoreComponent:
        """Compute fingerprint-based similarity score."""
        fp_a = self.fingerprinter.create_fingerprint(text_a, doc_a)
        fp_b = self.fingerprinter.create_fingerprint(text_b, doc_b)
        shingle_sim = self.fingerprinter.compare_fingerprints(fp_a, fp_b)
        para_match = len(set(fp_a.paragraph_hashes) & set(fp_b.paragraph_hashes))
        para_total = max(len(fp_a.paragraph_hashes), len(fp_b.paragraph_hashes))
        para_sim = para_match / para_total if para_total > 0 else 0
        score = 0.7 * shingle_sim + 0.3 * para_sim
        return ScoreComponent(
            method="fingerprint",
            score=min(score, 1.0),
            confidence=0.90,
            details={
                "shingle_similarity": shingle_sim,
                "paragraph_matches": para_match,
                "paragraph_total": para_total,
            },
        )

    def compute_ensemble_score(
        self, components: List[ScoreComponent]
    ) -> ScoreComponent:
        """Compute weighted ensemble score from all components."""
        total_weight = 0
        weighted_sum = 0
        for comp in components:
            weight = self.config.weights.get(comp.method, 0.1)
            weighted_sum += comp.weighted_score() * weight
            total_weight += weight * comp.confidence
        score = weighted_sum / total_weight if total_weight > 0 else 0
        confidence = np.mean([c.confidence for c in components]) if components else 0.5
        return ScoreComponent(
            method="ensemble",
            score=min(score, 1.0),
            confidence=confidence,
            details={
                "component_count": len(components),
                "weights_used": {
                    c.method: self.config.weights.get(c.method, 0) for c in components
                },
            },
        )

    def determine_severity(self, score: float) -> SeverityLevel:
        """Determine severity level from score."""
        thresholds = self.config.severity_thresholds
        if score >= thresholds.get("high", 0.8):
            return SeverityLevel.CRITICAL
        elif score >= thresholds.get("moderate", 0.6):
            return SeverityLevel.HIGH
        elif score >= thresholds.get("low", 0.4):
            return SeverityLevel.MODERATE
        elif score >= thresholds.get("clean", 0.2):
            return SeverityLevel.LOW
        return SeverityLevel.CLEAN

    def score_documents(
        self, text_a: str, text_b: str, doc_a: str = "doc_a", doc_b: str = "doc_b"
    ) -> PlagiarismScore:
        """
        Compute complete plagiarism score for two documents.

        Args:
            text_a: First document text
            text_b: Second document text
            doc_a: First document name
            doc_b: Second document name

        Returns:
            PlagiarismScore with all component scores
        """
        components = [
            self.compute_semantic_score(text_a, text_b),
            self.compute_lexical_score(text_a, text_b),
            self.compute_structural_score(text_a, text_b),
            self.compute_statistical_score(text_a, text_b),
        ]

        if self.config.enable_fingerprint:
            components.append(
                self.compute_fingerprint_score(text_a, text_b, doc_a, doc_b)
            )

        ensemble = self.compute_ensemble_score(components)
        components.append(ensemble)

        overall = ensemble.score
        severity = self.determine_severity(overall)

        fp_match = any(
            c.method == "fingerprint"
            and c.details.get("shingle_similarity", 0)
            >= self.config.fingerprint_threshold
            for c in components
        )

        return PlagiarismScore(
            doc_a=doc_a,
            doc_b=doc_b,
            overall_score=overall,
            severity=severity,
            components=components,
            fingerprint_match=fp_match,
            metadata={
                "text_a_length": len(text_a),
                "text_b_length": len(text_b),
                "timestamp": datetime.now().isoformat(),
            },
        )

        return PlagiarismScore(
            doc_a=doc_a,
            doc_b=doc_b,
            overall_score=overall,
            severity=severity,
            components=components,
            fingerprint_match=fp_match,
            metadata={
                "text_a_length": len(text_a),
                "text_b_length": len(text_b),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def batch_score(self, documents: dict[str, str]) -> list[PlagiarismScore]:
        """Score all document pairs in a batch."""
        scores = []
        doc_names = list(documents.keys())
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                score = self.score_documents(
                    documents[doc_names[i]],
                    documents[doc_names[j]],
                    doc_names[i],
                    doc_names[j],
                )
                scores.append(score)
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        return scores

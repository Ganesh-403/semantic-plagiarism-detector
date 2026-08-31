"""
document_health_scorer.py
-------------------------
Scoring engine that evaluates document quality across multiple dimensions.

Produces a 0–100 health score per document based on:
  • Metadata completeness  (student name, class, assignment title, tags, language)
  • Chunk balance          (even chunk sizes, no suspiciously short/long chunks)
  • Embedding coverage     (every chunk has a valid 384-dim embedding)
  • Content quality        (word count, character diversity, stop-word ratio)
  • Duplicate fingerprints (hash uniqueness across corpus)

Each dimension is independently weighted and the composite score is persisted
for trend analysis and quality-gate decisions.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight constants (must sum to 1.0)
# ---------------------------------------------------------------------------
W_METADATA = 0.25
W_CHUNK_BALANCE = 0.20
W_EMBEDDING = 0.20
W_CONTENT = 0.25
W_FINGERPRINT = 0.10

# Thresholds
MIN_WORDS = 50
IDEAL_CHUNK_WORDS = 150
CHUNK_STDEV_MAX = 300  # acceptable standard deviation of chunk word counts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single health dimension."""

    name: str
    score: float  # 0–100
    weight: float
    details: str = ""

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class HealthReport:
    """Complete health report for a single document."""

    filename: str
    overall_score: float
    grade: str  # A+, A, B+, B, C, D, F
    dimensions: list[DimensionScore]
    checked_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "dimensions": [
                {
                    "name": d.name,
                    "score": round(d.score, 2),
                    "weight": d.weight,
                    "weighted": round(d.weighted, 2),
                    "details": d.details,
                }
                for d in self.dimensions
            ],
            "checked_at": self.checked_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_metadata_completeness(doc: dict[str, Any]) -> DimensionScore:
    """Evaluate how complete the document metadata is."""
    fields = [
        ("student_name", "Student Name"),
        ("class_section", "Class Section"),
        ("assignment_title", "Assignment Title"),
        ("detected_language", "Language"),
        ("tags", "Tags"),
    ]
    present = 0
    missing = []
    for key, label in fields:
        val = doc.get(key)
        if val and str(val).strip():
            present += 1
        else:
            missing.append(label)

    score = (present / len(fields)) * 100
    detail = f"{present}/{len(fields)} fields populated"
    if missing:
        detail += f" — missing: {', '.join(missing)}"

    return DimensionScore(
        name="metadata_completeness",
        score=score,
        weight=W_METADATA,
        details=detail,
    )


def _score_chunk_balance(chunk_word_counts: list[int]) -> DimensionScore:
    """Evaluate whether chunk sizes are balanced (no extreme outliers)."""
    if not chunk_word_counts:
        return DimensionScore(
            name="chunk_balance",
            score=0.0,
            weight=W_CHUNK_BALANCE,
            details="No chunks found",
        )

    n = len(chunk_word_counts)
    mean = sum(chunk_word_counts) / n
    variance = sum((wc - mean) ** 2 for wc in chunk_word_counts) / n
    stdev = math.sqrt(variance)

    # Perfect score when stdev ≤ 50, zero when stdev ≥ CHUNK_STDEV_MAX
    if stdev <= 50:
        score = 100.0
    elif stdev >= CHUNK_STDEV_MAX:
        score = 0.0
    else:
        score = 100.0 * (1.0 - (stdev - 50) / (CHUNK_STDEV_MAX - 50))

    # Penalize very short chunks
    short_chunks = sum(1 for wc in chunk_word_counts if wc < 20)
    if short_chunks > 0:
        penalty = min(30, short_chunks * 5)
        score = max(0, score - penalty)

    detail = f"{n} chunks, mean={mean:.0f}w, stdev={stdev:.0f}"
    if short_chunks:
        detail += f", {short_chunks} very short"

    return DimensionScore(
        name="chunk_balance",
        score=round(score, 2),
        weight=W_CHUNK_BALANCE,
        details=detail,
    )


def _score_embedding_coverage(total_chunks: int, chunks_with_embeddings: int) -> DimensionScore:
    """Evaluate embedding coverage ratio."""
    if total_chunks == 0:
        return DimensionScore(
            name="embedding_coverage",
            score=0.0,
            weight=W_EMBEDDING,
            details="No chunks to embed",
        )

    ratio = chunks_with_embeddings / total_chunks
    score = ratio * 100
    detail = f"{chunks_with_embeddings}/{total_chunks} chunks have embeddings ({ratio:.0%})"
    return DimensionScore(
        name="embedding_coverage",
        score=round(score, 2),
        weight=W_EMBEDDING,
        details=detail,
    )


def _score_content_quality(
    chunk_texts: list[str],
    total_words: int,
) -> DimensionScore:
    """Evaluate content quality based on word count and character diversity."""
    if not chunk_texts or total_words == 0:
        return DimensionScore(
            name="content_quality",
            score=0.0,
            weight=W_CONTENT,
            details="No content to evaluate",
        )

    full_text = " ".join(chunk_texts)

    # --- Word count scoring ---
    if total_words < MIN_WORDS:
        word_score = max(0, (total_words / MIN_WORDS) * 60)
    elif total_words < 200:
        word_score = 60 + (total_words - MIN_WORDS) / (200 - MIN_WORDS) * 20
    else:
        word_score = 80 + min(20, (total_words - 200) / 500 * 20)

    # --- Character diversity (type-token ratio) ---
    alpha_chars = [c.lower() for c in full_text if c.isalpha()]
    unique_chars = len(set(alpha_chars))
    total_chars = len(alpha_chars) if alpha_chars else 1
    diversity = unique_chars / total_chars
    diversity_score = min(100, diversity * 200)  # ~50% unique → 100

    # --- Stop-word ratio penalty ---
    common_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
        "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    }
    words = re.findall(r"\b\w+\b", full_text.lower())
    if words:
        stop_ratio = sum(1 for w in words if w in common_words) / len(words)
        # Ideal stop-word ratio for academic text: 40-60%
        if 0.35 <= stop_ratio <= 0.65:
            stop_score = 100
        elif stop_ratio < 0.35:
            stop_score = max(0, (stop_ratio / 0.35) * 100)
        else:
            stop_score = max(0, (1.0 - (stop_ratio - 0.65) / 0.35) * 100)
    else:
        stop_score = 50

    # --- Composite ---
    score = word_score * 0.35 + diversity_score * 0.35 + stop_score * 0.30

    detail = (
        f"{total_words} words, diversity={diversity:.2f}, "
        f"word_score={word_score:.0f}, div_score={diversity_score:.0f}, "
        f"stop_score={stop_score:.0f}"
    )
    return DimensionScore(
        name="content_quality",
        score=round(score, 2),
        weight=W_CONTENT,
        details=detail,
    )


def _score_fingerprint(file_hash: str, existing_hashes: set[str]) -> DimensionScore:
    """Check if the document hash is unique in the corpus."""
    if not file_hash:
        return DimensionScore(
            name="fingerprint_uniqueness",
            score=50.0,
            weight=W_FINGERPRINT,
            details="No file hash available",
        )

    if file_hash in existing_hashes:
        return DimensionScore(
            name="fingerprint_uniqueness",
            score=0.0,
            weight=W_FINGERPRINT,
            details=f"Hash {file_hash[:12]}… is a duplicate",
        )

    return DimensionScore(
        name="fingerprint_uniqueness",
        score=100.0,
        weight=W_FINGERPRINT,
        details=f"Hash {file_hash[:12]}… is unique",
    )


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------

def _assign_grade(score: float) -> str:
    """Map a 0–100 numeric score to a letter grade."""
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_document(
    doc: dict[str, Any],
    chunk_texts: list[str],
    chunk_word_counts: list[int],
    total_chunks: int,
    chunks_with_embeddings: int,
    total_words: int,
    existing_hashes: Optional[set[str]] = None,
) -> HealthReport:
    """
    Score a single document across all health dimensions.

    Args:
        doc: Document metadata dict (from corpus DB row).
        chunk_texts: List of chunk text content.
        chunk_word_counts: Word count per chunk.
        total_chunks: Total chunk count.
        chunks_with_embeddings: Number of chunks with valid embeddings.
        total_words: Total word count across all chunks.
        existing_hashes: Set of known corpus file hashes for uniqueness check.

    Returns:
        A HealthReport with the composite score and per-dimension breakdown.
    """
    if existing_hashes is None:
        existing_hashes = set()

    dimensions = [
        _score_metadata_completeness(doc),
        _score_chunk_balance(chunk_word_counts),
        _score_embedding_coverage(total_chunks, chunks_with_embeddings),
        _score_content_quality(chunk_texts, total_words),
        _score_fingerprint(doc.get("file_hash", ""), existing_hashes),
    ]

    overall = sum(d.weighted for d in dimensions)
    grade = _assign_grade(overall)

    return HealthReport(
        filename=doc.get("filename", "unknown"),
        overall_score=round(overall, 2),
        grade=grade,
        dimensions=dimensions,
        checked_at=datetime.now(timezone.utc).isoformat(),
        metadata={
            "total_chunks": total_chunks,
            "total_words": total_words,
            "chunks_with_embeddings": chunks_with_embeddings,
        },
    )


def score_corpus_batch(
    documents: list[dict[str, Any]],
    get_chunks_for_doc: Any = None,
    existing_hashes: Optional[set[str]] = None,
) -> list[HealthReport]:
    """
    Score an entire batch of documents.

    Args:
        documents: List of document metadata dicts.
        get_chunks_for_doc: Callable(filename) → (chunk_texts, embeddings_array)
        existing_hashes: Set of known corpus hashes.

    Returns:
        List of HealthReport objects, one per document.
    """
    if existing_hashes is None:
        existing_hashes = {d.get("file_hash", "") for d in documents}

    reports: list[HealthReport] = []
    for doc in documents:
        filename = doc.get("filename", "")
        chunk_texts: list[str] = []
        total_chunks = 0
        chunks_with_embeddings = 0
        total_words = 0

        if get_chunks_for_doc is not None:
            try:
                result = get_chunks_for_doc(filename)
                if result:
                    texts, embeddings = result
                    chunk_texts = texts
                    total_chunks = len(texts)
                    if embeddings is not None and len(embeddings) > 0:
                        chunks_with_embeddings = len(embeddings)
                    total_words = sum(len(t.split()) for t in texts)
            except Exception as exc:
                logger.warning("Failed to load chunks for %s: %s", filename, exc)

        report = score_document(
            doc=doc,
            chunk_texts=chunk_texts,
            chunk_word_counts=[len(t.split()) for t in chunk_texts],
            total_chunks=total_chunks,
            chunks_with_embeddings=chunks_with_embeddings,
            total_words=total_words,
            existing_hashes=existing_hashes,
        )
        reports.append(report)

    return reports


def compute_quality_gate(
    report: HealthReport,
    *,
    min_score: float = 60.0,
    min_grade: str = "D",
) -> dict[str, Any]:
    """
    Determine whether a document passes the quality gate.

    Args:
        report: The health report for the document.
        min_score: Minimum acceptable overall score.
        min_grade: Minimum acceptable letter grade.

    Returns:
        Dict with 'passed' bool, reason, and the score/grade.
    """
    grade_order = {"F": 0, "D": 1, "C": 2, "B-": 3, "B": 4, "B+": 5, "A-": 6, "A": 7, "A+": 8}

    passed = True
    reasons: list[str] = []

    if report.overall_score < min_score:
        passed = False
        reasons.append(
            f"Score {report.overall_score:.1f} is below minimum {min_score:.1f}"
        )

    doc_grade_rank = grade_order.get(report.grade, -1)
    min_grade_rank = grade_order.get(min_grade, -1)
    if doc_grade_rank < min_grade_rank:
        passed = False
        reasons.append(
            f"Grade {report.grade} is below minimum {min_grade}"
        )

    # Check for any dimension scoring below 25
    critical_dims = [d for d in report.dimensions if d.score < 25]
    if critical_dims:
        dim_names = ", ".join(d.name for d in critical_dims)
        reasons.append(f"Critical low scores in: {dim_names}")
        # Don't auto-fail, but flag it
        if not reasons:
            passed = False

    return {
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "All quality checks passed",
        "overall_score": report.overall_score,
        "grade": report.grade,
    }


# ---------------------------------------------------------------------------
# Batch analytics
# ---------------------------------------------------------------------------

def aggregate_reports(reports: list[HealthReport]) -> dict[str, Any]:
    """Compute aggregate statistics across a batch of health reports."""
    if not reports:
        return {
            "count": 0,
            "avg_score": 0.0,
            "median_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "grade_distribution": {},
            "pass_rate": 0.0,
            "dimension_averages": {},
        }

    scores = sorted(r.overall_score for r in reports)
    n = len(scores)

    grade_dist: dict[str, int] = {}
    for r in reports:
        grade_dist[r.grade] = grade_dist.get(r.grade, 0) + 1

    # Dimension averages
    dim_totals: dict[str, float] = {}
    dim_counts: dict[str, int] = {}
    for r in reports:
        for d in r.dimensions:
            dim_totals[d.name] = dim_totals.get(d.name, 0) + d.score
            dim_counts[d.name] = dim_counts.get(d.name, 0) + 1
    dim_avgs = {
        name: round(dim_totals[name] / dim_counts[name], 2)
        for name in dim_totals
    }

    # Pass rate (score >= 60)
    passing = sum(1 for s in scores if s >= 60)

    median = scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2

    return {
        "count": n,
        "avg_score": round(sum(scores) / n, 2),
        "median_score": round(median, 2),
        "min_score": round(scores[0], 2),
        "max_score": round(scores[-1], 2),
        "grade_distribution": grade_dist,
        "pass_rate": round(passing / n * 100, 1),
        "dimension_averages": dim_avgs,
    }

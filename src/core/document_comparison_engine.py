"""
document_comparison_engine.py
-----------------------------
Detailed side-by-side document comparison engine that produces
paragraph-level similarity breakdowns, highlighted matching sections,
and structured comparison reports for any two documents in the corpus.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ParagraphMatch:
    """A single paragraph-level match between two documents."""
    source_index: int
    target_index: int
    source_text: str
    target_text: str
    similarity: float
    is_exact: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WordOverlap:
    """Word-level overlap statistics between two texts."""
    common_words: int
    unique_source_words: int
    unique_target_words: int
    jaccard_similarity: float
    top_common: List[Tuple[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["top_common"] = self.top_common
        return d


@dataclass
class ComparisonResult:
    """Complete comparison result between two documents."""
    source_filename: str
    target_filename: str
    source_paragraphs: List[str]
    target_paragraphs: List[str]
    paragraph_matches: List[ParagraphMatch]
    document_similarity: float
    max_paragraph_similarity: float
    avg_paragraph_similarity: float
    source_coverage: float
    target_coverage: float
    word_overlap: WordOverlap
    matched_paragraph_count: int
    total_paragraphs: int
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "target_filename": self.target_filename,
            "paragraph_matches": [m.to_dict() for m in self.paragraph_matches],
            "document_similarity": self.document_similarity,
            "max_paragraph_similarity": self.max_paragraph_similarity,
            "avg_paragraph_similarity": self.avg_paragraph_similarity,
            "source_coverage": self.source_coverage,
            "target_coverage": self.target_coverage,
            "word_overlap": self.word_overlap.to_dict(),
            "matched_paragraph_count": self.matched_paragraph_count,
            "total_paragraphs": self.total_paragraphs,
            "severity": self.severity,
        }


# ── Text preprocessing ───────────────────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _split_paragraphs(text: str, min_words: int = 5) -> List[str]:
    """Split text into paragraphs, discarding short ones."""
    paragraphs = re.split(r"\n\s*\n", text)
    result = []
    for p in paragraphs:
        cleaned = p.strip()
        if cleaned and len(cleaned.split()) >= min_words:
            result.append(cleaned)
    return result


# ── Similarity computation ───────────────────────────────────────────────────


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _compute_word_overlap(source_text: str, target_text: str) -> WordOverlap:
    """Compute word-level overlap statistics between two texts."""
    tokens_a = _tokenize(source_text)
    tokens_b = _tokenize(target_text)
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    common = set_a & set_b

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    top_common = sorted(
        [(w, counter_a[w] + counter_b[w]) for w in common],
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    jaccard = _jaccard_similarity(tokens_a, tokens_b)
    return WordOverlap(
        common_words=len(common),
        unique_source_words=len(set_a),
        unique_target_words=len(set_b),
        jaccard_similarity=round(jaccard, 4),
        top_common=top_common,
    )


# ── Paragraph matching ───────────────────────────────────────────────────────


def _find_paragraph_matches(
    source_paragraphs: List[str],
    target_paragraphs: List[str],
    source_embeddings: Optional[np.ndarray] = None,
    target_embeddings: Optional[np.ndarray] = None,
    threshold: float = 0.30,
) -> List[ParagraphMatch]:
    """Find matching paragraphs between two documents.

    When embeddings are provided, uses cosine similarity on vectors.
    Falls back to Jaccard-based lexical similarity.
    """
    matches: List[ParagraphMatch] = []
    use_vectors = (
        source_embeddings is not None
        and target_embeddings is not None
        and source_embeddings.size > 0
        and target_embeddings.size > 0
    )

    for i, src_text in enumerate(source_paragraphs):
        best_sim = 0.0
        best_j = -1
        best_target = ""

        for j, tgt_text in enumerate(target_paragraphs):
            if use_vectors and i < source_embeddings.shape[0] and j < target_embeddings.shape[0]:
                sim = _cosine_similarity(source_embeddings[i], target_embeddings[j])
            else:
                tokens_src = _tokenize(src_text)
                tokens_tgt = _tokenize(tgt_text)
                sim = _jaccard_similarity(tokens_src, tokens_tgt)

            if sim > best_sim:
                best_sim = sim
                best_j = j
                best_target = tgt_text

        if best_sim >= threshold and best_j >= 0:
            is_exact = src_text.strip().lower() == best_target.strip().lower()
            matches.append(ParagraphMatch(
                source_index=i,
                target_index=best_j,
                source_text=src_text,
                target_text=best_target,
                similarity=round(best_sim, 4),
                is_exact=is_exact,
            ))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


# ── Coverage computation ──────────────────────────────────────────────────────


def _compute_coverage(
    paragraphs: List[str],
    matches: List[ParagraphMatch],
    is_source: bool = True,
) -> float:
    """Compute what fraction of paragraphs are covered by matches."""
    if not paragraphs:
        return 0.0
    idx_key = "source_index" if is_source else "target_index"
    matched_indices = {getattr(m, idx_key) for m in matches}
    return round(len(matched_indices) / len(paragraphs), 4)


# ── Severity classification ──────────────────────────────────────────────────


def _classify_severity(score: float) -> str:
    """Classify document similarity into severity."""
    if score >= 0.90:
        return "High"
    if score >= 0.75:
        return "Medium"
    if score >= 0.59:
        return "Low"
    return "None"


# ── Main comparison function ─────────────────────────────────────────────────


def compare_documents(
    source_text: str,
    target_text: str,
    source_filename: str = "source",
    target_filename: str = "target",
    source_embeddings: Optional[np.ndarray] = None,
    target_embeddings: Optional[np.ndarray] = None,
    match_threshold: float = 0.30,
    min_paragraph_words: int = 5,
) -> ComparisonResult:
    """Compare two documents and produce a detailed comparison result.

    Args:
        source_text: Full text of the source document.
        target_text: Full text of the target document.
        source_filename: Name of the source document.
        target_filename: Name of the target document.
        source_embeddings: Optional paragraph embeddings for source (N x dim).
        target_embeddings: Optional paragraph embeddings for target (M x dim).
        match_threshold: Minimum similarity to consider a paragraph match.
        min_paragraph_words: Minimum words per paragraph to include.

    Returns:
        ComparisonResult with all comparison data.
    """
    source_paras = _split_paragraphs(source_text, min_words=min_paragraph_words)
    target_paras = _split_paragraphs(target_text, min_words=min_paragraph_words)

    matches = _find_paragraph_matches(
        source_paras, target_paras,
        source_embeddings, target_embeddings,
        threshold=match_threshold,
    )

    # Document-level similarity via word overlap
    word_overlap = _compute_word_overlap(source_text, target_text)
    doc_similarity = word_overlap.jaccard_similarity

    # Paragraph-level stats
    if matches:
        max_para_sim = max(m.similarity for m in matches)
        avg_para_sim = round(
            sum(m.similarity for m in matches) / len(matches), 4
        )
    else:
        max_para_sim = 0.0
        avg_para_sim = 0.0

    source_coverage = _compute_coverage(source_paras, matches, is_source=True)
    target_coverage = _compute_coverage(target_paras, matches, is_source=False)

    # Override doc_similarity with embedding-based if available
    if source_embeddings is not None and target_embeddings is not None:
        if source_embeddings.ndim == 2 and target_embeddings.ndim == 2:
            src_mean = np.mean(source_embeddings, axis=0)
            tgt_mean = np.mean(target_embeddings, axis=0)
            doc_similarity = _cosine_similarity(src_mean, tgt_mean)

    total_paras = len(source_paras) + len(target_paras)
    severity = _classify_severity(doc_similarity)

    return ComparisonResult(
        source_filename=source_filename,
        target_filename=target_filename,
        source_paragraphs=source_paras,
        target_paragraphs=target_paras,
        paragraph_matches=matches,
        document_similarity=round(doc_similarity, 4),
        max_paragraph_similarity=max_para_sim,
        avg_paragraph_similarity=avg_para_sim,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        word_overlap=word_overlap,
        matched_paragraph_count=len(matches),
        total_paragraphs=total_paras,
        severity=severity,
    )


# ── Highlighted text generation ──────────────────────────────────────────────


def _highlight_common_words(source_text: str, target_text: str) -> Tuple[str, str]:
    """Highlight common words in both texts with <mark> tags."""
    tokens_src = _tokenize(source_text)
    tokens_tgt = _tokenize(target_text)
    common = set(tokens_src) & set(tokens_tgt)

    highlighted_src = source_text
    highlighted_tgt = target_text

    # Sort by length descending to avoid partial replacements
    for word in sorted(common, key=len, reverse=True):
        pattern = re.compile(r"\b(" + re.escape(word) + r")\b", re.IGNORECASE)
        highlighted_src = pattern.sub(r"<mark>\1</mark>", highlighted_src)
        highlighted_tgt = pattern.sub(r"<mark>\1</mark>", highlighted_tgt)

    return highlighted_src, highlighted_tgt


def generate_highlighted_paragraphs(
    result: ComparisonResult,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Generate highlighted paragraph pairs for the top N matches.

    Returns a list of dicts with highlighted source/target text and score.
    """
    output: List[Dict[str, Any]] = []
    for match in result.paragraph_matches[:top_n]:
        h_src, h_tgt = _highlight_common_words(match.source_text, match.target_text)
        output.append({
            "source_index": match.source_index,
            "target_index": match.target_index,
            "source_highlighted": h_src,
            "target_highlighted": h_tgt,
            "similarity": match.similarity,
            "is_exact": match.is_exact,
        })
    return output

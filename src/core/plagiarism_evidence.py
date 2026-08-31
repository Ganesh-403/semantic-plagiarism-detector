"""
src/core/plagiarism_evidence.py
--------------------------------
Structured evidence graphs for plagiarism decisions (Issue #3914).
Traces final plagiarism scores back to matched chunks and contributing components.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """Metadata about a matched chunk."""

    doc_name: str
    chunk_index: int
    text: str
    embedding_dimension: Optional[int] = None


@dataclass
class EvidenceScoreComponent:
    """A single scoring component contributing to final score."""

    component_type: str  # 'semantic', 'lexical', 'hybrid', 'chunk_similarity'
    score: float
    weight: Optional[float] = None  # Contribution weight if combined
    description: Optional[str] = None


@dataclass
class EvidenceMatch:
    """A single matched chunk pair."""

    source_chunk: EvidenceChunk
    target_chunk: EvidenceChunk
    similarity_score: float
    component_scores: List[EvidenceScoreComponent] = field(default_factory=list)


@dataclass
class PlagiarismEvidence:
    """Complete evidence graph for a plagiarism flag."""

    doc_a: str
    doc_b: str
    timestamp: str
    final_score: float
    final_severity: str
    threshold_plagiarism: float
    threshold_medium: float
    threshold_high: float
    matched_chunks: List[EvidenceMatch] = field(default_factory=list)
    score_components: List[EvidenceScoreComponent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence to dictionary."""
        return {
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "timestamp": self.timestamp,
            "final_score": self.final_score,
            "final_severity": self.final_severity,
            "thresholds": {
                "plagiarism": self.threshold_plagiarism,
                "medium": self.threshold_medium,
                "high": self.threshold_high,
            },
            "matched_chunks": [
                {
                    "source": {
                        "doc_name": m.source_chunk.doc_name,
                        "chunk_index": m.source_chunk.chunk_index,
                        "text_preview": m.source_chunk.text[:100] + "..."
                        if len(m.source_chunk.text) > 100
                        else m.source_chunk.text,
                    },
                    "target": {
                        "doc_name": m.target_chunk.doc_name,
                        "chunk_index": m.target_chunk.chunk_index,
                        "text_preview": m.target_chunk.text[:100] + "..."
                        if len(m.target_chunk.text) > 100
                        else m.target_chunk.text,
                    },
                    "similarity_score": m.similarity_score,
                    "components": [asdict(c) for c in m.component_scores],
                }
                for m in self.matched_chunks
            ],
            "score_components": [asdict(c) for c in self.score_components],
            "metadata": self.metadata,
        }

    def trace_to_chunks(self) -> List[Tuple[str, str, float]]:
        """
        Trace evidence back to source/target chunk pairs.

        Returns:
            List of (source_text, target_text, similarity_score) tuples.
        """
        return [
            (m.source_chunk.text, m.target_chunk.text, m.similarity_score)
            for m in self.matched_chunks
        ]

    def explain(self) -> str:
        """
        Generate human-readable explanation of the evidence.

        Returns:
            Formatted explanation string.
        """
        lines = [
            f"Plagiarism Evidence for '{self.doc_a}' vs '{self.doc_b}'",
            f"=" * 60,
            f"Final Score: {self.final_score:.4f}",
            f"Severity: {self.final_severity}",
            f"Threshold: {self.threshold_plagiarism} (plagiarism), "
            f"{self.threshold_medium} (medium), {self.threshold_high} (high)",
            "",
            f"Number of matched chunks: {len(self.matched_chunks)}",
        ]

        if self.matched_chunks:
            lines.append("\nTop Matches:")
            for i, match in enumerate(self.matched_chunks[:5], 1):
                lines.append(f"\n  Match {i} (similarity: {match.similarity_score:.4f})")
                lines.append(
                    f"    Source: {match.source_chunk.doc_name} "
                    f"(chunk {match.source_chunk.chunk_index})"
                )
                lines.append(
                    f"    Target: {match.target_chunk.doc_name} "
                    f"(chunk {match.target_chunk.chunk_index})"
                )
                if match.component_scores:
                    for comp in match.component_scores:
                        lines.append(
                            f"      - {comp.component_type}: {comp.score:.4f}"
                        )

        if self.score_components:
            lines.append("\nScore Components:")
            for comp in self.score_components:
                lines.append(
                    f"  - {comp.component_type}: {comp.score:.4f}"
                    f" (weight: {comp.weight})" if comp.weight else ""
                )

        return "\n".join(lines)


def build_plagiarism_evidence(
    doc_a: str,
    doc_b: str,
    semantic_score: float,
    lexical_score: Optional[float] = None,
    hybrid_score: Optional[float] = None,
    matched_chunks: Optional[Tuple[str, str]] = None,
    threshold: float = 0.59,
    severity: str = "Low",
    chunk_similarity_matrix: Optional[np.ndarray] = None,
) -> PlagiarismEvidence:
    """
    Build evidence graph for a plagiarism flag.

    Args:
        doc_a: Name of suspected source document.
        doc_b: Name of suspected target document.
        semantic_score: Semantic similarity score.
        lexical_score: Optional lexical similarity score.
        hybrid_score: Optional hybrid similarity score.
        matched_chunks: Tuple of (source_chunk_text, target_chunk_text).
        threshold: Plagiarism threshold that was crossed.
        severity: Assigned severity level.
        chunk_similarity_matrix: Optional matrix of chunk-level similarities.

    Returns:
        PlagiarismEvidence with structured evidence.
    """
    from src.core.config import DEFAULT_THRESHOLDS

    now = datetime.now().isoformat()

    # Build score components
    components = []
    if semantic_score is not None:
        components.append(
            EvidenceScoreComponent(
                component_type="semantic",
                score=semantic_score,
                description="Embedding-based semantic similarity (bi-encoder)",
            )
        )
    if lexical_score is not None:
        components.append(
            EvidenceScoreComponent(
                component_type="lexical",
                score=lexical_score,
                description="Token-overlap lexical similarity (TF-IDF/BM25)",
            )
        )
    if hybrid_score is not None:
        components.append(
            EvidenceScoreComponent(
                component_type="hybrid",
                score=hybrid_score,
                description="Weighted combination of semantic and lexical",
            )
        )

    # Build matched chunk evidence
    matched_list = []
    if matched_chunks:
        source_text, target_text = matched_chunks
        source_chunk = EvidenceChunk(
            doc_name=doc_a,
            chunk_index=0,
            text=source_text,
        )
        target_chunk = EvidenceChunk(
            doc_name=doc_b,
            chunk_index=0,
            text=target_text,
        )

        chunk_match = EvidenceMatch(
            source_chunk=source_chunk,
            target_chunk=target_chunk,
            similarity_score=semantic_score if semantic_score else 0.0,
            component_scores=components.copy(),
        )
        matched_list.append(chunk_match)

    # Final score is semantic by default, or hybrid if available
    final_score = hybrid_score if hybrid_score is not None else semantic_score

    metadata = {}
    if chunk_similarity_matrix is not None:
        metadata["chunk_matrix_shape"] = chunk_similarity_matrix.shape
        metadata["max_chunk_similarity"] = float(np.max(chunk_similarity_matrix))
        metadata["mean_chunk_similarity"] = float(np.mean(chunk_similarity_matrix))

    evidence = PlagiarismEvidence(
        doc_a=doc_a,
        doc_b=doc_b,
        timestamp=now,
        final_score=final_score,
        final_severity=severity,
        threshold_plagiarism=DEFAULT_THRESHOLDS.plagiarism,
        threshold_medium=DEFAULT_THRESHOLDS.medium,
        threshold_high=DEFAULT_THRESHOLDS.high,
        matched_chunks=matched_list,
        score_components=components,
        metadata=metadata,
    )

    return evidence


def validate_evidence(evidence: PlagiarismEvidence) -> Tuple[bool, List[str]]:
    """
    Validate evidence structure for completeness.

    Args:
        evidence: PlagiarismEvidence to validate.

    Returns:
        Tuple of (is_valid, list_of_warnings).
    """
    warnings = []

    if not evidence.doc_a or not evidence.doc_b:
        warnings.append("Missing document names")

    if evidence.final_score < 0.0 or evidence.final_score > 1.0:
        warnings.append(
            f"Final score {evidence.final_score} outside [0.0, 1.0]"
        )

    if not evidence.matched_chunks:
        warnings.append("No matched chunks in evidence")

    if not evidence.score_components:
        warnings.append("No score components recorded")

    if (
        evidence.threshold_plagiarism
        > evidence.threshold_medium
        > evidence.threshold_high
    ):
        warnings.append(
            "Thresholds not in ascending order "
            "(plagiarism <= medium <= high)"
        )

    return len(warnings) == 0, warnings


def merge_evidence(evidence_list: List[PlagiarismEvidence]) -> PlagiarismEvidence:
    """
    Merge multiple evidence graphs (e.g., from multiple chunk pairs).

    Args:
        evidence_list: List of PlagiarismEvidence to merge.

    Returns:
        Merged PlagiarismEvidence with combined chunks and components.
    """
    if not evidence_list:
        raise ValueError("Cannot merge empty evidence list")

    if len(evidence_list) == 1:
        return evidence_list[0]

    # Use first evidence as base
    base = evidence_list[0]

    # Collect all matches and components
    all_matches = []
    all_components = []
    all_metadata = {}

    for ev in evidence_list:
        all_matches.extend(ev.matched_chunks)
        all_components.extend(ev.score_components)
        all_metadata.update(ev.metadata)

    # Sort by similarity score (highest first)
    all_matches.sort(key=lambda m: m.similarity_score, reverse=True)

    # Deduplicate components by type
    seen_types = set()
    dedup_components = []
    for comp in all_components:
        if comp.component_type not in seen_types:
            dedup_components.append(comp)
            seen_types.add(comp.component_type)

    merged = PlagiarismEvidence(
        doc_a=base.doc_a,
        doc_b=base.doc_b,
        timestamp=datetime.now().isoformat(),
        final_score=base.final_score,
        final_severity=base.final_severity,
        threshold_plagiarism=base.threshold_plagiarism,
        threshold_medium=base.threshold_medium,
        threshold_high=base.threshold_high,
        matched_chunks=all_matches,
        score_components=dedup_components,
        metadata=all_metadata,
    )

    return merged
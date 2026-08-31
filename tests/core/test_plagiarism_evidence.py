"""
tests/core/test_plagiarism_evidence.py
---------------------------------------
Tests for plagiarism evidence graphs (Issue #3914).
Verifies evidence building, tracing, validation, and serialization.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.plagiarism_evidence import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceScoreComponent,
    PlagiarismEvidence,
    build_plagiarism_evidence,
    merge_evidence,
    validate_evidence,
)


def test_evidence_chunk_creation():
    """Verify EvidenceChunk metadata storage."""
    chunk = EvidenceChunk(
        doc_name="doc_a",
        chunk_index=0,
        text="The mitochondria is the powerhouse of the cell.",
    )

    assert chunk.doc_name == "doc_a"
    assert chunk.chunk_index == 0
    assert chunk.text == "The mitochondria is the powerhouse of the cell."


def test_evidence_score_component_creation():
    """Verify EvidenceScoreComponent tracks contribution."""
    comp = EvidenceScoreComponent(
        component_type="semantic",
        score=0.85,
        weight=0.7,
        description="Embedding similarity",
    )

    assert comp.component_type == "semantic"
    assert comp.score == 0.85
    assert comp.weight == 0.7


def test_evidence_match_creation():
    """Verify EvidenceMatch pairs chunks with scores."""
    source = EvidenceChunk("doc_a", 0, "Source text")
    target = EvidenceChunk("doc_b", 0, "Target text")

    match = EvidenceMatch(
        source_chunk=source,
        target_chunk=target,
        similarity_score=0.92,
    )

    assert match.source_chunk.doc_name == "doc_a"
    assert match.target_chunk.doc_name == "doc_b"
    assert match.similarity_score == 0.92


def test_build_plagiarism_evidence_semantic_only():
    """Verify evidence building from semantic score."""
    evidence = build_plagiarism_evidence(
        doc_a="essay_1.pdf",
        doc_b="essay_2.pdf",
        semantic_score=0.87,
        matched_chunks=(
            "The mitochondria is the powerhouse.",
            "The mitochondria is the powerhouse.",
        ),
        severity="High",
    )

    assert evidence.doc_a == "essay_1.pdf"
    assert evidence.doc_b == "essay_2.pdf"
    assert evidence.final_score == 0.87
    assert evidence.final_severity == "High"
    assert len(evidence.matched_chunks) == 1
    assert len(evidence.score_components) == 1
    assert evidence.score_components[0].component_type == "semantic"


def test_build_plagiarism_evidence_hybrid():
    """Verify evidence with lexical + semantic + hybrid."""
    evidence = build_plagiarism_evidence(
        doc_a="essay_1",
        doc_b="essay_2",
        semantic_score=0.80,
        lexical_score=0.75,
        hybrid_score=0.78,
        severity="Medium",
    )

    assert evidence.final_score == 0.78  # Uses hybrid
    assert len(evidence.score_components) == 3
    component_types = {c.component_type for c in evidence.score_components}
    assert component_types == {"semantic", "lexical", "hybrid"}


def test_plagiarism_evidence_to_dict():
    """Verify serialization to dictionary."""
    evidence = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        matched_chunks=("Source text", "Target text"),
        severity="High",
    )

    data = evidence.to_dict()

    assert data["doc_a"] == "doc_a"
    assert data["doc_b"] == "doc_b"
    assert data["final_score"] == 0.85
    assert data["final_severity"] == "High"
    assert "matched_chunks" in data
    assert "score_components" in data
    assert "thresholds" in data


def test_plagiarism_evidence_trace_to_chunks():
    """Verify tracing back to source/target chunks."""
    evidence = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        matched_chunks=("Source chunk text", "Target chunk text"),
    )

    traces = evidence.trace_to_chunks()

    assert len(traces) == 1
    source_text, target_text, score = traces[0]
    assert source_text == "Source chunk text"
    assert target_text == "Target chunk text"
    assert score == 0.85


def test_plagiarism_evidence_explain():
    """Verify human-readable explanation generation."""
    evidence = build_plagiarism_evidence(
        doc_a="essay_1",
        doc_b="essay_2",
        semantic_score=0.90,
        lexical_score=0.85,
        matched_chunks=("Text A", "Text B"),
        severity="High",
    )

    explanation = evidence.explain()

    assert "essay_1" in explanation
    assert "essay_2" in explanation
    assert "0.90" in explanation or "High" in explanation
    assert "matched chunks" in explanation


def test_validate_evidence_complete():
    """Verify validation passes for well-formed evidence."""
    evidence = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        matched_chunks=("Text A", "Text B"),
        severity="High",
    )

    is_valid, warnings = validate_evidence(evidence)

    assert is_valid is True
    assert len(warnings) == 0


def test_validate_evidence_missing_chunks():
    """Verify validation warns about missing chunks."""
    evidence = PlagiarismEvidence(
        doc_a="doc_a",
        doc_b="doc_b",
        timestamp="2026-08-29T00:00:00",
        final_score=0.85,
        final_severity="High",
        threshold_plagiarism=0.59,
        threshold_medium=0.75,
        threshold_high=0.90,
        matched_chunks=[],
        score_components=[],
    )

    is_valid, warnings = validate_evidence(evidence)

    assert is_valid is False
    assert any("matched chunks" in w for w in warnings)


def test_validate_evidence_invalid_score():
    """Verify validation catches invalid score ranges."""
    evidence = PlagiarismEvidence(
        doc_a="doc_a",
        doc_b="doc_b",
        timestamp="2026-08-29T00:00:00",
        final_score=1.5,  # Out of range
        final_severity="High",
        threshold_plagiarism=0.59,
        threshold_medium=0.75,
        threshold_high=0.90,
    )

    is_valid, warnings = validate_evidence(evidence)

    assert is_valid is False
    assert any("outside" in w for w in warnings)


def test_merge_evidence_multiple():
    """Verify merging multiple evidence graphs."""
    ev1 = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.90,
        matched_chunks=("Text 1", "Text 1"),
    )

    ev2 = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        lexical_score=0.80,
        matched_chunks=("Text 2", "Text 2"),
    )

    merged = merge_evidence([ev1, ev2])

    assert merged.doc_a == "doc_a"
    assert merged.doc_b == "doc_b"
    assert len(merged.matched_chunks) == 2
    assert merged.matched_chunks[0].similarity_score == 0.90  # Sorted by score


def test_merge_evidence_deduplicates_components():
    """Verify component deduplication during merge."""
    ev1 = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        lexical_score=0.80,
    )

    ev2 = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.82,
        lexical_score=0.78,
    )

    merged = merge_evidence([ev1, ev2])

    component_types = [c.component_type for c in merged.score_components]
    assert component_types.count("semantic") == 1
    assert component_types.count("lexical") == 1


def test_evidence_with_chunk_similarity_matrix():
    """Verify metadata capture from chunk similarity matrix."""
    matrix = np.array([[1.0, 0.9, 0.5], [0.9, 1.0, 0.4], [0.5, 0.4, 1.0]])

    evidence = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        chunk_similarity_matrix=matrix,
    )

    assert evidence.metadata["chunk_matrix_shape"] == (3, 3)
    assert evidence.metadata["max_chunk_similarity"] == 1.0
    assert 0.5 < evidence.metadata["mean_chunk_similarity"] < 0.7


def test_evidence_chunk_text_truncation_in_dict():
    """Verify long text is truncated in serialization."""
    long_text = "A" * 200

    evidence = build_plagiarism_evidence(
        doc_a="doc_a",
        doc_b="doc_b",
        semantic_score=0.85,
        matched_chunks=(long_text, long_text),
    )

    data = evidence.to_dict()
    preview = data["matched_chunks"][0]["source"]["text_preview"]

    assert len(preview) < len(long_text)
    assert preview.endswith("...")
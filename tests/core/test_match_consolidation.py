"""Tests for chunk-level plagiarism match consolidation."""

from __future__ import annotations

from src.core.match_consolidation import (
    ChunkMatch,
    consolidate_chunk_matches,
    consolidated_target_coverage,
)


def _match(
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
    similarity: float,
    source_index: int = 0,
    target_index: int = 0,
) -> ChunkMatch:
    return ChunkMatch(
        source_index=source_index,
        target_index=target_index,
        source_start=source_start,
        source_end=source_end,
        target_start=target_start,
        target_end=target_end,
        similarity=similarity,
        source_text=f"source {source_index}",
        target_text=f"target {target_index}",
    )


def test_overlapping_matches_are_consolidated_and_evidence_is_preserved():
    matches = [
        _match(0, 100, 0, 100, 0.90, 0, 0),
        _match(80, 180, 80, 180, 0.80, 1, 1),
    ]

    segments = consolidate_chunk_matches(
        matches,
        source_length=300,
        target_length=300,
    )

    assert len(segments) == 1
    assert len(segments[0].matches) == 2
    assert segments[0].source_start == 0
    assert segments[0].source_end == 180
    assert segments[0].target_start == 0
    assert segments[0].target_end == 180
    assert segments[0].max_similarity == 0.90


def test_adjacent_matches_are_consolidated_into_one_segment():
    matches = [
        _match(0, 100, 0, 100, 0.85, 0, 0),
        _match(100, 200, 100, 200, 0.75, 1, 1),
    ]

    segments = consolidate_chunk_matches(
        matches,
        source_length=400,
        target_length=400,
    )

    assert len(segments) == 1
    assert segments[0].score == 0.80
    assert segments[0].source_coverage == 0.5
    assert segments[0].target_coverage == 0.5
    assert len(segments[0].matches) == 2


def test_independent_matches_stay_separate_even_with_similar_scores():
    matches = [
        _match(0, 100, 0, 100, 0.90, 0, 0),
        _match(300, 400, 300, 400, 0.90, 3, 3),
    ]

    segments = consolidate_chunk_matches(
        matches,
        source_length=500,
        target_length=500,
    )

    assert len(segments) == 2
    assert consolidated_target_coverage(segments, 500) == 0.4


def test_coverage_uses_union_and_does_not_double_count_overlap():
    matches = [
        _match(0, 120, 0, 120, 0.90, 0, 0),
        _match(80, 200, 80, 200, 0.80, 1, 1),
    ]

    segments = consolidate_chunk_matches(
        matches,
        source_length=400,
        target_length=400,
    )

    assert consolidated_target_coverage(segments, 400) == 0.5
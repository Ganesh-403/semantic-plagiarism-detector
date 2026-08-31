"""
Chunk-level plagiarism match consolidation.

Groups chunk matches that refer to the same source/target passage while
retaining every original match as evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ChunkMatch:
    """One scored source/target chunk pair with document offsets."""

    source_index: int
    target_index: int
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    similarity: float
    source_text: str
    target_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlagiarismSegment:
    """A consolidated source/target region and its original evidence."""

    source_start: int
    source_end: int
    target_start: int
    target_end: int
    score: float
    source_coverage: float
    target_coverage: float
    matches: list[ChunkMatch]

    @property
    def max_similarity(self) -> float:
        """Return the strongest chunk-level similarity in this segment."""
        return max((match.similarity for match in self.matches), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_range": [self.source_start, self.source_end],
            "target_range": [self.target_start, self.target_end],
            "score": round(self.score, 4),
            "max_similarity": round(self.max_similarity, 4),
            "source_coverage": round(self.source_coverage, 4),
            "target_coverage": round(self.target_coverage, 4),
            "matches": [match.to_dict() for match in self.matches],
        }


def _ranges_touch_or_overlap(
    start_a: int, end_a: int, start_b: int, end_b: int
) -> bool:
    """Return True when two half-open ranges overlap or are adjacent."""
    return start_a <= end_b and start_b <= end_a


def _ordered_match(left: ChunkMatch, right: ChunkMatch) -> bool:
    """Reject crossing source/target mappings that belong to different passages."""
    source_direction = right.source_start - left.source_start
    target_direction = right.target_start - left.target_start
    return source_direction == 0 or target_direction == 0 or (
        source_direction > 0 and target_direction > 0
    ) or (source_direction < 0 and target_direction < 0)


def _compatible(left: ChunkMatch, right: ChunkMatch) -> bool:
    """Return whether two matches can belong to one plagiarism segment."""
    return (
        _ranges_touch_or_overlap(
            left.source_start,
            left.source_end,
            right.source_start,
            right.source_end,
        )
        and _ranges_touch_or_overlap(
            left.target_start,
            left.target_end,
            right.target_start,
            right.target_end,
        )
        and _ordered_match(left, right)
    )


def _union_length(ranges: Sequence[tuple[int, int]]) -> int:
    """Return the length covered by the union of half-open ranges."""
    if not ranges:
        return 0

    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return sum(end - start for start, end in merged)


def _build_segment(
    matches: list[ChunkMatch],
    source_length: int,
    target_length: int,
) -> PlagiarismSegment:
    """Build one segment from compatible chunk matches."""
    source_start = min(match.source_start for match in matches)
    source_end = max(match.source_end for match in matches)
    target_start = min(match.target_start for match in matches)
    target_end = max(match.target_end for match in matches)

    weights = [max(1, match.target_end - match.target_start) for match in matches]
    weight_total = sum(weights)
    score = sum(
        match.similarity * weight
        for match, weight in zip(matches, weights)
    ) / weight_total

    source_covered = _union_length(
        [(match.source_start, match.source_end) for match in matches]
    )
    target_covered = _union_length(
        [(match.target_start, match.target_end) for match in matches]
    )

    return PlagiarismSegment(
        source_start=source_start,
        source_end=source_end,
        target_start=target_start,
        target_end=target_end,
        score=score,
        source_coverage=source_covered / source_length if source_length else 0.0,
        target_coverage=target_covered / target_length if target_length else 0.0,
        matches=matches,
    )


def chunk_offsets(chunks: Sequence[Any]) -> tuple[list[tuple[int, int]], int]:
    """Return chunk character ranges and the best available document length."""
    offsets: list[tuple[int, int]] = []
    fallback_start = 0

    has_positions = all(
        getattr(chunk, "char_start", None) is not None
        and getattr(chunk, "char_end", None) is not None
        and getattr(chunk, "char_end", 0) > getattr(chunk, "char_start", 0)
        for chunk in chunks
    )

    for chunk in chunks:
        text = chunk.text if hasattr(chunk, "text") else str(chunk)

        if has_positions:
            start = int(chunk.char_start)
            end = int(chunk.char_end)
        else:
            start = fallback_start
            end = start + len(text)

        offsets.append((start, end))
        fallback_start = max(fallback_start, end)

    return offsets, fallback_start


def consolidate_chunk_matches(
    matches: Sequence[ChunkMatch],
    source_length: int,
    target_length: int,
) -> list[PlagiarismSegment]:
    """Merge overlapping/adjacent matches while keeping unrelated passages separate.

    Matching requires overlap or adjacency on both documents and preserves
    source-to-target order, so high similarity alone cannot cause a merge.
    """
    if not matches:
        return []

    ordered = sorted(
        matches,
        key=lambda match: (
            match.source_start,
            match.target_start,
            match.source_end,
            match.target_end,
        ),
    )

    groups: list[list[ChunkMatch]] = []

    for match in ordered:
        if not groups or not any(
            _compatible(existing, match) for existing in groups[-1]
        ):
            groups.append([match])
        else:
            groups[-1].append(match)

    return [
        _build_segment(group, source_length, target_length)
        for group in groups
    ]


def consolidated_target_coverage(
    segments: Sequence[PlagiarismSegment],
    target_length: int,
) -> float:
    """Calculate document coverage from the consolidated target ranges."""
    if not segments or target_length <= 0:
        return 0.0

    covered = _union_length(
        [(segment.target_start, segment.target_end) for segment in segments]
    )

    return min(1.0, covered / target_length)
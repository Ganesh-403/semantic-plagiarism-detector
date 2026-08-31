"""
dedup_engine.py
---------------
Document fingerprinting and near-duplicate detection using MinHash
Locality-Sensitive Hashing (LSH). Identifies near-duplicate document
pairs in the corpus without requiring expensive pairwise embedding
comparisons — ideal for large-scale deduplication passes.

Algorithm:
  1. Each document is split into character-level shingles (n-grams).
  2. A MinHash signature is computed from the shingle set.
  3. MinHash signatures are compared via Jaccard estimation.
  4. Pairs exceeding the similarity threshold are reported.

The MinHash implementation uses a fixed set of hash functions derived
from a single hash with different seeds, avoiding external dependencies
beyond hashlib and numpy (both already in requirements.txt).
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SHINGLE_SIZE = 5
DEFAULT_NUM_HASH_FUNCS = 128
DEFAULT_JACCARD_THRESHOLD = 0.5


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DuplicatePair:
    """A pair of near-duplicate documents with their similarity score."""
    doc_a: str
    doc_b: str
    jaccard_estimate: float
    shared_shingles: int
    total_shingles_a: int
    total_shingles_b: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DedupReport:
    """Summary report from a deduplication pass."""
    total_documents: int
    total_pairs_checked: int
    duplicate_pairs_found: int
    avg_similarity: float
    max_similarity: float
    duplicates: List[DuplicatePair]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_pairs_checked": self.total_pairs_checked,
            "duplicate_pairs_found": self.duplicate_pairs_found,
            "avg_similarity": self.avg_similarity,
            "max_similarity": self.max_similarity,
            "duplicates": [d.to_dict() for d in self.duplicates],
        }


# ── Shingling ─────────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_shingles(text: str, shingle_size: int = DEFAULT_SHINGLE_SIZE) -> FrozenSet[str]:
    """Generate character-level shingles from normalized text.

    Args:
        text: Raw document text.
        shingle_size: Length of each shingle (character n-gram).

    Returns:
        Immutable set of shingle strings.
    """
    normalized = _normalize_text(text)
    if len(normalized) < shingle_size:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(
        normalized[i : i + shingle_size]
        for i in range(len(normalized) - shingle_size + 1)
    )


# ── MinHash ───────────────────────────────────────────────────────────────────


def _hash_shingle(shingle: str, seed: int) -> int:
    """Hash a single shingle with a given seed, returning a 32-bit int."""
    h = hashlib.blake2b(shingle.encode("utf-8"), digest_size=4, salt=struct.pack("<I", seed))
    return struct.unpack("<I", h.digest())[0]


class MinHashSignature:
    """Compute and store a MinHash signature for a shingle set."""

    __slots__ = ("num_hashes", "signature", "shingle_count")

    def __init__(self, shingles: FrozenSet[str], num_hashes: int = DEFAULT_NUM_HASH_FUNCS) -> None:
        self.num_hashes = num_hashes
        self.shingle_count = len(shingles)

        if not shingles:
            self.signature = np.full(num_hashes, np.iinfo(np.uint32).max, dtype=np.uint32)
        else:
            self.signature = np.empty(num_hashes, dtype=np.uint32)
            for i in range(num_hashes):
                self.signature[i] = min(
                    _hash_shingle(s, i) for s in shingles
                )

    def estimate_jaccard(self, other: MinHashSignature) -> float:
        """Estimate Jaccard similarity between this and another MinHash signature."""
        if self.num_hashes != other.num_hashes:
            raise ValueError("Signatures must use the same number of hash functions.")
        if self.num_hashes == 0:
            return 0.0
        matches = int(np.sum(self.signature == other.signature))
        return matches / self.num_hashes


# ── LSH Banding ───────────────────────────────────────────────────────────────


def _lsh_candidates(
    signatures: Dict[str, MinHashSignature],
    num_bands: int,
    rows_per_band: int,
) -> Set[Tuple[str, str]]:
    """Generate candidate pairs using LSH banding technique.

    Documents that share at least one band bucket are candidate pairs.
    This avoids O(n^2) full comparison.
    """
    num_hashes = num_bands * rows_per_band
    buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    for doc_name, sig in signatures.items():
        if sig.num_hashes < num_hashes:
            continue
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_hash = hash(sig.signature[start:end].tobytes())
            buckets[(band_idx, band_hash)].append(doc_name)

    candidates: Set[Tuple[str, str]] = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) < 2:
            continue
        bucket_docs_sorted = sorted(set(bucket_docs))
        for i in range(len(bucket_docs_sorted)):
            for j in range(i + 1, len(bucket_docs_sorted)):
                candidates.add((bucket_docs_sorted[i], bucket_docs_sorted[j]))

    return candidates


# ── Core deduplication ───────────────────────────────────────────────────────


def compute_minhash_signatures(
    documents: Dict[str, str],
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    num_hashes: int = DEFAULT_NUM_HASH_FUNCS,
) -> Dict[str, MinHashSignature]:
    """Compute MinHash signatures for a collection of documents.

    Args:
        documents: Dict mapping document name to raw text.
        shingle_size: Character shingle length.
        num_hashes: Number of hash functions for MinHash.

    Returns:
        Dict mapping document name to MinHashSignature.
    """
    signatures: Dict[str, MinHashSignature] = {}
    for name, text in documents.items():
        shingles = generate_shingles(text, shingle_size)
        signatures[name] = MinHashSignature(shingles, num_hashes)
    return signatures


def find_near_duplicates(
    documents: Dict[str, str],
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    num_hashes: int = DEFAULT_NUM_HASH_FUNCS,
    use_lsh: bool = True,
    lsh_bands: int = 16,
) -> DedupReport:
    """Find near-duplicate document pairs using MinHash + optional LSH.

    Args:
        documents: Dict mapping document name to raw text.
        threshold: Minimum Jaccard estimate to consider a duplicate.
        shingle_size: Character shingle length.
        num_hashes: Number of hash functions for MinHash.
        use_lsh: Whether to use LSH banding for speedup.
        lsh_bands: Number of LSH bands (must divide num_hashes).

    Returns:
        DedupReport with all detected duplicate pairs.
    """
    if len(documents) < 2:
        return DedupReport(
            total_documents=len(documents),
            total_pairs_checked=0,
            duplicate_pairs_found=0,
            avg_similarity=0.0,
            max_similarity=0.0,
            duplicates=[],
        )

    # Compute signatures
    sigs = compute_minhash_signatures(documents, shingle_size, num_hashes)
    doc_names = sorted(sigs.keys())

    # Generate candidate pairs
    if use_lsh and len(doc_names) > 50:
        rows_per_band = num_hashes // lsh_bands
        candidates = _lsh_candidates(sigs, lsh_bands, rows_per_band)
        logger.info("LSH generated %d candidate pairs from %d documents.", len(candidates), len(doc_names))
    else:
        # Brute force for small collections
        candidates = set()
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                candidates.add((doc_names[i], doc_names[j]))

    # Evaluate candidates
    duplicates: List[DuplicatePair] = []
    shingle_cache: Dict[str, FrozenSet[str]] = {}
    for name, text in documents.items():
        shingle_cache[name] = generate_shingles(text, shingle_size)

    for doc_a, doc_b in candidates:
        jaccard_est = sigs[doc_a].estimate_jaccard(sigs[doc_b])
        if jaccard_est >= threshold:
            shared = len(shingle_cache[doc_a] & shingle_cache[doc_b])
            duplicates.append(DuplicatePair(
                doc_a=doc_a,
                doc_b=doc_b,
                jaccard_estimate=round(jaccard_est, 4),
                shared_shingles=shared,
                total_shingles_a=sigs[doc_a].shingle_count,
                total_shingles_b=sigs[doc_b].shingle_count,
            ))

    duplicates.sort(key=lambda d: d.jaccard_estimate, reverse=True)

    similarities = [d.jaccard_estimate for d in duplicates]
    return DedupReport(
        total_documents=len(documents),
        total_pairs_checked=len(candidates),
        duplicate_pairs_found=len(duplicates),
        avg_similarity=round(float(np.mean(similarities)), 4) if similarities else 0.0,
        max_similarity=round(float(np.max(similarities)), 4) if similarities else 0.0,
        duplicates=duplicates,
    )


# ── Batch deduplication from corpus DB ───────────────────────────────────────


def deduplicate_corpus(
    chunk_texts: Dict[str, List[str]],
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    num_hashes: int = DEFAULT_NUM_HASH_FUNCS,
) -> DedupReport:
    """Run deduplication on corpus chunks (document-level concatenated text).

    Args:
        chunk_texts: Dict mapping filename to list of chunk texts.
        threshold: Minimum Jaccard estimate to flag as duplicate.
        shingle_size: Character shingle length.
        num_hashes: Number of hash functions.

    Returns:
        DedupReport with duplicate pairs.
    """
    # Concatenate chunks per document for document-level fingerprinting
    docs = {
        name: " ".join(chunks)
        for name, chunks in chunk_texts.items()
        if chunks
    }
    return find_near_duplicates(
        docs, threshold=threshold, shingle_size=shingle_size, num_hashes=num_hashes,
    )


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_dedup_stats(report: DedupReport) -> Dict[str, Any]:
    """Compute aggregate statistics from a dedup report.

    Returns:
        Dict with total_documents, duplicate_rate, avg_jaccard,
        severity_breakdown, most_duplicated_doc.
    """
    if report.duplicate_pairs_found == 0:
        return {
            "total_documents": report.total_documents,
            "duplicate_rate": 0.0,
            "avg_jaccard": 0.0,
            "severity_breakdown": {"High": 0, "Medium": 0, "Low": 0},
            "most_duplicated_doc": None,
        }

    # Count how many times each doc appears in duplicate pairs
    doc_counts: Dict[str, int] = defaultdict(int)
    severity = {"High": 0, "Medium": 0, "Low": 0}

    for dup in report.duplicates:
        doc_counts[dup.doc_a] += 1
        doc_counts[dup.doc_b] += 1
        if dup.jaccard_estimate >= 0.80:
            severity["High"] += 1
        elif dup.jaccard_estimate >= 0.60:
            severity["Medium"] += 1
        else:
            severity["Low"] += 1

    most_dup = max(doc_counts.items(), key=lambda x: x[1]) if doc_counts else None

    # Estimate unique docs (greedy: remove one doc from each pair)
    seen = set()
    for dup in report.duplicates:
        seen.add(dup.doc_a)
        seen.add(dup.doc_b)
    # Documents not in any duplicate pair
    unique_count = report.total_documents - len(seen) + report.duplicate_pairs_found

    return {
        "total_documents": report.total_documents,
        "duplicate_rate": round(report.duplicate_pairs_found / max(1, report.total_documents), 4),
        "avg_jaccard": report.avg_similarity,
        "severity_breakdown": severity,
        "most_duplicated_doc": {"name": most_dup[0], "pair_count": most_dup[1]} if most_dup else None,
        "unique_estimated": max(1, unique_count),
    }

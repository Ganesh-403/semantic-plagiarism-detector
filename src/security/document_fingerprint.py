"""
Document Fingerprinting & Deduplication Engine
================================================
Generates content-based fingerprints for documents and detects exact
and near-duplicate uploads across the corpus. Provides multiple
fingerprinting algorithms for different accuracy/speed trade-offs.

Provides:
  - MinHash fingerprinting for Jaccard similarity estimation
  - SimHash (locality-sensitive hashing) for near-duplicate detection
  - Trigram-based fingerprinting for fast exact-match screening
  - Persistent fingerprint store with dedup history
  - Configurable similarity thresholds per algorithm
  - Cluster extraction for duplicate document groups
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ── Enums & Constants ─────────────────────────────────────────────────────────

class FingerprintMethod(Enum):
    """Supported fingerprinting algorithms."""
    MINHASH = "minhash"
    SIMHASH = "simhash"
    TRIGRAM = "trigram"
    SHA256 = "sha256"


class MatchType(Enum):
    """Classification of duplicate match strength."""
    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"
    SIMILAR = "similar"
    UNIQUE = "unique"


# Default thresholds
DEFAULT_MINHASH_NUM_PERM = 128
DEFAULT_MINHASH_THRESHOLD = 0.85
DEFAULT_SIMHASH_BITS = 64
DEFAULT_SIMHASH_HAMMING_THRESHOLD = 3
DEFAULT_TRIGRAM_THRESHOLD = 0.90
DEFAULT_SHINGLE_SIZE = 3


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DocumentFingerprint:
    """A computed fingerprint for a single document."""
    document_id: str
    sha256_hash: str
    minhash_signature: Optional[Tuple[int, ...]] = None
    simhash_value: Optional[int] = None
    trigram_set: Optional[FrozenSet[str]] = None
    word_count: int = 0
    char_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateMatch:
    """A detected duplicate or near-duplicate relationship."""
    source_id: str
    target_id: str
    match_type: MatchType
    minhash_similarity: float = 0.0
    simhash_hamming_distance: int = 0
    trigram_overlap: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateCluster:
    """A group of documents determined to be duplicates of each other."""
    cluster_id: int
    document_ids: List[str]
    representative_id: str
    cluster_size: int = 0
    internal_similarity: float = 0.0
    match_type: MatchType = MatchType.UNIQUE

    def __post_init__(self):
        self.cluster_size = len(self.document_ids)


@dataclass
class DedupReport:
    """Summary report of a deduplication scan."""
    total_documents: int
    unique_documents: int
    duplicate_count: int
    near_duplicate_count: int
    exact_duplicate_count: int
    clusters: List[DuplicateCluster]
    matches: List[DuplicateMatch]
    scan_duration_ms: float = 0.0


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split into tokens."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return tokens


def _shingle_tokens(tokens: List[str], size: int) -> List[str]:
    """Generate n-gram shingles from a token list."""
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]


def _trigrams(text: str) -> Set[str]:
    """Extract character-level trigrams from text."""
    text = text.lower().strip()
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


# ── MinHash Implementation ────────────────────────────────────────────────────

class MinHash:
    """MinHash implementation for Jaccard similarity estimation.

    Uses the standard random hash function approach with optimal
    parameter selection for the given number of permutations.
    """

    def __init__(self, num_perm: int = DEFAULT_MINHASH_NUM_PERM, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        self._hash_funcs = self._generate_hash_functions()

    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        """Generate pairwise independent hash functions: h(x) = (a*x + b) mod p."""
        prime = (1 << 61) - 1  # Mersenne prime
        rng = _SeededRandom(self.seed)
        funcs = []
        for _ in range(self.num_perm):
            a = rng.randint(1, prime - 1)
            b = rng.randint(0, prime - 1)
            funcs.append((a, b))
        return funcs

    def compute(self, tokens: Sequence[str]) -> Tuple[int, ...]:
        """Compute the MinHash signature for a set of tokens.

        Args:
            tokens: Tokenized document content.

        Returns:
            Tuple of min-hash values, one per permutation.
        """
        prime = (1 << 61) - 1
        signature = [float("inf")] * self.num_perm

        shingles = set(tokens) if len(tokens) <= 10000 else set(tokens[:10000])

        for token in shingles:
            token_hash = int.from_bytes(
                hashlib.md5(token.encode("utf-8")).digest()[:8], "big"
            )
            for i, (a, b) in enumerate(self._hash_funcs):
                h = (a * token_hash + b) % prime
                if h < signature[i]:
                    signature[i] = h

        return tuple(int(s) for s in signature)

    @staticmethod
    def jaccard_estimate(
        sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]
    ) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig_a) != len(sig_b) or len(sig_a) == 0:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


# ── SimHash Implementation ────────────────────────────────────────────────────

class SimHash:
    """SimHash (locality-sensitive hashing) for near-duplicate detection.

    Maps high-dimensional feature vectors to a compact binary fingerprint
    such that similar documents produce fingerprints with small Hamming
    distance.
    """

    def __init__(self, num_bits: int = DEFAULT_SIMHASH_BITS, seed: int = 42):
        self.num_bits = num_bits
        self.seed = seed
        self._bit_positions = self._generate_bit_positions()

    def _generate_bit_positions(self) -> List[List[int]]:
        """Generate random bit positions for each feature hash."""
        rng = _SeededRandom(self.seed)
        positions = []
        for _ in range(self.num_bits):
            positions.append([rng.randint(0, self.num_bits - 1) for _ in range(1)])
        return positions

    def compute(self, tokens: Sequence[str]) -> int:
        """Compute SimHash fingerprint from tokens.

        Args:
            tokens: Tokenized document content.

        Returns:
            Integer fingerprint (bit vector).
        """
        if not tokens:
            return 0

        # Count token frequencies
        counts = Counter(tokens)
        total = sum(counts.values())
        if total == 0:
            return 0

        # Weighted feature hashing
        fingerprint = [0.0] * self.num_bits

        for token, count in counts.items():
            weight = count / total
            token_hash = int.from_bytes(
                hashlib.md5(token.encode("utf-8")).digest()[:8], "big"
            )

            for bit in range(self.num_bits):
                # Use bit position from token hash
                bit_val = (token_hash >> bit) & 1
                if bit_val:
                    fingerprint[bit] += weight
                else:
                    fingerprint[bit] -= weight

        # Convert to integer
        result = 0
        for bit in range(self.num_bits):
            if fingerprint[bit] > 0:
                result |= (1 << bit)

        return result

    @staticmethod
    def hamming_distance(hash_a: int, hash_b: int) -> int:
        """Compute Hamming distance between two SimHash fingerprints."""
        xor = hash_a ^ hash_b
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1  # Clear lowest set bit
        return distance


# ── Seeded Random ─────────────────────────────────────────────────────────────

class _SeededRandom:
    """Simple seeded pseudo-random number generator (xorshift64)."""

    def __init__(self, seed: int):
        self.state = seed if seed != 0 else 1

    def randint(self, lo: int, hi: int) -> int:
        """Return a random integer in [lo, hi]."""
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        self.state &= 0xFFFFFFFFFFFFFFFF
        return lo + (self.state % (hi - lo + 1))


# ── Trigram Fingerprint ───────────────────────────────────────────────────────

def compute_trigram_fingerprint(text: str) -> FrozenSet[str]:
    """Compute a frozen set of character trigrams."""
    return frozenset(_trigrams(text))


def trigram_jaccard(set_a: FrozenSet[str], set_b: FrozenSet[str]) -> float:
    """Compute Jaccard similarity between two trigram sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ── Fingerprint Store ─────────────────────────────────────────────────────────

class FingerprintStore:
    """Persistent store for document fingerprints with dedup queries.

    Maintains an in-memory index backed by optional disk persistence.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._fingerprints: Dict[str, DocumentFingerprint] = {}
        self._simhash_index: Dict[int, List[str]] = defaultdict(list)
        self._minhash_signatures: List[Tuple[str, Tuple[int, ...]]] = []
        self._trigram_sets: Dict[str, FrozenSet[str]] = {}

    @property
    def size(self) -> int:
        """Number of stored fingerprints."""
        return len(self._fingerprints)

    def add(self, fingerprint: DocumentFingerprint) -> None:
        """Add a fingerprint to the store."""
        doc_id = fingerprint.document_id
        self._fingerprints[doc_id] = fingerprint

        if fingerprint.simhash_value is not None:
            self._simhash_index[fingerprint.simhash_value].append(doc_id)

        if fingerprint.minhash_signature is not None:
            self._minhash_signatures.append(
                (doc_id, fingerprint.minhash_signature)
            )

        if fingerprint.trigram_set is not None:
            self._trigram_sets[doc_id] = fingerprint.trigram_set

    def get(self, document_id: str) -> Optional[DocumentFingerprint]:
        """Retrieve a fingerprint by document ID."""
        return self._fingerprints.get(document_id)

    def remove(self, document_id: str) -> bool:
        """Remove a fingerprint from the store. Returns True if found."""
        fp = self._fingerprints.pop(document_id, None)
        if fp is None:
            return False

        if fp.simhash_value is not None:
            bucket = self._simhash_index.get(fp.simhash_value, [])
            if document_id in bucket:
                bucket.remove(document_id)
            if not bucket:
                self._simhash_index.pop(fp.simhash_value, None)

        self._minhash_signatures = [
            (did, sig) for did, sig in self._minhash_signatures if did != document_id
        ]
        self._trigram_sets.pop(document_id, None)
        return True

    def list_ids(self) -> List[str]:
        """Return all stored document IDs."""
        return list(self._fingerprints.keys())

    def get_all(self) -> Dict[str, DocumentFingerprint]:
        """Return all stored fingerprints."""
        return dict(self._fingerprints)

    def find_exact_duplicates(self) -> List[DuplicateCluster]:
        """Find exact duplicates by SHA-256 hash."""
        hash_groups: Dict[str, List[str]] = defaultdict(list)
        for doc_id, fp in self._fingerprints.items():
            hash_groups[fp.sha256_hash].append(doc_id)

        clusters = []
        cluster_id = 0
        for sha_hash, doc_ids in hash_groups.items():
            if len(doc_ids) > 1:
                clusters.append(DuplicateCluster(
                    cluster_id=cluster_id,
                    document_ids=sorted(doc_ids),
                    representative_id=doc_ids[0],
                    internal_similarity=1.0,
                    match_type=MatchType.EXACT,
                ))
                cluster_id += 1

        return clusters

    def save(self, path: Optional[str] = None) -> None:
        """Persist the fingerprint store to disk as JSON."""
        save_path = path or self.storage_path
        if not save_path:
            return

        data = {}
        for doc_id, fp in self._fingerprints.items():
            entry = {
                "document_id": fp.document_id,
                "sha256_hash": fp.sha256_hash,
                "word_count": fp.word_count,
                "char_count": fp.char_count,
                "created_at": fp.created_at,
                "metadata": fp.metadata,
            }
            if fp.minhash_signature is not None:
                entry["minhash_signature"] = list(fp.minhash_signature)
            if fp.simhash_value is not None:
                entry["simhash_value"] = fp.simhash_value
            if fp.trigram_set is not None:
                entry["trigram_set"] = sorted(fp.trigram_set)
            data[doc_id] = entry

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %d fingerprints to %s", len(data), save_path)

    def load(self, path: Optional[str] = None) -> int:
        """Load fingerprints from a JSON file. Returns count loaded."""
        load_path = path or self.storage_path
        if not load_path or not os.path.exists(load_path):
            return 0

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded = 0
        for doc_id, entry in data.items():
            try:
                fp = DocumentFingerprint(
                    document_id=entry["document_id"],
                    sha256_hash=entry["sha256_hash"],
                    minhash_signature=tuple(entry["minhash_signature"])
                        if "minhash_signature" in entry else None,
                    simhash_value=entry.get("simhash_value"),
                    trigram_set=frozenset(entry["trigram_set"])
                        if "trigram_set" in entry else None,
                    word_count=entry.get("word_count", 0),
                    char_count=entry.get("char_count", 0),
                    created_at=entry.get("created_at", 0.0),
                    metadata=entry.get("metadata", {}),
                )
                self.add(fp)
                loaded += 1
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed fingerprint entry '%s': %s", doc_id, exc)

        logger.info("Loaded %d fingerprints from %s", loaded, load_path)
        return loaded

    def clear(self) -> None:
        """Remove all stored fingerprints."""
        self._fingerprints.clear()
        self._simhash_index.clear()
        self._minhash_signatures.clear()
        self._trigram_sets.clear()


# ── Fingerprinting Engine ─────────────────────────────────────────────────────

class DocumentFingerprintEngine:
    """Main engine for computing document fingerprints and detecting duplicates.

    Supports multiple fingerprinting methods that can be used individually
    or combined for a multi-stage deduplication pipeline.
    """

    def __init__(
        self,
        minhash_perms: int = DEFAULT_MINHASH_NUM_PERM,
        minhash_threshold: float = DEFAULT_MINHASH_THRESHOLD,
        simhash_bits: int = DEFAULT_SIMHASH_BITS,
        simhash_hamming_threshold: int = DEFAULT_SIMHASH_HAMMING_THRESHOLD,
        trigram_threshold: float = DEFAULT_TRIGRAM_THRESHOLD,
        shingle_size: int = DEFAULT_SHINGLE_SIZE,
        storage_path: Optional[str] = None,
    ):
        self.minhash_perms = minhash_perms
        self.minhash_threshold = minhash_threshold
        self.simhash_bits = simhash_bits
        self.simhash_hamming_threshold = simhash_hamming_threshold
        self.trigram_threshold = trigram_threshold
        self.shingle_size = shingle_size

        self._minhash = MinHash(num_perm=minhash_perms)
        self._simhash = SimHash(num_bits=simhash_bits)
        self.store = FingerprintStore(storage_path=storage_path)

    def compute_fingerprint(
        self,
        text: str,
        document_id: str,
        methods: Optional[List[FingerprintMethod]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentFingerprint:
        """Compute a fingerprint for the given text.

        Args:
            text: Raw document text content.
            document_id: Unique identifier for this document.
            methods: Which fingerprint methods to compute. Defaults to all.
            metadata: Optional metadata to attach to the fingerprint.

        Returns:
            DocumentFingerprint with the requested components.
        """
        if methods is None:
            methods = list(FingerprintMethod)

        # Always compute SHA-256
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        tokens = _tokenize(text)
        words = text.split()

        minhash_sig = None
        simhash_val = None
        trigram_s = None

        if FingerprintMethod.MINHASH in methods or FingerprintMethod.SHA256 not in methods:
            shingles = _shingle_tokens(tokens, self.shingle_size)
            minhash_sig = self._minhash.compute(shingles)

        if FingerprintMethod.SIMHASH in methods or FingerprintMethod.SHA256 not in methods:
            simhash_val = self._simhash.compute(tokens)

        if FingerprintMethod.TRIGRAM in methods or FingerprintMethod.SHA256 not in methods:
            trigram_s = compute_trigram_fingerprint(text)

        fingerprint = DocumentFingerprint(
            document_id=document_id,
            sha256_hash=sha256,
            minhash_signature=minhash_sig,
            simhash_value=simhash_val,
            trigram_set=trigram_s,
            word_count=len(words),
            char_count=len(text),
            metadata=metadata or {},
        )

        self.store.add(fingerprint)
        return fingerprint

    def find_duplicates(
        self,
        text: str,
        document_id: Optional[str] = None,
        methods: Optional[List[FingerprintMethod]] = None,
    ) -> List[DuplicateMatch]:
        """Find duplicates of the given text against the stored corpus.

        Args:
            text: Document text to check.
            document_id: ID of the document being checked (excluded from results).
            methods: Which methods to use for matching.

        Returns:
            List of DuplicateMatch objects sorted by overall_score descending.
        """
        if methods is None:
            methods = list(FingerprintMethod)

        fp = self.compute_fingerprint(text, document_id or f"query_{id(text)}", methods)
        matches: List[DuplicateMatch] = []

        for stored_id, stored_fp in self.store.get_all().items():
            if stored_id == fp.document_id:
                continue

            match = self._compare_fingerprints(fp, stored_fp, methods)
            if match and match.overall_score > 0:
                matches.append(match)

        matches.sort(key=lambda m: m.overall_score, reverse=True)
        return matches

    def _compare_fingerprints(
        self,
        fp_a: DocumentFingerprint,
        fp_b: DocumentFingerprint,
        methods: List[FingerprintMethod],
    ) -> Optional[DuplicateMatch]:
        """Compare two fingerprints and return a match if similarity is above threshold."""
        # Exact match
        if fp_a.sha256_hash == fp_b.sha256_hash:
            return DuplicateMatch(
                source_id=fp_a.document_id,
                target_id=fp_b.document_id,
                match_type=MatchType.EXACT,
                overall_score=1.0,
                details={"sha256_match": True},
            )

        scores = {}
        details = {}

        # MinHash comparison
        if (FingerprintMethod.MINHASH in methods
                and fp_a.minhash_signature is not None
                and fp_b.minhash_signature is not None):
            mh_sim = MinHash.jaccard_estimate(
                fp_a.minhash_signature, fp_b.minhash_signature
            )
            scores["minhash"] = mh_sim
            details["minhash_similarity"] = round(mh_sim, 4)

        # SimHash comparison
        if (FingerprintMethod.SIMHASH in methods
                and fp_a.simhash_value is not None
                and fp_b.simhash_value is not None):
            hamming = SimHash.hamming_distance(fp_a.simhash_value, fp_b.simhash_value)
            # Normalize to [0, 1] similarity
            sim_sim = max(0.0, 1.0 - hamming / self.simhash_bits)
            scores["simhash"] = sim_sim
            details["simhash_hamming_distance"] = hamming
            details["simhash_similarity"] = round(sim_sim, 4)

        # Trigram comparison
        if (FingerprintMethod.TRIGRAM in methods
                and fp_a.trigram_set is not None
                and fp_b.trigram_set is not None):
            tri_sim = trigram_jaccard(fp_a.trigram_set, fp_b.trigram_set)
            scores["trigram"] = tri_sim
            details["trigram_similarity"] = round(tri_sim, 4)

        if not scores:
            return None

        # Weighted average
        weights = {"minhash": 0.4, "simhash": 0.35, "trigram": 0.25}
        total_weight = sum(weights.get(k, 0) for k in scores)
        if total_weight == 0:
            return None

        overall = sum(scores[k] * weights.get(k, 0) for k in scores) / total_weight

        # Determine match type
        if overall >= 0.99:
            match_type = MatchType.EXACT
        elif overall >= self.minhash_threshold:
            match_type = MatchType.NEAR_DUPLICATE
        elif overall >= self.trigram_threshold:
            match_type = MatchType.SIMILAR
        else:
            return None  # Below all thresholds

        return DuplicateMatch(
            source_id=fp_a.document_id,
            target_id=fp_b.document_id,
            match_type=match_type,
            minhash_similarity=scores.get("minhash", 0.0),
            simhash_hamming_distance=details.get("simhash_hamming_distance", 0),
            trigram_overlap=scores.get("trigram", 0.0),
            overall_score=round(overall, 4),
            details=details,
        )

    def scan_corpus(self, method: Optional[FingerprintMethod] = None) -> DedupReport:
        """Scan the entire stored corpus for duplicates.

        Args:
            method: Optional primary method for the scan.

        Returns:
            DedupReport with all detected clusters and matches.
        """
        start_time = time.monotonic()
        all_matches: List[DuplicateMatch] = []

        # Exact duplicate detection via SHA-256
        exact_clusters = self.store.find_exact_duplicates()

        # Pairwise comparison for near-duplicates
        fingerprints = list(self.store.get_all().values())
        n = len(fingerprints)

        for i in range(n):
            for j in range(i + 1, n):
                match = self._compare_fingerprints(
                    fingerprints[i], fingerprints[j], list(FingerprintMethod)
                )
                if match and match.overall_score > 0:
                    all_matches.append(match)

        # Build near-duplicate clusters using union-find
        near_clusters = self._build_clusters(all_matches)

        # Combine all clusters
        all_clusters = []
        for i, cluster in enumerate(exact_clusters):
            cluster.cluster_id = i
            all_clusters.append(cluster)

        offset = len(exact_clusters)
        for i, cluster in enumerate(near_clusters):
            cluster.cluster_id = offset + i
            all_clusters.append(cluster)

        duration = (time.monotonic() - start_time) * 1000

        # Count statistics
        exact_count = sum(1 for m in all_matches if m.match_type == MatchType.EXACT)
        near_count = sum(1 for m in all_matches if m.match_type == MatchType.NEAR_DUPLICATE)

        # Documents in any duplicate cluster
        dup_docs = set()
        for cluster in all_clusters:
            dup_docs.update(cluster.document_ids)

        return DedupReport(
            total_documents=n,
            unique_documents=n - len(dup_docs) + len(all_clusters),
            duplicate_count=len(dup_docs),
            near_duplicate_count=near_count,
            exact_duplicate_count=exact_count,
            clusters=all_clusters,
            matches=all_matches,
            scan_duration_ms=round(duration, 2),
        )

    def _build_clusters(self, matches: List[DuplicateMatch]) -> List[DuplicateCluster]:
        """Build clusters from matches using union-find."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for match in matches:
            if match.overall_score >= self.minhash_threshold:
                union(match.source_id, match.target_id)

        # Group by root
        groups: Dict[str, List[str]] = defaultdict(list)
        all_ids = set()
        for match in matches:
            all_ids.add(match.source_id)
            all_ids.add(match.target_id)

        for doc_id in all_ids:
            root = find(doc_id)
            groups[root].append(doc_id)

        clusters = []
        for root, members in groups.items():
            if len(members) > 1:
                # Compute average internal similarity
                sim_scores = []
                for match in matches:
                    if match.source_id in members and match.target_id in members:
                        sim_scores.append(match.overall_score)
                avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

                clusters.append(DuplicateCluster(
                    cluster_id=0,
                    document_ids=sorted(members),
                    representative_id=root,
                    internal_similarity=round(avg_sim, 4),
                    match_type=MatchType.NEAR_DUPLICATE,
                ))

        clusters.sort(key=lambda c: c.cluster_size, reverse=True)
        return clusters

    def export_report_json(self, report: DedupReport) -> str:
        """Export a DedupReport as a JSON string."""
        return json.dumps({
            "total_documents": report.total_documents,
            "unique_documents": report.unique_documents,
            "duplicate_count": report.duplicate_count,
            "near_duplicate_count": report.near_duplicate_count,
            "exact_duplicate_count": report.exact_duplicate_count,
            "scan_duration_ms": report.scan_duration_ms,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "document_ids": c.document_ids,
                    "representative_id": c.representative_id,
                    "cluster_size": c.cluster_size,
                    "internal_similarity": c.internal_similarity,
                    "match_type": c.match_type.value,
                }
                for c in report.clusters
            ],
            "match_summary": [
                {
                    "source": m.source_id,
                    "target": m.target_id,
                    "match_type": m.match_type.value,
                    "overall_score": m.overall_score,
                    "details": m.details,
                }
                for m in report.matches[:100]  # Cap at 100 for readability
            ],
        }, indent=2, default=str)

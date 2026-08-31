"""
Tests for src.core.dedup_engine
--------------------------------
Covers text normalization, shingle generation, MinHash signature computation,
Jaccard estimation, LSH candidate generation, near-duplicate detection,
batch corpus deduplication, and aggregate statistics.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.core.dedup_engine import (
    DEFAULT_NUM_HASH_FUNCS,
    DedupReport,
    DuplicatePair,
    MinHashSignature,
    _normalize_text,
    _hash_shingle,
    _lsh_candidates,
    compute_dedup_stats,
    compute_minhash_signatures,
    deduplicate_corpus,
    find_near_duplicates,
    generate_shingles,
)


# ── Text normalization ────────────────────────────────────────────────────────


class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalize_text("it's a test!") == "it s a test"

    def test_collapses_whitespace(self):
        assert _normalize_text("  hello   world  ") == "hello world"

    def test_empty(self):
        assert _normalize_text("") == ""


# ── Shingling ─────────────────────────────────────────────────────────────────


class TestGenerateShingles:
    def test_basic(self):
        shingles = generate_shingles("abcdef", shingle_size=3)
        assert "abc" in shingles
        assert "bcd" in shingles
        assert "def" in shingles
        assert len(shingles) == 4

    def test_short_text(self):
        shingles = generate_shingles("ab", shingle_size=3)
        assert len(shingles) == 1  # Entire text as single shingle

    def test_empty(self):
        shingles = generate_shingles("", shingle_size=3)
        assert len(shingles) == 0

    def test_normalized(self):
        # Punctuation and case should be normalized
        s1 = generate_shingles("Hello, World!", shingle_size=3)
        s2 = generate_shingles("hello world", shingle_size=3)
        # "hello world" normalizes to "hello world" which has fewer shingles
        # than "hello  world" (two spaces) — but the normalized forms match
        assert len(s1) > 0
        assert len(s2) > 0

    def test_identical_texts(self):
        s1 = generate_shingles("machine learning is great", shingle_size=3)
        s2 = generate_shingles("machine learning is great", shingle_size=3)
        assert s1 == s2

    def test_frozenset(self):
        shingles = generate_shingles("test text here", shingle_size=3)
        assert isinstance(shingles, frozenset)


# ── Hash shingle ──────────────────────────────────────────────────────────────


class TestHashShingle:
    def test_deterministic(self):
        h1 = _hash_shingle("hello", 0)
        h2 = _hash_shingle("hello", 0)
        assert h1 == h2

    def test_different_seeds(self):
        h1 = _hash_shingle("hello", 0)
        h2 = _hash_shingle("hello", 1)
        assert h1 != h2

    def test_different_shingles(self):
        h1 = _hash_shingle("hello", 0)
        h2 = _hash_shingle("world", 0)
        assert h1 != h2

    def test_uint32_range(self):
        h = _hash_shingle("test", 0)
        assert 0 <= h <= np.iinfo(np.uint32).max


# ── MinHash signature ─────────────────────────────────────────────────────────


class TestMinHashSignature:
    def test_identical_shingles(self):
        shingles = frozenset({"abc", "bcd", "cde"})
        sig1 = MinHashSignature(shingles, num_hashes=64)
        sig2 = MinHashSignature(shingles, num_hashes=64)
        assert sig1.estimate_jaccard(sig2) == pytest.approx(1.0)

    def test_disjoint_shingles(self):
        sig1 = MinHashSignature(frozenset({"aaa", "bbb"}), num_hashes=64)
        sig2 = MinHashSignature(frozenset({"xxx", "yyy"}), num_hashes=64)
        jaccard = sig1.estimate_jaccard(sig2)
        assert jaccard < 0.5  # Should be low

    def test_empty_shingles(self):
        sig = MinHashSignature(frozenset(), num_hashes=64)
        assert sig.shingle_count == 0
        assert len(sig.signature) == 64

    def test_shingle_count(self):
        sig = MinHashSignature(frozenset({"a", "b", "c"}), num_hashes=32)
        assert sig.shingle_count == 3

    def test_mismatched_num_hashes(self):
        sig1 = MinHashSignature(frozenset({"a"}), num_hashes=32)
        sig2 = MinHashSignature(frozenset({"a"}), num_hashes=64)
        with pytest.raises(ValueError, match="same number"):
            sig1.estimate_jaccard(sig2)


# ── LSH candidates ────────────────────────────────────────────────────────────


class TestLSHCandidates:
    def test_identical_docs_generate_candidates(self):
        sigs = {
            "doc1": MinHashSignature(frozenset({"abc", "bcd"}), num_hashes=32),
            "doc2": MinHashSignature(frozenset({"abc", "bcd"}), num_hashes=32),
        }
        candidates = _lsh_candidates(sigs, num_bands=4, rows_per_band=8)
        assert ("doc1", "doc2") in candidates

    def test_empty_sigs(self):
        candidates = _lsh_candidates({}, num_bands=4, rows_per_band=8)
        assert len(candidates) == 0

    def test_single_doc(self):
        sigs = {"doc1": MinHashSignature(frozenset({"abc"}), num_hashes=32)}
        candidates = _lsh_candidates(sigs, num_bands=4, rows_per_band=8)
        assert len(candidates) == 0


# ── Near-duplicate detection ─────────────────────────────────────────────────


class TestFindNearDuplicates:
    def test_identical_documents(self):
        docs = {
            "doc1": "Machine learning is a branch of artificial intelligence.",
            "doc2": "Machine learning is a branch of artificial intelligence.",
        }
        report = find_near_duplicates(docs, threshold=0.3, num_hashes=64)
        assert report.duplicate_pairs_found >= 1
        assert report.duplicates[0].jaccard_estimate > 0.9

    def test_similar_documents(self):
        docs = {
            "doc1": "Machine learning is a powerful technique for data analysis and prediction.",
            "doc2": "Machine learning is a powerful method for data analysis and classification.",
        }
        report = find_near_duplicates(docs, threshold=0.3, num_hashes=64)
        assert report.duplicate_pairs_found >= 1

    def test_different_documents(self):
        docs = {
            "doc1": "The weather today is sunny and warm with clear blue skies.",
            "doc2": "Quantum computing leverages superposition for parallel computation.",
        }
        report = find_near_duplicates(docs, threshold=0.8, num_hashes=64)
        assert report.duplicate_pairs_found == 0

    def test_three_docs_two_duplicates(self):
        docs = {
            "doc1": "Deep learning uses neural networks for image recognition tasks.",
            "doc2": "Deep learning employs neural networks for image recognition tasks.",
            "doc3": "Cooking pasta requires boiling water for approximately ten minutes.",
        }
        report = find_near_duplicates(docs, threshold=0.3, num_hashes=64)
        # doc1 and doc2 should be duplicates, doc3 should not pair with either
        assert report.duplicate_pairs_found >= 1
        dup_names = {(d.doc_a, d.doc_b) for d in report.duplicates}
        # At least one pair involves doc1 and doc2
        assert any("doc1" in pair and "doc2" in pair for pair in dup_names)

    def test_empty_collection(self):
        report = find_near_duplicates({}, threshold=0.5)
        assert report.total_documents == 0
        assert report.duplicate_pairs_found == 0

    def test_single_document(self):
        report = find_near_duplicates({"doc1": "hello world"}, threshold=0.5)
        assert report.total_documents == 1
        assert report.duplicate_pairs_found == 0

    def test_report_structure(self):
        docs = {"a": "test text here for testing purposes only", "b": "test text here for testing purposes only"}
        report = find_near_duplicates(docs, threshold=0.1, num_hashes=32)
        assert isinstance(report, DedupReport)
        assert report.total_documents == 2
        d = report.to_dict()
        assert "total_documents" in d
        assert "duplicates" in d

    def test_lsh_disabled(self):
        docs = {
            f"doc{i}": f"Document number {i} about topic {i % 3} with similar content" * 5
            for i in range(10)
        }
        report = find_near_duplicates(docs, threshold=0.5, use_lsh=False, num_hashes=32)
        assert isinstance(report, DedupReport)

    def test_high_threshold_no_duplicates(self):
        docs = {
            "a": "cats are wonderful pets that love to play",
            "b": "dogs are loyal companions that enjoy walks",
        }
        report = find_near_duplicates(docs, threshold=0.95, num_hashes=64)
        assert report.duplicate_pairs_found == 0


# ── Compute minhash signatures ───────────────────────────────────────────────


class TestComputeMinhashSignatures:
    def test_returns_dict(self):
        docs = {"d1": "hello world", "d2": "goodbye world"}
        sigs = compute_minhash_signatures(docs)
        assert "d1" in sigs
        assert "d2" in sigs
        assert isinstance(sigs["d1"], MinHashSignature)

    def test_empty_docs(self):
        sigs = compute_minhash_signatures({})
        assert sigs == {}


# ── Batch corpus deduplication ────────────────────────────────────────────────


class TestDeduplicateCorpus:
    def test_basic(self):
        chunks = {
            "essay1.pdf": ["Machine learning is great for AI.", "It helps with prediction."],
            "essay2.pdf": ["Machine learning is great for AI.", "It assists with forecasting."],
            "essay3.pdf": ["Cooking involves many techniques and skills."],
        }
        report = deduplicate_corpus(chunks, threshold=0.3, num_hashes=64)
        assert report.total_documents == 3
        assert report.duplicate_pairs_found >= 1

    def test_empty_chunks(self):
        report = deduplicate_corpus({})
        assert report.total_documents == 0

    def test_single_document(self):
        report = deduplicate_corpus({"doc.pdf": ["just one doc"]})
        assert report.total_documents == 1
        assert report.duplicate_pairs_found == 0


# ── Dedup statistics ──────────────────────────────────────────────────────────


class TestComputeDedupStats:
    def test_with_duplicates(self):
        docs = {
            "doc1": "Machine learning is powerful for data analysis and prediction models.",
            "doc2": "Machine learning is powerful for data analysis and forecasting models.",
            "doc3": "Cooking pasta requires boiling water and salt for best results.",
        }
        report = find_near_duplicates(docs, threshold=0.3, num_hashes=64)
        stats = compute_dedup_stats(report)
        assert "total_documents" in stats
        assert "duplicate_rate" in stats
        assert "severity_breakdown" in stats

    def test_no_duplicates(self):
        report = DedupReport(
            total_documents=5, total_pairs_checked=10, duplicate_pairs_found=0,
            avg_similarity=0.0, max_similarity=0.0, duplicates=[],
        )
        stats = compute_dedup_stats(report)
        assert stats["duplicate_rate"] == 0.0
        assert stats["most_duplicated_doc"] is None

    def test_most_duplicated_doc(self):
        dups = [
            DuplicatePair("a", "b", 0.9, 50, 100, 100),
            DuplicatePair("a", "c", 0.85, 45, 100, 95),
            DuplicatePair("b", "c", 0.3, 20, 100, 95),
        ]
        report = DedupReport(5, 10, 3, 0.68, 0.9, dups)
        stats = compute_dedup_stats(report)
        assert stats["most_duplicated_doc"]["name"] == "a"
        assert stats["most_duplicated_doc"]["pair_count"] == 2


# ── DuplicatePair ─────────────────────────────────────────────────────────────


class TestDuplicatePair:
    def test_to_dict(self):
        dp = DuplicatePair("a.pdf", "b.pdf", 0.85, 50, 100, 100)
        d = dp.to_dict()
        assert d["doc_a"] == "a.pdf"
        assert d["jaccard_estimate"] == 0.85
        assert d["shared_shingles"] == 50

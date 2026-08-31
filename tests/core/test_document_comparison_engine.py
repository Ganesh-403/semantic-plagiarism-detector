"""
Tests for src.core.document_comparison_engine
----------------------------------------------
Covers tokenization, paragraph splitting, word overlap, paragraph matching,
coverage computation, severity classification, highlighted text generation,
and the full compare_documents pipeline.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.core.document_comparison_engine import (
    ComparisonResult,
    ParagraphMatch,
    WordOverlap,
    _compute_word_overlap,
    _cosine_similarity,
    _find_paragraph_matches,
    _highlight_common_words,
    _jaccard_similarity,
    _split_paragraphs,
    _tokenize,
    _classify_severity,
    _compute_coverage,
    compare_documents,
    generate_highlighted_paragraphs,
)


# ── Tokenization ──────────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert "123" in _tokenize("Value is 123 here")

    def test_punctuation_stripped(self):
        tokens = _tokenize("it's a test, really!")
        assert "it" in tokens
        assert "s" not in tokens


# ── Paragraph splitting ───────────────────────────────────────────────────────


class TestSplitParagraphs:
    def test_basic_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paras = _split_paragraphs(text, min_words=1)
        assert len(paras) == 3

    def test_min_words_filter(self):
        text = "Short.\n\nThis is a much longer paragraph with many words in it."
        paras = _split_paragraphs(text, min_words=5)
        assert len(paras) == 1

    def test_empty_text(self):
        assert _split_paragraphs("") == []

    def test_whitespace_only(self):
        assert _split_paragraphs("   \n\n   ") == []


# ── Similarity functions ──────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_empty_vectors(self):
        a = np.array([])
        b = np.array([])
        assert _cosine_similarity(a, b) == 0.0

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self):
        assert _jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity(["a", "b", "c"], ["b", "c", "d"])
        assert sim == pytest.approx(0.5)

    def test_empty(self):
        assert _jaccard_similarity([], []) == 0.0


# ── Word overlap ──────────────────────────────────────────────────────────────


class TestWordOverlap:
    def test_basic(self):
        wo = _compute_word_overlap("the cat sat", "the cat ran")
        assert wo.common_words == 2  # "the", "cat"
        assert wo.jaccard_similarity > 0.0

    def test_empty_texts(self):
        wo = _compute_word_overlap("", "")
        assert wo.common_words == 0

    def test_top_common_sorted(self):
        wo = _compute_word_overlap("test test test hello", "test test world")
        assert wo.top_common[0][0] == "test"


# ── Paragraph matching ────────────────────────────────────────────────────────


class TestParagraphMatching:
    def test_exact_match(self):
        src = ["Machine learning is a branch of AI."]
        tgt = ["Machine learning is a branch of AI."]
        matches = _find_paragraph_matches(src, tgt, threshold=0.1)
        assert len(matches) == 1
        assert matches[0].is_exact is True

    def test_similar_paragraphs(self):
        src = ["Deep learning uses neural networks for tasks."]
        tgt = ["Deep learning employs neural networks for various tasks."]
        matches = _find_paragraph_matches(src, tgt, threshold=0.1)
        assert len(matches) == 1
        assert matches[0].similarity > 0.3

    def test_unrelated_text(self):
        src = ["The weather is sunny today."]
        tgt = ["Quantum computing is advancing rapidly."]
        matches = _find_paragraph_matches(src, tgt, threshold=0.8)
        assert len(matches) == 0

    def test_threshold_filtering(self):
        src = ["Hello world"]
        tgt = ["Hello universe"]
        matches_strict = _find_paragraph_matches(src, tgt, threshold=0.9)
        assert len(matches_strict) == 0
        matches_loose = _find_paragraph_matches(src, tgt, threshold=0.1)
        assert len(matches_loose) >= 1

    def test_multiple_paragraphs(self):
        src = ["First paragraph about cats.", "Second paragraph about dogs."]
        tgt = ["Cats are wonderful animals.", "Dogs are loyal companions."]
        matches = _find_paragraph_matches(src, tgt, threshold=0.0)
        assert len(matches) == 2

    def test_sorted_by_similarity(self):
        src = ["A simple sentence.", "A very long and detailed technical paragraph about machine learning algorithms."]
        tgt = ["A different sentence.", "A very long and detailed technical paragraph about machine learning models."]
        matches = _find_paragraph_matches(src, tgt, threshold=0.0)
        if len(matches) >= 2:
            assert matches[0].similarity >= matches[1].similarity


# ── Coverage ──────────────────────────────────────────────────────────────────


class TestCoverage:
    def test_full_coverage(self):
        paras = ["a", "b", "c"]
        matches = [ParagraphMatch(0, 0, "", "", 0.9, False),
                   ParagraphMatch(1, 1, "", "", 0.8, False),
                   ParagraphMatch(2, 2, "", "", 0.7, False)]
        assert _compute_coverage(paras, matches, is_source=True) == 1.0

    def test_partial_coverage(self):
        paras = ["a", "b", "c", "d"]
        matches = [ParagraphMatch(0, 0, "", "", 0.9, False)]
        assert _compute_coverage(paras, matches, is_source=True) == 0.25

    def test_empty_paragraphs(self):
        assert _compute_coverage([], [], is_source=True) == 0.0


# ── Severity ──────────────────────────────────────────────────────────────────


class TestSeverity:
    def test_high(self):
        assert _classify_severity(0.95) == "High"

    def test_medium(self):
        assert _classify_severity(0.80) == "Medium"

    def test_low(self):
        assert _classify_severity(0.60) == "Low"

    def test_none(self):
        assert _classify_severity(0.30) == "None"


# ── Highlighting ──────────────────────────────────────────────────────────────


class TestHighlighting:
    def test_marks_common_words(self):
        src, tgt = _highlight_common_words("the cat sat", "the dog ran")
        assert "<mark>the</mark>" in src
        assert "<mark>the</mark>" in tgt

    def test_no_common_words(self):
        src, tgt = _highlight_common_words("xyz", "abc")
        assert "<mark>" not in src

    def test_exact_text_unchanged_besides_marks(self):
        src, _ = _highlight_common_words("hello world", "hello universe")
        assert "hello" in src


# ── Full pipeline ─────────────────────────────────────────────────────────────


class TestCompareDocuments:
    def test_identical_documents(self):
        text = "Machine learning is transforming industries. Deep learning powers modern AI."
        result = compare_documents(text, text, "a.pdf", "b.pdf")
        assert result.document_similarity == pytest.approx(1.0)
        assert result.severity == "High"
        assert result.matched_paragraph_count >= 1

    def test_different_documents(self):
        src = "The quick brown fox jumps over the lazy dog in the afternoon."
        tgt = "Quantum computing leverages superposition and entanglement for computation."
        result = compare_documents(src, tgt, "src.pdf", "tgt.pdf")
        assert result.document_similarity < 0.5
        assert result.severity in ("None", "Low")

    def test_partial_overlap(self):
        src = "Machine learning is a powerful tool for data analysis and pattern recognition."
        tgt = "Machine learning is a useful technique for data analysis and classification."
        result = compare_documents(src, tgt)
        assert 0.3 < result.document_similarity < 1.0
        assert result.matched_paragraph_count >= 1

    def test_to_dict(self):
        result = compare_documents("hello world", "hello world")
        d = result.to_dict()
        assert "document_similarity" in d
        assert "paragraph_matches" in d
        assert "word_overlap" in d

    def test_empty_documents(self):
        result = compare_documents("", "")
        assert result.document_similarity == 0.0
        assert result.matched_paragraph_count == 0

    def test_with_embeddings(self):
        src_emb = np.random.rand(3, 384).astype(np.float32)
        tgt_emb = np.random.rand(3, 384).astype(np.float32)
        result = compare_documents(
            "Paragraph one text here. Paragraph two text here. Paragraph three text here.",
            "Paragraph one text here. Paragraph two text here. Paragraph three text here.",
            source_embeddings=src_emb,
            target_embeddings=tgt_emb,
        )
        assert 0.0 <= result.document_similarity <= 1.0

    def test_highlighted_paragraphs(self):
        result = compare_documents(
            "The quick brown fox jumps.",
            "The quick brown fox leaps.",
        )
        highlighted = generate_highlighted_paragraphs(result, top_n=3)
        assert isinstance(highlighted, list)
        if highlighted:
            assert "source_highlighted" in highlighted[0]


# ── ComparisonResult dataclass ────────────────────────────────────────────────


class TestComparisonResultDataclass:
    def test_frozen_fields(self):
        r = ComparisonResult(
            source_filename="a", target_filename="b",
            source_paragraphs=[], target_paragraphs=[],
            paragraph_matches=[], document_similarity=0.5,
            max_paragraph_similarity=0.5, avg_paragraph_similarity=0.5,
            source_coverage=0.0, target_coverage=0.0,
            word_overlap=WordOverlap(0, 0, 0, 0.0, []),
            matched_paragraph_count=0, total_paragraphs=0,
            severity="Low",
        )
        assert r.source_filename == "a"
        d = r.to_dict()
        assert d["document_similarity"] == 0.5

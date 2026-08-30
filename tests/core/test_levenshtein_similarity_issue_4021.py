"""Tests for levenshtein_similarity (Issue #4021)."""

from src.core.lexical_similarity import levenshtein_similarity


def test_levenshtein_identical_strings():
    assert levenshtein_similarity("Alice Smith", "Alice Smith") == 1.0
    assert levenshtein_similarity("", "") == 1.0


def test_levenshtein_completely_distinct():
    assert levenshtein_similarity("abc", "xyz") == 0.0
    assert levenshtein_similarity("a", "b") == 0.0
    assert levenshtein_similarity("", "title") == 0.0


def test_levenshtein_partial_similarity():
    score = levenshtein_similarity("kitten", "sitting")
    assert 0.0 < score < 1.0
    # distance=3, max_len=7 → 1 - 3/7
    assert abs(score - (1.0 - 3 / 7)) < 1e-9

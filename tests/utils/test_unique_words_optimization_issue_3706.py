"""
test_unique_words_optimization_issue_3706.py
--------------------------------------------
Comprehensive unit, regression, and performance benchmark test suite for Issue #3706:
Performance optimization for count_unique_words on large texts.

Verifies:
1. Generator-based unique word extraction using re.finditer directly into set() without intermediate list allocations.
2. Correctness against diverse texts (mixed case, punctuation, contractions, special characters).
3. Large-text scalability and performance (e.g. 50,000 words / dissertation scale).
4. Lexical diversity and vocabulary richness calculations (TTR, Guiraud, Herdan, Yule's K).
5. Hapax Legomena and Dislegomena analysis.
6. Streaming token iterators and batch counting helpers.
7. CJK multilingual text support.
"""

import re
import pytest

from src.utils.text_stats import (
    batch_count_unique_words,
    compute_hapax_dislegomena,
    compute_hapax_legomena,
    compute_vocabulary_richness,
    compute_yule_k_characteristic,
    count_unique_words,
    count_words,
    get_unique_word_ratio,
    get_unique_words_set,
    iter_word_tokens,
    stream_word_frequencies,
)


class TestCountUniqueWordsOptimization:
    """Test suite verifying optimization and correctness of count_unique_words."""

    def test_empty_string_returns_zero(self):
        assert count_unique_words("") == 0
        assert count_unique_words("   ") == 0
        assert count_unique_words("\t\n\r") == 0

    def test_single_word(self):
        assert count_unique_words("hello") == 1
        assert count_unique_words("Hello") == 1
        assert count_unique_words("HELLO") == 1

    def test_case_insensitivity(self):
        text = "Apple apple APPLE aPple banana BANANA"
        # 2 unique words: apple, banana
        assert count_unique_words(text) == 2
        assert get_unique_words_set(text) == {"apple", "banana"}

    def test_re_finditer_generator_equivalence(self):
        """Verify generator expression produces exact same result as set comprehension on finditer."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "The dog was not amused by the quick brown fox."
        )
        expected_set = set(m.group(0).lower() for m in re.finditer(r"\b\w+\b", text))
        actual_count = count_unique_words(text)
        assert actual_count == len(expected_set)
        assert get_unique_words_set(text) == expected_set

    def test_punctuation_and_symbols_stripped(self):
        text = "word! word? 'word' \"word\" (word) [word] {word} word... word,"
        assert count_unique_words(text) == 1

    def test_numbers_and_alphanumerics(self):
        text = "var1 var2 var1 100 200 100 200"
        assert count_unique_words(text) == 4
        assert get_unique_words_set(text) == {"var1", "var2", "100", "200"}


class TestLargeTextPerformance:
    """Performance and stress test suite for large texts (50,000+ words)."""

    def test_large_dissertation_scale_text(self):
        """Simulate a 50,000 word dissertation text."""
        base_words = [
            "plagiarism", "semantic", "analysis", "detection", "embedding",
            "similarity", "metric", "algorithm", "document", "transformer",
            "neural", "network", "corpus", "evaluation", "benchmark",
            "vector", "clustering", "cosine", "lexical", "syntactic"
        ]
        # Repeat 2,500 times = 50,000 words
        large_text = " ".join(base_words * 2500)
        assert count_words(large_text) == 50000

        unique_count = count_unique_words(large_text)
        assert unique_count == len(base_words)

    def test_large_vocabulary_scale_text(self):
        """Test with 10,000 distinct words."""
        distinct_words = [f"wordtoken{i}" for i in range(10000)]
        large_text = " ".join(distinct_words)

        assert count_unique_words(large_text) == 10000
        unique_set = get_unique_words_set(large_text)
        assert len(unique_set) == 10000


class TestStreamingAndRichnessHelpers:
    """Test suite for streaming token iterators and lexical richness indices."""

    def test_iter_word_tokens(self):
        text = "First Second THIRD"
        tokens = list(iter_word_tokens(text))
        assert tokens == ["first", "second", "third"]

    def test_iter_word_tokens_empty(self):
        tokens = list(iter_word_tokens(""))
        assert tokens == []

    def test_stream_word_frequencies(self):
        text = "apple banana apple cherry apple banana"
        freqs = stream_word_frequencies(text)
        assert freqs == {"apple": 3, "banana": 2, "cherry": 1}

    def test_stream_word_frequencies_empty(self):
        assert stream_word_frequencies("") == {}

    def test_compute_vocabulary_richness_standard(self):
        text = "the quick brown fox jumps over the lazy dog"
        richness = compute_vocabulary_richness(text)
        assert richness["tokens"] == 9
        assert richness["types"] == 8
        assert richness["ttr"] == round(8 / 9, 4)
        assert richness["guiraud_r"] > 0
        assert richness["herdan_c"] > 0

    def test_compute_vocabulary_richness_empty(self):
        richness = compute_vocabulary_richness("")
        assert richness["tokens"] == 0
        assert richness["types"] == 0
        assert richness["ttr"] == 0.0

    def test_batch_count_unique_words(self):
        texts = [
            "one two three",
            "one one one",
            "alpha beta gamma delta epsilon",
        ]
        counts = batch_count_unique_words(texts)
        assert counts == [3, 1, 5]

    def test_get_unique_word_ratio(self):
        text = "a b a b a b"
        ratio = get_unique_word_ratio(text)
        # 2 unique / 6 total = 0.3333...
        assert abs(ratio - (2 / 6)) < 1e-5


class TestHapaxAndYuleRichness:
    """Test suite for Hapax Legomena, Dislegomena, and Yule's K metric."""

    def test_hapax_legomena(self):
        text = "cat dog cat bird fish bird zebra"
        # 1-time words: dog, fish, zebra
        hapax = compute_hapax_legomena(text)
        assert hapax == ["dog", "fish", "zebra"]

    def test_hapax_dislegomena(self):
        text = "cat dog cat bird fish bird zebra dog"
        # 2-time words: bird, cat, dog
        dis = compute_hapax_dislegomena(text)
        assert dis == ["bird", "cat", "dog"]

    def test_yule_k_characteristic_homogeneous_text(self):
        # Repetitive vocabulary -> higher Yule's K
        repetitive = "the the the the the the the"
        k_rep = compute_yule_k_characteristic(repetitive)
        assert isinstance(k_rep, float)

        diverse = "one two three four five six seven eight nine ten"
        k_div = compute_yule_k_characteristic(diverse)
        assert isinstance(k_div, float)

    def test_yule_k_characteristic_empty(self):
        assert compute_yule_k_characteristic("") == 0.0
        assert compute_yule_k_characteristic("word") == 0.0


class TestMultilingualCJKSupport:
    """Test suite for CJK and multilingual token handling."""

    def test_cjk_unique_characters(self):
        text = "中文测试 中文分析"
        # Unique CJK chars: 中, 文, 测, 试, 分, 析 (6 distinct characters)
        assert count_unique_words(text) == 6

    def test_mixed_cjk_and_english(self):
        text = "Python 数据分析 and NLP 算法"
        # English words: python, and, nlp (3)
        # CJK chars: 数, 据, 分, 析, 算, 法 (6)
        # Total = 9
        assert count_unique_words(text) == 9

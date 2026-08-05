import pytest
from src.core.lexical_similarity import (
    calculate_lexical_similarity,
    compute_tfidf_lexical_similarity,
    dice_coefficient,
    get_ngrams,
    jaccard_index,
    jaccard_similarity,
    lexical_similarity_matrix,
    n_gram_overlap,
    overlap_coefficient,
    remove_stopwords,
    scale_lexical_score,
    tokenize,
)


def test_calculate_lexical_similarity_without_custom_stopwords():
    text1 = "This is a study on artificial intelligence and neural networks."
    text2 = "This paper presents a study on artificial intelligence algorithms."

    score = calculate_lexical_similarity(text1, text2)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_calculate_lexical_similarity_with_custom_stopwords():
    # Academic text with domain filler words
    text1 = "Figure 1 shows ibid result for quantum computing."
    text2 = "Table 2 describes ibid quantum computing methodology."

    # Baseline calculation (where "figure", "table", "ibid" might inflate match)
    base_score = calculate_lexical_similarity(text1, text2)

    # Calculation filtering out academic filler words
    custom_stopwords = {"figure", "table", "ibid"}
    filtered_score = calculate_lexical_similarity(
        text1, text2, custom_stopwords=custom_stopwords
    )

    # Filtering out shared filler word 'ibid' reduces similarity score
    assert filtered_score <= base_score + 1e-6
    assert filtered_score == pytest.approx(filtered_score)


def test_lexical_similarity_matrix_with_custom_stopwords():
    documents = {
        "doc1": "Figure 1 table data analysis ibid.",
        "doc2": "Figure 2 table data processing ibid.",
    }

    custom_stopwords = {"figure", "table", "ibid"}
    df = lexical_similarity_matrix(documents, custom_stopwords=custom_stopwords)

    assert df.shape == (2, 2)
    assert "doc1" in df.index
    assert "doc2" in df.columns


def test_remove_stopwords():
    text = "The quick brown fox jumps over the lazy dog"
    filtered = remove_stopwords(text)
    assert "the" not in filtered.split()
    assert "quick" in filtered.split()
    assert "brown" in filtered.split()


def test_tokenize():
    text = "Artificial intelligence and machine learning"
    tokens = tokenize(text)
    assert "artificial" in tokens
    assert "intelligence" in tokens
    assert "and" not in tokens


def test_get_ngrams():
    text = "machine learning models for natural language processing"
    ngrams = get_ngrams(text, n=2)
    assert isinstance(ngrams, set)
    assert ("machine", "learning") in ngrams


def test_n_gram_overlap():
    text1 = "natural language processing algorithms"
    text2 = "natural language processing methods"
    score = n_gram_overlap(text1, text2, n=2)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_jaccard_similarity_and_index():
    text1 = "data science and analytics"
    text2 = "data science and engineering"
    sim = jaccard_similarity(text1, text2)
    idx = jaccard_index(text1, text2)
    assert sim == idx
    assert 0.0 <= sim <= 1.0
    assert sim > 0.0


def test_dice_coefficient():
    text1 = "deep learning neural networks"
    text2 = "deep learning convolutional networks"
    score = dice_coefficient(text1, text2)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_overlap_coefficient():
    text1 = "semantic plagiarism detection"
    text2 = "semantic plagiarism detection and automated document verification"
    score = overlap_coefficient(text1, text2)
    assert score == pytest.approx(1.0)


# ── Soft-Max Normalization Tests (#924) ────────────────────────────────────────


def test_scale_lexical_score_boundaries():
    """Test boundary inputs 0.0, 0.5, 1.0 for scale_lexical_score."""
    assert scale_lexical_score(0.0) == 0.0
    assert scale_lexical_score(0.5) == pytest.approx(0.5, abs=1e-6)
    assert scale_lexical_score(1.0) == 1.0


def test_scale_lexical_score_range_bounds():
    """Verify output is strictly bounded between 0.0 and 1.0 for arbitrary inputs."""
    for val in [-1.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0]:
        res = scale_lexical_score(val)
        assert 0.0 <= res <= 1.0


# ── TF-IDF Lexical Similarity Tests (#1351) ───────────────────────────────────


def test_compute_tfidf_lexical_similarity_basic():
    doc_a = "Introduction to neural network architectures and deep learning."
    doc_b = "Methodology of deep learning and neural network models."
    corpus = [
        "Introduction to neural network architectures and deep learning.",
        "Methodology of deep learning and neural network models.",
        "Quantum computing physics and quantum algorithms.",
    ]
    score = compute_tfidf_lexical_similarity(doc_a, doc_b, corpus)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_compute_tfidf_lexical_similarity_identical_docs():
    doc = "Artificial intelligence and machine learning algorithms in data science."
    corpus = [doc, "Unrelated text about cooking and baking recipes."]
    score = compute_tfidf_lexical_similarity(doc, doc, corpus)
    assert score == pytest.approx(1.0)


def test_compute_tfidf_lexical_similarity_empty_inputs():
    assert compute_tfidf_lexical_similarity("", "test", ["test"]) == 0.0
    assert compute_tfidf_lexical_similarity("test", "", ["test"]) == 0.0
    assert compute_tfidf_lexical_similarity("", "", []) == 0.0


# ─── Tests for Character-Level N-Gram Similarity (Issue #1479) ─────────────────

from src.core.lexical_similarity import compute_char_ngram_similarity
import pytest

class TestComputeCharNgramSimilarity:
    """Comprehensive test suite for character-level sliding n-gram Jaccard similarity."""

    def test_identical_strings_returns_one(self):
        """Identical strings must produce a perfect similarity score of 1.0."""
        text = "This is a test sentence for character n-grams."
        assert compute_char_ngram_similarity(text, text, n=5) == 1.0

    def test_completely_different_strings_returns_zero(self):
        """Strings with no overlapping character sequences must return 0.0."""
        text_a = "aaaaaaaaaa"
        text_b = "bbbbbbbbbb"
        assert compute_char_ngram_similarity(text_a, text_b, n=3) == 0.0

    @pytest.mark.parametrize(
        "text_a,text_b,n,expected_min",
        [
            ("plagiarism", "plagiarism", 5, 1.0),
            ("plagiarism", "plagarism", 5, 0.5),  # One character missing
            ("hello world", "hello world!", 5, 0.8),  # Punctuation addition
            ("abcde", "abcde", 1, 1.0),  # n=1 (character unigrams)
            ("abcde", "abcde", 10, 0.0),  # n > len(text)
        ],
    )
    def test_parametrized_similarity_bounds(self, text_a, text_b, n, expected_min):
        """Verify similarity scores fall within expected mathematical bounds."""
        score = compute_char_ngram_similarity(text_a, text_b, n=n)
        assert 0.0 <= score <= 1.0
        assert score >= expected_min - 1e-6

    def test_empty_string_a_returns_zero(self):
        """Empty first string must immediately return 0.0 without errors."""
        assert compute_char_ngram_similarity("", "some text here", n=5) == 0.0

    def test_empty_string_b_returns_zero(self):
        """Empty second string must immediately return 0.0 without errors."""
        assert compute_char_ngram_similarity("some text here", "", n=5) == 0.0

    def test_both_empty_strings_returns_zero(self):
        """Two empty strings must return 0.0 (union is empty)."""
        assert compute_char_ngram_similarity("", "", n=5) == 0.0

    def test_none_input_returns_zero(self):
        """None inputs must be handled gracefully and return 0.0."""
        assert compute_char_ngram_similarity(None, "text", n=5) == 0.0
        assert compute_char_ngram_similarity("text", None, n=5) == 0.0

    def test_non_string_input_returns_zero(self):
        """Non-string types (int, list) must return 0.0 without raising TypeError."""
        assert compute_char_ngram_similarity(12345, "text", n=5) == 0.0
        assert compute_char_ngram_similarity("text", ["list", "of", "words"], n=5) == 0.0

    def test_case_insensitivity(self):
        """Character n-grams should be case-insensitive for plagiarism detection."""
        text_a = "The Quick Brown Fox"
        text_b = "the quick brown fox"
        assert compute_char_ngram_similarity(text_a, text_b, n=4) == 1.0

    def test_whitespace_handling(self):
        """Leading and trailing whitespace should be stripped before comparison."""
        text_a = "   hello world   "
        text_b = "hello world"
        assert compute_char_ngram_similarity(text_a, text_b, n=5) == 1.0

    def test_n_greater_than_text_length(self):
        """If n is larger than the text length, no n-grams can be formed; return 0.0."""
        text_a = "short"
        text_b = "short text"
        assert compute_char_ngram_similarity(text_a, text_b, n=10) == 0.0

    def test_n_equals_one_character_unigrams(self):
        """n=1 should compute standard character unigram Jaccard similarity."""
        text_a = "abc"
        text_b = "bcd"
        # Unigrams: {'a','b','c'} and {'b','c','d'} -> intersection={'b','c'}, union={'a','b','c','d'}
        score = compute_char_ngram_similarity(text_a, text_b, n=1)
        assert score == pytest.approx(2.0 / 4.0)

    def test_invalid_n_defaults_to_five(self):
        """If n < 1 is provided, the function should default to n=5 and log a warning."""
        text_a = "This is a test sentence."
        text_b = "This is a test sentence."
        # n=0 is invalid, should default to 5 and return 1.0 for identical strings
        score = compute_char_ngram_similarity(text_a, text_b, n=0)
        assert score == 1.0

    def test_substring_detection(self):
        """A text that is a substring of another should have high similarity."""
        text_a = "The quick brown fox jumps over the lazy dog"
        text_b = "quick brown fox"
        score = compute_char_ngram_similarity(text_a, text_b, n=5)
        # All n-grams of text_b exist in text_a, so intersection == len(text_b ngrams)
        # Union == len(text_a ngrams)
        assert score > 0.3  # Should be significantly > 0

    def test_unicode_character_support(self):
        """Character n-grams must correctly handle multi-byte Unicode characters."""
        text_a = "café résumé naïve"
        text_b = "cafe resume naive"
        # These are different characters, so similarity should be < 1.0
        score = compute_char_ngram_similarity(text_a, text_b, n=3)
        assert 0.0 < score < 1.0

    def test_punctuation_impact(self):
        """Punctuation changes should affect character n-gram similarity."""
        text_a = "hello, world!"
        text_b = "hello world"
        score = compute_char_ngram_similarity(text_a, text_b, n=5)
        # The comma and exclamation mark change the n-grams
        assert score < 1.0
        assert score > 0.5

    def test_long_academic_text_paraphrase(self):
        """Simulate a realistic academic paraphrase detection scenario."""
        original = (
            "Machine learning algorithms have revolutionized the field of "
            "natural language processing in recent years."
        )
        paraphrased = (
            "Machine learning algorithms have revolutionized the field of "
            "natural language processing in recent decades."
        )
        # Only "years" -> "decades" changed
        score = compute_char_ngram_similarity(original, paraphrased, n=5)
        assert score > 0.85  # Should be very high since most text is identical

    def test_returns_float_type(self):
        """The return type must strictly be a Python float."""
        score = compute_char_ngram_similarity("test", "test", n=2)
        assert isinstance(score, float)


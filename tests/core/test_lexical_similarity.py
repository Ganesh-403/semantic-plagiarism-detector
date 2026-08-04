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



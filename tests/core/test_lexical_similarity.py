import pytest
from src.core.lexical_similarity import (
    calculate_lexical_similarity,
    lexical_similarity_matrix,
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
    
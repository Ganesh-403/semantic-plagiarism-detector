import numpy as np
import pandas as pd
import pytest

from src.core.lexical_similarity import (STOPWORDS,  # noqa: E402
                                         jaccard_similarity,
                                         lexical_similarity_matrix,
                                         remove_stopwords, tokenize)
from src.core.similarity import (calculate_paragraph_similarity_breakdown,
                                 chunk_max_similarity, chunk_similarity_matrix,
                                 document_similarity_matrix,
                                 find_most_similar_chunks, flag_plagiarism,
                                 hybrid_similarity_matrix)


def test_chunk_max_similarity(dummy_embeddings):
    emb_a = dummy_embeddings["doc_A"]
    emb_b = dummy_embeddings["doc_B"]
    emb_c = dummy_embeddings["doc_C"]

    # Similarity should be high between A and B
    sim_ab = chunk_max_similarity(emb_a, emb_b)
    assert sim_ab > 0.8

    # Similarity should be low between A and C
    sim_ac = chunk_max_similarity(emb_a, emb_c)
    assert sim_ac < 0.1

    # Empty embedding handling
    assert chunk_max_similarity(emb_a, np.array([])) == 0.0


def test_chunk_max_similarity_supports_batching(dummy_embeddings):
    sim_unbatched = chunk_max_similarity(
        dummy_embeddings["doc_A"], dummy_embeddings["doc_B"]
    )
    sim_batched = chunk_max_similarity(
        dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=1
    )
    assert np.isclose(sim_batched, sim_unbatched)


def test_chunk_max_similarity_rejects_invalid_batch_size(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_max_similarity(
            dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=0.5
        )


def test_document_similarity_matrix(dummy_embeddings):
    df = document_similarity_matrix(dummy_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["doc_A", "doc_B", "doc_C"]


def test_document_similarity_matrix_accepts_batch_size_basic(dummy_embeddings):
    df = document_similarity_matrix(dummy_embeddings, batch_size=2)
    assert isinstance(df, pd.DataFrame)

    # Diagonal should be ~1.0
    assert np.isclose(df.loc["doc_A", "doc_A"], 1.0)

    # A and B should be more similar to each other than A and C
    assert df.loc["doc_A", "doc_B"] > df.loc["doc_A", "doc_C"]


def test_document_similarity_matrix_accepts_batch_size(dummy_embeddings):
    unbatched = document_similarity_matrix(dummy_embeddings)
    batched = document_similarity_matrix(dummy_embeddings, batch_size=2)
    assert isinstance(batched, pd.DataFrame)
    assert np.allclose(unbatched.values, batched.values)


def test_document_similarity_matrix_rejects_invalid_batch_size(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        document_similarity_matrix(dummy_embeddings, batch_size=0.5)


def test_chunk_similarity_matrix(dummy_embeddings):
    df = chunk_similarity_matrix(dummy_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)


def test_chunk_similarity_matrix_accepts_batch_size_basic(dummy_embeddings):
    df = chunk_similarity_matrix(dummy_embeddings, batch_size=1)
    assert isinstance(df, pd.DataFrame)
    assert df.loc["doc_A", "doc_A"] == 1.0

    # Symmetric
    assert df.loc["doc_A", "doc_B"] == df.loc["doc_B", "doc_A"]


def test_chunk_similarity_matrix_accepts_batch_size(dummy_embeddings):
    unbatched = chunk_similarity_matrix(dummy_embeddings)
    batched = chunk_similarity_matrix(dummy_embeddings, batch_size=1)
    assert isinstance(batched, pd.DataFrame)
    assert np.allclose(unbatched.values, batched.values)


def test_batch_size_rejects_non_integer(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        document_similarity_matrix(dummy_embeddings, batch_size=0.5)
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_max_similarity(
            dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=0.5
        )
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_similarity_matrix(dummy_embeddings, batch_size=0.5)


def test_document_similarity_matrix_1d_embedding():
    emb_1d = np.array([1.0, 0.0, 0.0])
    df = document_similarity_matrix({"doc_1d": emb_1d})
    assert np.isclose(df.loc["doc_1d", "doc_1d"], 1.0)


def test_document_similarity_matrix_empty_embedding():
    df = document_similarity_matrix({"empty": np.array([])})
    assert df.shape == (1, 1)


def test_find_most_similar_chunks_returns_top_pairs():
    emb_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    emb_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    chunks_a = ["chunk a0", "chunk a1"]
    chunks_b = ["chunk b0", "chunk b1"]
    pairs = find_most_similar_chunks(
        chunks_a, chunks_b, emb_a, emb_b, top_k=2, threshold=0.5
    )
    assert len(pairs) >= 1
    assert pairs[0][0] == "chunk a0"
    assert pairs[0][1] == "chunk b0"
    assert pairs[0][2] > 0.5


def test_find_most_similar_chunks_empty_embeddings():
    result = find_most_similar_chunks([], [], np.array([]), np.array([]), top_k=3)
    assert result == []


def test_find_most_similar_chunks_threshold_filters():
    emb_a = np.array([[1.0, 0.0, 0.0]])
    emb_b = np.array([[0.0, 1.0, 0.0]])  # orthogonal → similarity 0.0
    pairs = find_most_similar_chunks(["a"], ["b"], emb_a, emb_b, top_k=3, threshold=0.5)
    assert pairs == []


def test_flag_plagiarism():
    data = [[1.0, 0.95, 0.60], [0.95, 1.0, 0.80], [0.60, 0.80, 1.0]]
    df = pd.DataFrame(data, index=["D1", "D2", "D3"], columns=["D1", "D2", "D3"])

    flags = flag_plagiarism(df, threshold=0.75)

    assert len(flags) == 2

    d1_d2 = next(f for f in flags if f["doc_a"] == "D1" and f["doc_b"] == "D2")
    assert d1_d2["similarity"] == 0.95
    assert "High" in d1_d2["severity"]

    d2_d3 = next(f for f in flags if f["doc_a"] == "D2" and f["doc_b"] == "D3")
    assert d2_d3["similarity"] == 0.80
    assert "Medium" in d2_d3["severity"]


def test_lexical_similarity_matrix_identical_documents():
    documents = {
        "doc1": "This is a test document with some text.",
        "doc2": "This is a test document with some text.",
        "doc3": "This is completely different content.",
    }
    df = lexical_similarity_matrix(documents)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["doc1", "doc2", "doc3"]

    # Identical documents should have similarity 1.0
    assert np.isclose(df.loc["doc1", "doc2"], 1.0)
    assert np.isclose(df.loc["doc1", "doc1"], 1.0)

    # Different documents should have lower similarity
    assert df.loc["doc1", "doc3"] < 0.9


def test_lexical_similarity_matrix_empty_documents():
    df = lexical_similarity_matrix({})
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (0, 0)


def test_lexical_similarity_matrix_single_document():
    documents = {"doc1": "Single document content."}
    df = lexical_similarity_matrix(documents)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 1)
    assert np.isclose(df.loc["doc1", "doc1"], 1.0)


def test_lexical_similarity_matrix_caching():
    """Test that caching works correctly for identical document sets."""
    documents = {
        "doc1": "This is a test document with some text.",
        "doc2": "This is a test document with some text.",
        "doc3": "This is completely different content.",
    }

    # Clear cache before test
    from src.core.lexical_similarity import _cached_lexical_similarity_matrix

    _cached_lexical_similarity_matrix.cache_clear()

    # First call - should compute
    df1 = lexical_similarity_matrix(documents, use_cache=True)

    # Second call with same documents - should use cache
    df2 = lexical_similarity_matrix(documents, use_cache=True)

    # Results should be identical
    assert np.allclose(df1.values, df2.values)

    # Cache should have been used (cache_info should show hits)
    cache_info = _cached_lexical_similarity_matrix.cache_info()
    assert cache_info.hits > 0


def test_lexical_similarity_matrix_cache_bypass():
    """Test that use_cache=False bypasses the cache."""
    documents = {
        "doc1": "This is a test document with some text.",
        "doc2": "This is a test document with some text.",
    }

    # Clear cache before test
    from src.core.lexical_similarity import _cached_lexical_similarity_matrix

    _cached_lexical_similarity_matrix.cache_clear()

    # Call with cache enabled
    df_cached = lexical_similarity_matrix(documents, use_cache=True)

    # Call with cache disabled - should bypass cache
    df_uncached = lexical_similarity_matrix(documents, use_cache=False)

    # Results should still be identical
    assert np.allclose(df_cached.values, df_uncached.values)


def test_lexical_similarity_matrix_different_documents():
    """Test that different document sets are cached separately."""
    documents1 = {
        "doc1": "This is about machine learning and artificial intelligence.",
        "doc2": "This is about deep learning and neural networks.",
    }

    documents2 = {
        "doc1": "This is about cooking recipes and baking techniques.",
        "doc2": "This is about grilling and barbecue methods.",
    }

    # Clear cache before test
    from src.core.lexical_similarity import _cached_lexical_similarity_matrix

    _cached_lexical_similarity_matrix.cache_clear()

    _ = lexical_similarity_matrix(documents1, use_cache=True)
    _ = lexical_similarity_matrix(documents2, use_cache=True)

    # Cache should have 2 entries (both document sets computed)
    cache_info = _cached_lexical_similarity_matrix.cache_info()
    assert cache_info.misses == 2  # Both were cache misses (computed)


def test_hybrid_similarity_matrix_boundary_conditions():
    semantic_df = pd.DataFrame(
        {"doc1": [1.0, 0.8, 0.3], "doc2": [0.8, 1.0, 0.4], "doc3": [0.3, 0.4, 1.0]},
        index=["doc1", "doc2", "doc3"],
    )

    lexical_df = pd.DataFrame(
        {"doc1": [1.0, 0.6, 0.2], "doc2": [0.6, 1.0, 0.3], "doc3": [0.2, 0.3, 1.0]},
        index=["doc1", "doc2", "doc3"],
    )

    # w=1.0 should return pure semantic
    hybrid_pure_semantic = hybrid_similarity_matrix(semantic_df, lexical_df, w=1.0)
    assert np.allclose(hybrid_pure_semantic.values, semantic_df.values)

    # w=0.0 should return pure lexical
    hybrid_pure_lexical = hybrid_similarity_matrix(semantic_df, lexical_df, w=0.0)
    assert np.allclose(hybrid_pure_lexical.values, lexical_df.values)

    # w=0.5 should be average
    hybrid_avg = hybrid_similarity_matrix(semantic_df, lexical_df, w=0.5)
    expected = (semantic_df + lexical_df) / 2
    assert np.allclose(hybrid_avg.values, expected.values)


def test_hybrid_similarity_matrix_default_weight():
    semantic_df = pd.DataFrame(
        {"doc1": [1.0, 0.8], "doc2": [0.8, 1.0]}, index=["doc1", "doc2"]
    )

    lexical_df = pd.DataFrame(
        {"doc1": [1.0, 0.6], "doc2": [0.6, 1.0]}, index=["doc1", "doc2"]
    )

    # Default weight should be 0.7
    hybrid_df = hybrid_similarity_matrix(semantic_df, lexical_df)
    expected = 0.7 * semantic_df + 0.3 * lexical_df
    assert np.allclose(hybrid_df.values, expected.values)


def test_hybrid_similarity_matrix_invalid_weight():
    semantic_df = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
    lexical_df = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]])

    with pytest.raises(ValueError, match="Weight w must be between 0.0 and 1.0"):
        hybrid_similarity_matrix(semantic_df, lexical_df, w=1.5)

    with pytest.raises(ValueError, match="Weight w must be between 0.0 and 1.0"):
        hybrid_similarity_matrix(semantic_df, lexical_df, w=-0.1)


def test_hybrid_similarity_matrix_shape_mismatch():
    semantic_df = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
    lexical_df = pd.DataFrame([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])

    with pytest.raises(
        ValueError, match="Semantic and lexical matrices must have the same shape"
    ):
        hybrid_similarity_matrix(semantic_df, lexical_df)


def test_hybrid_similarity_matrix_index_mismatch():
    semantic_df = pd.DataFrame(
        {"doc1": [1.0, 0.8], "doc2": [0.8, 1.0]}, index=["doc1", "doc2"]
    )

    lexical_df = pd.DataFrame(
        {"docA": [1.0, 0.6], "docB": [0.6, 1.0]}, index=["docA", "docB"]
    )

    with pytest.raises(
        ValueError,
        match="Semantic and lexical matrices must have the same index and columns",
    ):
        hybrid_similarity_matrix(semantic_df, lexical_df)


# ── Stop-word filtering (issue #222) ──────────────────────────────────────────
# Common function words (the, and, is, …) must not inflate lexical similarity.
# These tests exercise both the TF-IDF path and the standalone Jaccard helper.


def test_remove_stopwords_strips_common_function_words():
    """The high-frequency words named in the issue are filtered out."""
    text = "the cat and the dog are playing in the garden"
    filtered = remove_stopwords(text)
    # "the", "and", "are", "in" are stop-words and must be gone;
    # content words remain.
    assert "the" not in filtered.split()
    assert "and" not in filtered.split()
    assert "are" not in filtered.split()
    assert "in" not in filtered.split()
    for content_word in ("cat", "dog", "playing", "garden"):
        assert content_word in filtered.split()


def test_remove_stopwords_handles_empty_and_all_stopwords():
    assert remove_stopwords("") == ""
    assert remove_stopwords("the and is a of") == ""


def test_remove_stopwords_preserves_content_words_only():
    assert remove_stopwords("Machine learning is awesome") == "machine learning awesome"


def test_tokenize_returns_stopword_free_set():
    tokens = tokenize("The quick brown fox jumps over the lazy dog")
    assert isinstance(tokens, set)
    assert "the" not in tokens
    # Content words are kept; "over" may or may not be a stop-word
    # depending on whether NLTK is available, so only assert on the
    # unambiguous content tokens.
    assert {"quick", "brown", "fox", "jumps", "lazy", "dog"} <= tokens


def test_jaccard_similarity_identical_content():
    text = "machine learning models predict outcomes"
    assert jaccard_similarity(text, text) == 1.0


def test_jaccard_similarity_unrelated_content_is_low():
    """Two essays that share only stop-words must score ~0, not ~1."""
    essay_a = "the history of ancient rome and its emperors"
    essay_b = "the fundamentals of quantum mechanics and particles"
    score = jaccard_similarity(essay_a, essay_b)
    # After stop-word removal the only shared token is "of" if it slips
    # through, but "of" is in the fallback list too — so overlap is ~0.
    assert score <= 0.2, f"expected low similarity, got {score}"


def test_jaccard_similarity_partial_overlap_is_between_zero_and_one():
    a = "neural networks for image classification"
    b = "neural networks for sequence prediction"
    score = jaccard_similarity(a, b)
    assert 0.0 < score < 1.0


def test_jaccard_similarity_empty_inputs_return_zero():
    assert jaccard_similarity("", "") == 0.0
    assert jaccard_similarity("the and is", "the and is") == 0.0


def test_lexical_similarity_matrix_filters_stopwords():
    """Two documents that differ only in stop-words should be near-identical
    after filtering (they carry the same content), while a document that
    shares ONLY stop-words with another should have low similarity.

    This is the core regression guard for issue #222.
    """
    # Same content words, different stop-words → should still be very similar
    # because stop-words are now filtered out before TF-IDF.
    docs = {
        "doc1": "the cat sat on the mat",
        "doc2": "a cat is on a mat",
        "doc3": "dogs run in the park",
    }
    df = lexical_similarity_matrix(docs, use_cache=False)

    # doc1 vs doc2: identical content words (cat, sat/on, mat) → high
    assert df.loc["doc1", "doc2"] > 0.5

    # doc1 vs doc3: no content-word overlap → low (this is the bug #222 fix)
    assert df.loc["doc1", "doc3"] < 0.3


def test_stopwords_set_is_nonempty_and_contains_core_words():
    """The module-level STOPWORDS set must be populated and contain at least
    the words explicitly called out in the issue description."""
    assert len(STOPWORDS) > 0
    for word in ("the", "and", "is"):
        assert word in STOPWORDS


# ── Per-Paragraph Similarity Breakdown Tests ──────────────────────────────────


def test_calculate_paragraph_similarity_breakdown_matches_highest_pairs():
    """Each paragraph in Doc A must map to the highest-matching paragraph in Doc B."""
    # emb_a: 3 paragraphs in a 3-dim space (identity-like rows)
    emb_a = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # emb_b: 3 paragraphs – para 0 matches A[1], para 1 matches A[0], para 2 matches A[2]
    emb_b = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    breakdown = calculate_paragraph_similarity_breakdown(emb_a, emb_b)

    # Must return one tuple per paragraph in Doc A
    assert len(breakdown) == 3

    # Each result must be (paragraph_a_idx, paragraph_b_idx, score)
    for item in breakdown:
        assert len(item) == 3
        idx_a, idx_b, score = item
        assert isinstance(idx_a, int)
        assert isinstance(idx_b, int)
        assert 0.0 <= score <= 1.0

    # Build a lookup: paragraph_a_idx → (paragraph_b_idx, score)
    lookup = {idx_a: (idx_b, score) for idx_a, idx_b, score in breakdown}

    # A[0] = [1,0,0] → best match is B[1] = [1,0,0] (score ~1.0)
    assert lookup[0][0] == 1
    assert np.isclose(lookup[0][1], 1.0, atol=1e-5)

    # A[1] = [0,1,0] → best match is B[0] = [0,1,0] (score ~1.0)
    assert lookup[1][0] == 0
    assert np.isclose(lookup[1][1], 1.0, atol=1e-5)

    # A[2] = [0,0,1] → best match is B[2] = [0,0,1] (score ~1.0)
    assert lookup[2][0] == 2
    assert np.isclose(lookup[2][1], 1.0, atol=1e-5)

    # Sorted by descending score
    scores = [s for _, _, s in breakdown]
    assert scores == sorted(scores, reverse=True)


def test_calculate_paragraph_similarity_breakdown_empty_embeddings():
    """Returns empty list when either embedding matrix is empty."""
    emb_a = np.array([[1.0, 0.0, 0.0]])
    emb_empty = np.array([])

    assert calculate_paragraph_similarity_breakdown(emb_empty, emb_a) == []
    assert calculate_paragraph_similarity_breakdown(emb_a, emb_empty) == []
    assert calculate_paragraph_similarity_breakdown(emb_empty, emb_empty) == []


def test_calculate_paragraph_similarity_breakdown_single_paragraph_1d():
    """Handles 1-D (single paragraph) embeddings for both documents."""
    emb_a = np.array([1.0, 0.0, 0.0])   # 1-D – single paragraph
    emb_b = np.array([1.0, 0.0, 0.0])   # identical paragraph

    breakdown = calculate_paragraph_similarity_breakdown(emb_a, emb_b)

    assert len(breakdown) == 1
    idx_a, idx_b, score = breakdown[0]
    assert idx_a == 0
    assert idx_b == 0
    assert np.isclose(score, 1.0, atol=1e-5)


def test_calculate_paragraph_similarity_breakdown_asymmetric_doc_sizes():
    """Doc A may have a different number of paragraphs than Doc B."""
    # 2 paragraphs in A, 4 paragraphs in B
    emb_a = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    emb_b = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ])

    breakdown = calculate_paragraph_similarity_breakdown(emb_a, emb_b)

    # One result per paragraph in Doc A (= 2)
    assert len(breakdown) == 2
    para_a_indices = {t[0] for t in breakdown}
    assert para_a_indices == {0, 1}

    # All para_b_idx values must be valid indices into emb_b
    for _, idx_b, _ in breakdown:
        assert 0 <= idx_b < emb_b.shape[0]

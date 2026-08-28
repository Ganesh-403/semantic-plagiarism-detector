# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest

from src.core import (
    HybridSimilarityEngine,
    LexicalSimilarityEngine,
    SemanticSimilarityEngine,
    SimilarityEngineFactory,
)


def test_factory_creation():
    # Test Semantic
    engine = SimilarityEngineFactory.create_engine("semantic")
    assert isinstance(engine, SemanticSimilarityEngine)

    # Test Lexical
    engine = SimilarityEngineFactory.create_engine("lexical", algorithm="jaccard")
    assert isinstance(engine, LexicalSimilarityEngine)
    assert engine.algorithm == "jaccard"

    # Test Hybrid
    engine = SimilarityEngineFactory.create_engine("hybrid", alpha=0.6)
    assert isinstance(engine, HybridSimilarityEngine)
    assert engine.alpha == 0.6
    assert isinstance(engine.semantic_engine, SemanticSimilarityEngine)
    assert isinstance(engine.lexical_engine, LexicalSimilarityEngine)

    # Test invalid engine type
    with pytest.raises(ValueError, match="Unknown similarity engine type"):
        SimilarityEngineFactory.create_engine("invalid_type")


def test_semantic_similarity_engine_pairwise(dummy_embeddings):
    engine = SemanticSimilarityEngine()

    emb_a = dummy_embeddings["doc_A"]
    emb_b = dummy_embeddings["doc_B"]
    emb_c = dummy_embeddings["doc_C"]

    # Test with embeddings directly
    sim_ab = engine.compute_pairwise_similarity(emb_a, emb_b)
    sim_ac = engine.compute_pairwise_similarity(emb_a, emb_c)

    assert sim_ab > 0.8
    assert sim_ac < 0.2
    assert 0.0 <= sim_ab <= 1.0
    assert 0.0 <= sim_ac <= 1.0

    # Test self-similarity
    sim_aa = engine.compute_pairwise_similarity(emb_a, emb_a)
    assert np.isclose(sim_aa, 1.0)


def test_semantic_similarity_engine_matrix(dummy_embeddings):
    engine = SemanticSimilarityEngine()

    # Test dictionary inputs
    matrix = engine.compute_matrix(dummy_embeddings)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    # Self-similarities on diagonal should be close to 1.0
    for i in range(3):
        assert np.isclose(matrix[i, i], 1.0)

    # Test list inputs
    emb_list = [dummy_embeddings["doc_A"], dummy_embeddings["doc_B"]]
    matrix_list = engine.compute_matrix(emb_list)
    assert matrix_list.shape == (2, 2)
    assert np.isclose(matrix_list[0, 0], 1.0)
    assert np.isclose(matrix_list[1, 1], 1.0)


def test_lexical_similarity_engine_algorithms():
    doc1 = "The quick brown fox jumps over the lazy dog"
    doc2 = "A quick brown fox jumps over a lazy dog"
    doc3 = "Hello world from python unit tests"

    # 1. Jaccard
    jaccard_engine = LexicalSimilarityEngine(algorithm="jaccard")
    sim_ab = jaccard_engine.compute_pairwise_similarity(doc1, doc2)
    sim_ac = jaccard_engine.compute_pairwise_similarity(doc1, doc3)
    assert sim_ab > 0.5
    assert sim_ac == 0.0

    # 2. Dice
    dice_engine = LexicalSimilarityEngine(algorithm="dice")
    sim_ab_dice = dice_engine.compute_pairwise_similarity(doc1, doc2)
    assert sim_ab_dice > sim_ab  # Sørensen-Dice is typically higher than Jaccard

    # 3. Overlap
    overlap_engine = LexicalSimilarityEngine(algorithm="overlap")
    sim_ab_overlap = overlap_engine.compute_pairwise_similarity(doc1, doc2)
    assert 0.0 <= sim_ab_overlap <= 1.0

    # 4. Word N-gram
    ngram_engine = LexicalSimilarityEngine(algorithm="ngram", ngram_size=2)
    sim_ab_ngram = ngram_engine.compute_pairwise_similarity(doc1, doc2)
    assert 0.0 <= sim_ab_ngram <= 1.0

    # 5. Char N-gram
    char_ngram_engine = LexicalSimilarityEngine(algorithm="char_ngram", ngram_size=5)
    sim_ab_char = char_ngram_engine.compute_pairwise_similarity(doc1, doc2)
    assert sim_ab_char > 0.5

    # 6. Levenshtein
    lev_engine = LexicalSimilarityEngine(algorithm="levenshtein")
    sim_ab_lev = lev_engine.compute_pairwise_similarity(doc1, doc2)
    assert sim_ab_lev > 0.7


def test_lexical_similarity_engine_matrix():
    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog",
        "doc2": "A quick brown fox jumps over a lazy dog",
        "doc3": "Hello world from python unit tests",
    }

    # TF-IDF matrix
    engine = LexicalSimilarityEngine(algorithm="tfidf")
    matrix = engine.compute_matrix(docs)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    assert np.isclose(matrix[0, 0], 1.0)
    assert matrix[0, 1] > 0.5
    assert matrix[0, 2] == 0.0

    # List inputs for Jaccard
    jaccard_engine = LexicalSimilarityEngine(algorithm="jaccard")
    list_matrix = jaccard_engine.compute_matrix(list(docs.values()))
    assert list_matrix.shape == (3, 3)
    assert np.isclose(list_matrix[0, 0], 1.0)
    assert list_matrix[0, 1] > 0.5
    assert list_matrix[0, 2] == 0.0


def test_hybrid_similarity_engine(dummy_embeddings):
    # Setup mock or helper to get embeddings or strings
    # We can mock lexical engine to return 0.5 and semantic to return 0.8
    sem_engine = SemanticSimilarityEngine()
    lex_engine = LexicalSimilarityEngine(algorithm="jaccard")

    hybrid_engine = HybridSimilarityEngine(
        semantic_engine=sem_engine,
        lexical_engine=lex_engine,
        alpha=0.7,
    )

    doc1 = "The quick brown fox jumps over the lazy dog"
    doc2 = "A quick brown fox jumps over a lazy dog"

    # Compute pairwise
    score = hybrid_engine.compute_pairwise_similarity(doc1, doc2)
    assert 0.0 <= score <= 1.0

    # Compute matrix on strings
    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog",
        "doc2": "A quick brown fox jumps over a lazy dog",
    }
    matrix = hybrid_engine.compute_matrix(docs)
    assert matrix.shape == (2, 2)
    assert np.isclose(matrix[0, 0], 1.0)

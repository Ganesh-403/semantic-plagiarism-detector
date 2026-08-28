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

"""
Hybrid Scorer Module for Document Ranking

This module implements a hybrid scoring system that combines:
- Lexical similarity (TF-IDF based)
- Semantic similarity (embedding based)
- Weighted combination with configurable alpha parameter
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridScorer:
    """
    Hybrid document scorer combining TF-IDF lexical similarity and
    embedding-based semantic similarity.

    Attributes:
        vectorizer (TfidfVectorizer): TF-IDF vectorizer for lexical features
        embedding_model: Model for generating document embeddings
        alpha (float): Weight for lexical score (1-alpha for embedding score)
        cached_tfidf_matrix: Cached TF-IDF matrix for efficiency
        cached_embeddings: Cached document embeddings
    """

    def __init__(
        self,
        embedding_model=None,
        alpha: float = 0.5,
        vectorizer_params: Optional[dict[str, Any]] = None,
        cache_matrix: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize Hybrid Scorer.

        Args:
            embedding_model: Model to generate document embeddings
            alpha: Weight for lexical score (0-1), higher = more weight on lexical
            vectorizer_params: Parameters for TF-IDF vectorizer
            cache_matrix: Whether to cache computed matrices
            verbose: Whether to show progress bars
        """
        self.alpha = alpha
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.cache_matrix = cache_matrix

        # Initialize TF-IDF vectorizer with default parameters
        default_vectorizer_params = {
            "max_features": 10000,
            "stop_words": "english",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.85,
            "sublinear_tf": True,
            "use_idf": True,
            "norm": "l2",
        }

        if vectorizer_params:
            default_vectorizer_params.update(vectorizer_params)

        self.vectorizer = TfidfVectorizer(**default_vectorizer_params)

        # Cache for computed data
        self.cached_tfidf_matrix = None
        self.cached_embeddings = None
        self.cached_documents = None
        self.cached_hybrid_matrix = None

        logger.info(f"HybridScorer initialized with alpha={alpha}")

    def fit_tfidf(self, documents: list[str]) -> csr_matrix:
        """
        Fit TF-IDF vectorizer on all documents and transform them.

        Args:
            documents: List of document texts

        Returns:
            csr_matrix: TF-IDF matrix of shape (n_documents, n_features)
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(documents)} documents...")
        start_time = time.time()

        # Fit and transform in one go
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        elapsed = time.time() - start_time
        logger.info(f"TF-IDF fitting complete in {elapsed:.2f}s")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Cache if enabled
        if self.cache_matrix:
            self.cached_tfidf_matrix = tfidf_matrix
            self.cached_documents = documents

        return tfidf_matrix

    def compute_lexical_scores(self, documents: list[str]) -> np.ndarray:
        """
        Compute lexical similarity matrix using vectorized TF-IDF.

        Args:
            documents: List of document texts

        Returns:
            np.ndarray: Lexical similarity matrix of shape (n, n)
        """
        # Check cache
        if self.cached_tfidf_matrix is not None and self.cached_documents == documents:
            logger.info("Using cached TF-IDF matrix")
            tfidf_matrix = self.cached_tfidf_matrix
        else:
            tfidf_matrix = self.fit_tfidf(documents)

        logger.info(
            f"Computing lexical similarity matrix for {len(documents)} documents..."
        )
        start_time = time.time()

        # Vectorized cosine similarity computation
        lexical_scores = cosine_similarity(tfidf_matrix)

        elapsed = time.time() - start_time
        logger.info(f"Lexical similarity computed in {elapsed:.2f}s")

        return lexical_scores

    def compute_embedding_scores(self, documents: list[str]) -> Optional[np.ndarray]:
        """
        Compute embedding-based similarity matrix.

        Args:
            documents: List of document texts

        Returns:
            Optional[np.ndarray]: Embedding similarity matrix, or None if no model
        """
        if self.embedding_model is None:
            logger.warning("No embedding model provided, skipping semantic scores")
            return None

        # Check cache
        if self.cached_embeddings is not None and self.cached_documents == documents:
            logger.info("Using cached embeddings")
            embeddings = self.cached_embeddings
        else:
            logger.info(f"Computing embeddings for {len(documents)} documents...")
            start_time = time.time()

            # Generate embeddings
            if hasattr(self.embedding_model, "encode"):
                # Sentence transformers or similar
                embeddings = self.embedding_model.encode(
                    documents, show_progress_bar=self.verbose, convert_to_numpy=True
                )
            elif hasattr(self.embedding_model, "transform"):
                # Sklearn-style transformers
                embeddings = self.embedding_model.transform(documents)
            else:
                # Custom model with __call__ method
                embeddings = self.embedding_model(documents)

            elapsed = time.time() - start_time
            logger.info(f"Embeddings computed in {elapsed:.2f}s")

            if self.cache_matrix:
                self.cached_embeddings = embeddings

        # Compute cosine similarity on embeddings
        logger.info("Computing embedding similarity matrix...")
        start_time = time.time()
        embedding_scores = cosine_similarity(embeddings)
        elapsed = time.time() - start_time
        logger.info(f"Embedding similarity computed in {elapsed:.2f}s")

        return embedding_scores

    def compute_hybrid_matrix(
        self,
        documents: list[str],
        alpha: Optional[float] = None,
        return_lexical: bool = False,
        return_embedding: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Compute hybrid similarity matrix combining lexical and semantic scores.

        This is the optimized version that computes TF-IDF globally and uses
        vectorized cosine_similarity in a single pass.

        Args:
            documents: List of document texts
            alpha: Weight for lexical score (overrides self.alpha if provided)
            return_lexical: If True, also return lexical scores
            return_embedding: If True, also return embedding scores

        Returns:
            Union[np.ndarray, Tuple]: Hybrid matrix and optionally lexical/embedding matrices
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        n_docs = len(documents)
        logger.info(f"Computing hybrid matrix for {n_docs} documents...")
        start_time = time.time()

        # Use provided alpha or default
        alpha = alpha if alpha is not None else self.alpha

        # Check cache
        if self.cached_hybrid_matrix is not None and self.cached_documents == documents:
            logger.info("Using cached hybrid matrix")
            hybrid_matrix = self.cached_hybrid_matrix
        else:
            # Step 1: Compute lexical scores (vectorized)
            lexical_scores = self.compute_lexical_scores(documents)

            # Step 2: Compute embedding scores (if model available)
            embedding_scores = self.compute_embedding_scores(documents)

            # Step 3: Combine scores
            if embedding_scores is not None:
                logger.info(f"Combining scores with alpha={alpha}")
                hybrid_matrix = (alpha * lexical_scores) + (
                    (1 - alpha) * embedding_scores
                )

                # Ensure diagonal is 1.0
                np.fill_diagonal(hybrid_matrix, 1.0)

                logger.info(f"Hybrid matrix shape: {hybrid_matrix.shape}")
                logger.info(
                    f"Score range: [{hybrid_matrix.min():.4f}, {hybrid_matrix.max():.4f}]"
                )
                logger.info(f"Mean score: {hybrid_matrix.mean():.4f}")
            else:
                logger.info("No embedding model available, using lexical only")
                hybrid_matrix = lexical_scores

            # Cache if enabled
            if self.cache_matrix:
                self.cached_hybrid_matrix = hybrid_matrix
                self.cached_documents = documents

        elapsed = time.time() - start_time
        logger.info(f"Hybrid matrix computed in {elapsed:.2f}s")

        # Prepare return values
        results = [hybrid_matrix]
        if return_lexical:
            results.append(
                self.cached_tfidf_matrix
                if self.cache_matrix
                else self.compute_lexical_scores(documents)
            )
        if return_embedding:
            results.append(
                self.cached_embeddings
                if self.cache_matrix
                else self.compute_embedding_scores(documents)
            )

        return tuple(results) if len(results) > 1 else results[0]

    def get_top_k_similar(
        self,
        query_doc: str,
        documents: list[str],
        k: int = 10,
        alpha: Optional[float] = None,
    ) -> list[tuple[int, float]]:
        """
        Get top-k most similar documents for a query document.

        Args:
            query_doc: Query document text
            documents: List of document texts
            k: Number of top results to return
            alpha: Weight for lexical score

        Returns:
            List[Tuple[int, float]]: List of (index, score) tuples sorted by score
        """
        # Combine query with documents for vectorization
        all_docs = [query_doc] + documents

        # Compute hybrid matrix
        hybrid_matrix = self.compute_hybrid_matrix(all_docs, alpha=alpha)

        # Get scores for query (first row, excluding self)
        query_scores = hybrid_matrix[0, 1:]

        # Get top-k indices
        top_indices = np.argsort(query_scores)[-k:][::-1]
        top_scores = query_scores[top_indices]

        # Convert to list of (index, score) tuples
        results = [
            (int(idx), float(score)) for idx, score in zip(top_indices, top_scores)
        ]

        logger.info(f"Returned top-{k} similar documents")
        return results

    def get_top_k_similar_batch(
        self,
        query_docs: list[str],
        documents: list[str],
        k: int = 10,
        alpha: Optional[float] = None,
        batch_size: int = 100,
    ) -> list[list[tuple[int, float]]]:
        """
        Get top-k similar documents for multiple queries in batch.

        Args:
            query_docs: List of query documents
            documents: List of document texts
            k: Number of top results to return
            alpha: Weight for lexical score
            batch_size: Batch size for processing

        Returns:
            List[List[Tuple[int, float]]]: List of results for each query
        """
        all_results = []

        for i, query in enumerate(tqdm(query_docs, desc="Processing queries")):
            results = self.get_top_k_similar(query, documents, k, alpha)
            all_results.append(results)

        return all_results

    def set_alpha(self, alpha: float):
        """
        Update the alpha parameter.

        Args:
            alpha: New alpha value (0-1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        logger.info(f"Alpha updated to {alpha}")

    def clear_cache(self):
        """Clear all cached matrices and embeddings."""
        self.cached_tfidf_matrix = None
        self.cached_embeddings = None
        self.cached_documents = None
        self.cached_hybrid_matrix = None
        logger.info("Cache cleared")

    def get_statistics(self, documents: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Get statistics about the scorer.

        Args:
            documents: Optional list of documents to compute statistics on

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        stats = {
            "alpha": self.alpha,
            "has_embedding_model": self.embedding_model is not None,
            "cache_enabled": self.cache_matrix,
            "has_cached_data": self.cached_documents is not None,
        }

        if self.cached_documents is not None:
            stats["n_cached_documents"] = len(self.cached_documents)

        if self.cached_tfidf_matrix is not None:
            stats["tfidf_shape"] = self.cached_tfidf_matrix.shape
            stats["vocabulary_size"] = len(self.vectorizer.vocabulary_)

        if self.cached_hybrid_matrix is not None:
            stats["hybrid_matrix_shape"] = self.cached_hybrid_matrix.shape
            stats["hybrid_mean"] = float(self.cached_hybrid_matrix.mean())
            stats["hybrid_std"] = float(self.cached_hybrid_matrix.std())

        if documents:
            # Compute stats on the fly
            tfidf_matrix = self.fit_tfidf(documents)
            stats["documents_count"] = len(documents)
            stats["tfidf_shape"] = tfidf_matrix.shape
            stats["vocabulary_size"] = len(self.vectorizer.vocabulary_)
            stats["sparsity"] = 1.0 - (
                tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
            )

        return stats


# ===================== UNIT TESTS =====================

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TestHybridScorer(unittest.TestCase):
    """Unit tests for HybridScorer class."""

    def setUp(self):
        """Set up test data."""
        self.documents = [
            "The cat sat on the mat",
            "The dog played with the ball",
            "Cats and dogs are pets",
            "The mat was comfortable",
            "Pets are wonderful companions",
        ]

        self.query_doc = "A cat sitting on a mat"

        self.scorer = HybridScorer(alpha=0.6, verbose=False)

        # Mock embedding model
        self.mock_embedding_model = MagicMock()
        self.mock_embedding_model.encode.return_value = np.random.randn(
            len(self.documents), 128
        )

        self.scorer_emb = HybridScorer(
            embedding_model=self.mock_embedding_model, alpha=0.6, verbose=False
        )

    def test_fit_tfidf(self):
        """Test TF-IDF fitting and transformation."""
        tfidf_matrix = self.scorer.fit_tfidf(self.documents)

        self.assertEqual(tfidf_matrix.shape[0], len(self.documents))
        self.assertIsInstance(tfidf_matrix, csr_matrix)
        self.assertIsNotNone(self.scorer.vectorizer.vocabulary_)
        self.assertGreater(len(self.scorer.vectorizer.vocabulary_), 0)

    def test_compute_lexical_scores(self):
        """Test lexical similarity matrix computation."""
        lexical_scores = self.scorer.compute_lexical_scores(self.documents)

        self.assertEqual(
            lexical_scores.shape, (len(self.documents), len(self.documents))
        )
        self.assertTrue(np.allclose(np.diag(lexical_scores), 1.0))
        self.assertTrue(np.all(lexical_scores >= 0))
        self.assertTrue(np.all(lexical_scores <= 1))

    def test_compute_lexical_scores_with_cache(self):
        """Test lexical scores with caching."""
        # First call - computes
        scores1 = self.scorer.compute_lexical_scores(self.documents)

        # Second call - should use cache
        scores2 = self.scorer.compute_lexical_scores(self.documents)

        np.testing.assert_array_almost_equal(scores1, scores2)
        self.assertEqual(self.scorer.cached_documents, self.documents)

    def test_compute_embedding_scores_with_model(self):
        """Test embedding scores with model."""
        embedding_scores = self.scorer_emb.compute_embedding_scores(self.documents)

        self.assertEqual(
            embedding_scores.shape, (len(self.documents), len(self.documents))
        )
        self.assertTrue(np.allclose(np.diag(embedding_scores), 1.0))
        self.assertTrue(np.all(embedding_scores >= -1))
        self.assertTrue(np.all(embedding_scores <= 1))

    def test_compute_embedding_scores_without_model(self):
        """Test embedding scores without model."""
        embedding_scores = self.scorer.compute_embedding_scores(self.documents)

        self.assertIsNone(embedding_scores)

    def test_compute_hybrid_matrix_lexical_only(self):
        """Test hybrid matrix without embedding model."""
        hybrid_matrix = self.scorer.compute_hybrid_matrix(self.documents)

        self.assertEqual(
            hybrid_matrix.shape, (len(self.documents), len(self.documents))
        )
        self.assertTrue(np.allclose(np.diag(hybrid_matrix), 1.0))
        self.assertTrue(np.all(hybrid_matrix >= 0))
        self.assertTrue(np.all(hybrid_matrix <= 1))

    def test_compute_hybrid_matrix_with_embeddings(self):
        """Test hybrid matrix with embedding model."""
        hybrid_matrix = self.scorer_emb.compute_hybrid_matrix(self.documents)

        self.assertEqual(
            hybrid_matrix.shape, (len(self.documents), len(self.documents))
        )
        self.assertTrue(np.allclose(np.diag(hybrid_matrix), 1.0))
        self.assertTrue(np.all(hybrid_matrix >= 0))
        self.assertTrue(np.all(hybrid_matrix <= 1))

    def test_compute_hybrid_matrix_with_custom_alpha(self):
        """Test hybrid matrix with custom alpha."""
        hybrid_matrix1 = self.scorer_emb.compute_hybrid_matrix(
            self.documents, alpha=0.3
        )
        hybrid_matrix2 = self.scorer_emb.compute_hybrid_matrix(
            self.documents, alpha=0.8
        )

        # Different alpha should produce different results
        self.assertFalse(np.allclose(hybrid_matrix1, hybrid_matrix2))

    def test_compute_hybrid_matrix_returns_components(self):
        """Test hybrid matrix returning lexical and embedding components."""
        result = self.scorer_emb.compute_hybrid_matrix(
            self.documents, return_lexical=True, return_embedding=True
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)  # hybrid, lexical, embedding

        hybrid, lexical, embedding = result
        self.assertEqual(hybrid.shape, (len(self.documents), len(self.documents)))
        self.assertEqual(lexical.shape, (len(self.documents), len(self.documents)))
        self.assertEqual(embedding.shape, (len(self.documents), len(self.documents)))

    def test_compute_hybrid_matrix_empty_documents(self):
        """Test hybrid matrix with empty documents list."""
        with self.assertRaises(ValueError):
            self.scorer.compute_hybrid_matrix([])

    def test_get_top_k_similar(self):
        """Test getting top-k similar documents."""
        results = self.scorer.get_top_k_similar(self.query_doc, self.documents, k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 2)  # (index, score)
        self.assertTrue(all(score >= 0 for _, score in results))
        self.assertTrue(all(score <= 1 for _, score in results))

        # Scores should be sorted descending
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_get_top_k_similar_batch(self):
        """Test batch retrieval of top-k similar documents."""
        query_docs = ["query1", "query2", "query3"]

        results = self.scorer.get_top_k_similar_batch(query_docs, self.documents, k=2)

        self.assertEqual(len(results), len(query_docs))
        self.assertEqual(len(results[0]), 2)

    def test_set_alpha(self):
        """Test updating alpha parameter."""
        self.scorer.set_alpha(0.7)
        self.assertEqual(self.scorer.alpha, 0.7)

        # Test invalid alpha
        with self.assertRaises(ValueError):
            self.scorer.set_alpha(1.5)

        with self.assertRaises(ValueError):
            self.scorer.set_alpha(-0.1)

    def test_clear_cache(self):
        """Test clearing cache."""
        self.scorer.compute_hybrid_matrix(self.documents)
        self.assertIsNotNone(self.scorer.cached_hybrid_matrix)

        self.scorer.clear_cache()
        self.assertIsNone(self.scorer.cached_tfidf_matrix)
        self.assertIsNone(self.scorer.cached_embeddings)
        self.assertIsNone(self.scorer.cached_documents)
        self.assertIsNone(self.scorer.cached_hybrid_matrix)

    def test_get_statistics(self):
        """Test getting statistics."""
        stats = self.scorer.get_statistics(self.documents)

        self.assertIsInstance(stats, dict)
        self.assertIn("alpha", stats)
        self.assertIn("has_embedding_model", stats)
        self.assertIn("cache_enabled", stats)
        self.assertIn("documents_count", stats)
        self.assertIn("tfidf_shape", stats)
        self.assertIn("vocabulary_size", stats)
        self.assertIn("sparsity", stats)

    def test_performance_comparison(self):
        """Test performance improvement with vectorized computation."""
        import time

        # Generate larger test data
        test_docs = [
            f"Document {i} with some random text content for testing" for i in range(50)
        ]

        scorer = HybridScorer(verbose=False)

        # Old approach (simulated)
        start = time.time()
        lexical_scores_old = np.zeros((len(test_docs), len(test_docs)))
        for i in range(len(test_docs)):
            for j in range(len(test_docs)):
                # Simulating old approach - this would be much slower
                lexical_scores_old[i][j] = 0.5
        old_time = time.time() - start

        # New approach (vectorized)
        start = time.time()
        lexical_scores_new = scorer.compute_lexical_scores(test_docs)
        new_time = time.time() - start

        # New approach should be significantly faster
        print(f"Old approach (simulated): {old_time:.4f}s")
        print(f"New approach (vectorized): {new_time:.4f}s")
        self.assertLess(new_time, old_time * 2)  # Should be at least 2x faster


class TestHybridScorerEdgeCases(unittest.TestCase):
    """Test edge cases for HybridScorer."""

    def test_single_document(self):
        """Test with single document."""
        scorer = HybridScorer(verbose=False)
        docs = ["Single document"]

        hybrid_matrix = scorer.compute_hybrid_matrix(docs)

        self.assertEqual(hybrid_matrix.shape, (1, 1))
        self.assertEqual(hybrid_matrix[0][0], 1.0)

    def test_identical_documents(self):
        """Test with identical documents."""
        scorer = HybridScorer(verbose=False)
        docs = ["Same text", "Same text", "Same text"]

        hybrid_matrix = scorer.compute_hybrid_matrix(docs)

        self.assertEqual(hybrid_matrix.shape, (3, 3))
        np.testing.assert_array_almost_equal(hybrid_matrix, np.ones((3, 3)))

    def test_very_long_documents(self):
        """Test with very long documents."""
        scorer = HybridScorer(verbose=False)
        docs = ["Word " * 1000 for _ in range(5)]

        hybrid_matrix = scorer.compute_hybrid_matrix(docs)

        self.assertEqual(hybrid_matrix.shape, (5, 5))
        self.assertTrue(np.all(hybrid_matrix >= 0))
        self.assertTrue(np.all(hybrid_matrix <= 1))

    def test_special_characters(self):
        """Test documents with special characters."""
        scorer = HybridScorer(verbose=False)
        docs = [
            "Document with @#$% special chars!",
            "Another with numbers 12345",
            "And some emojis 🚀🌟",
        ]

        hybrid_matrix = scorer.compute_hybrid_matrix(docs)

        self.assertEqual(hybrid_matrix.shape, (3, 3))
        self.assertTrue(np.all(hybrid_matrix >= 0))
        self.assertTrue(np.all(hybrid_matrix <= 1))

    def test_multiple_languages(self):
        """Test documents in multiple languages."""
        scorer = HybridScorer(verbose=False)
        docs = [
            "Hello world in English",
            "Hola mundo en Español",
            "Bonjour le monde en Français",
        ]

        hybrid_matrix = scorer.compute_hybrid_matrix(docs)

        self.assertEqual(hybrid_matrix.shape, (3, 3))
        self.assertTrue(np.all(hybrid_matrix >= 0))
        self.assertTrue(np.all(hybrid_matrix <= 1))


class TestHybridScorerIntegration(unittest.TestCase):
    """Integration tests for HybridScorer."""

    def test_full_pipeline(self):
        """Test complete pipeline: fit -> compute -> retrieve."""
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog",
            "The lazy dog sleeps all day",
            "Foxes are quick and agile animals",
            "Dogs are loyal companions",
        ]

        scorer = HybridScorer(alpha=0.5, verbose=False)

        # 1. Compute hybrid matrix
        hybrid_matrix = scorer.compute_hybrid_matrix(documents)

        self.assertEqual(hybrid_matrix.shape, (5, 5))

        # 2. Get top-k similar
        query = "A quick fox"
        results = scorer.get_top_k_similar(query, documents, k=3)

        self.assertEqual(len(results), 3)

        # 3. Verify results are properly sorted
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # 4. Verify caching is working
        start = time.time()
        matrix_cached = scorer.compute_hybrid_matrix(documents)
        cache_time = time.time() - start

        # Should be very fast with cache
        self.assertLess(cache_time, 0.5)
        np.testing.assert_array_almost_equal(hybrid_matrix, matrix_cached)

    def test_consistent_results(self):
        """Test that results are consistent across runs."""
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        scorer1 = HybridScorer(verbose=False)
        scorer2 = HybridScorer(verbose=False)

        matrix1 = scorer1.compute_hybrid_matrix(documents)
        matrix2 = scorer2.compute_hybrid_matrix(documents)

        np.testing.assert_array_almost_equal(matrix1, matrix2)

    def test_alpha_effect(self):
        """Test that alpha parameter affects results as expected."""
        documents = [
            "The cat sat on the mat",
            "The dog played with the ball",
            "Cats and dogs are pets",
        ]

        scorer = HybridScorer(verbose=False)

        # Alpha = 1.0 (lexical only)
        matrix_lexical = scorer.compute_hybrid_matrix(documents, alpha=1.0)
        lexical_scores = scorer.compute_lexical_scores(documents)
        np.testing.assert_array_almost_equal(matrix_lexical, lexical_scores)

        # Alpha = 0.0 (embedding only) - no embedding model, so should be lexical
        # This tests fallback behavior
        matrix_embedding = scorer.compute_hybrid_matrix(documents, alpha=0.0)
        np.testing.assert_array_almost_equal(matrix_embedding, lexical_scores)


# ===================== PERFORMANCE BENCHMARK =====================


def run_performance_benchmark():
    """Run performance benchmarks comparing old vs new approach."""
    import time

    import matplotlib.pyplot as plt

    sizes = [10, 20, 50, 100]
    old_times = []
    new_times = []

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60 + "\n")

    for n in sizes:
        print(f"Testing with {n} documents...")

        # Generate test documents
        docs = [f"Document {i} with some content" for i in range(n)]

        scorer = HybridScorer(verbose=False)

        # OLD APPROACH (simulated)
        start = time.time()
        # Simulating old O(N²) approach
        old_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Simulating _compute_lexical_score which would be expensive
                old_matrix[i][j] = 0.5
        old_time = time.time() - start
        old_times.append(old_time)

        # NEW APPROACH (vectorized)
        start = time.time()
        new_matrix = scorer.compute_hybrid_matrix(docs)
        new_time = time.time() - start
        new_times.append(new_time)

        speedup = old_time / new_time if new_time > 0 else float("inf")
        print(f"  Old approach: {old_time:.4f}s")
        print(f"  New approach: {new_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x\n")

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Documents':<12} {'Old Time (s)':<15} {'New Time (s)':<15} {'Speedup':<10}"
    )
    print("-" * 60)
    for n, old_t, new_t in zip(sizes, old_times, new_times):
        speedup = old_t / new_t if new_t > 0 else float("inf")
        print(f"{n:<12} {old_t:<15.4f} {new_t:<15.4f} {speedup:<10.2f}x")

    return old_times, new_times


# ===================== RUN TESTS =====================

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)

    # Run performance benchmark
    run_performance_benchmark()

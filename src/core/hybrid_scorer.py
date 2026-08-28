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
Hybrid Lexical-Semantic Scoring Module

Combines lexical (TF-IDF, Jaccard) and semantic (Sentence-BERT) similarity
for more robust plagiarism detection.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.lexical_similarity import (
    STOPWORDS,
    compute_char_ngram_similarity,
    dice_coefficient,
    jaccard_similarity,
    n_gram_overlap,
    overlap_coefficient,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridScore:
    """Hybrid similarity score for a document pair."""

    doc_a: str
    doc_b: str
    semantic_score: float
    lexical_score: float
    hybrid_score: float
    alpha: float  # semantic weight
    lexical_method: str
    is_flagged: bool = False
    threshold: float = 0.59


@dataclass
class HybridConfig:
    """Configuration for hybrid scoring."""

    alpha: float = 0.7  # Weight for semantic similarity (0.7 = 70% semantic)
    lexical_method: str = (
        "tfidf"  # Options: 'tfidf', 'jaccard', 'dice', 'overlap', 'ngram', 'char_ngram'
    )
    min_threshold: float = 0.0
    use_stopwords: bool = True
    ngram_n: int = 3
    char_ngram_n: int = 5
    tfidf_max_features: int = 5000
    normalize: Optional[str] = None  # Normalization method: None, 'minmax', or 'zscore'


class LRUCache(OrderedDict):
    """Bounded LRU Cache backed by collections.OrderedDict."""

    def __init__(self, maxsize: int = 50000, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)


class HybridScorer:
    """
    Hybrid scorer combining lexical and semantic similarity.

    Features:
    - Weighted combination of lexical and semantic scores
    - Multiple lexical methods (TF-IDF, Jaccard, Dice, N-gram)
    - Configurable alpha weight
    - Parallel processing support
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._lexical_cache: LRUCache = LRUCache(maxsize=50000)
        self._stats = {
            "total_pairs": 0,
            "flagged_pairs": 0,
            "avg_semantic": 0.0,
            "avg_lexical": 0.0,
            "avg_hybrid": 0.0,
        }

    def _get_tfidf_vectorizer(self) -> TfidfVectorizer:
        """Get or create TF-IDF vectorizer."""
        if self._tfidf_vectorizer is None:
            stop_words = "english" if self.config.use_stopwords else None
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                stop_words=stop_words,
                lowercase=True,
                ngram_range=(1, 2),
            )
        return self._tfidf_vectorizer

    def _compute_tfidf_similarity(self, texts: list[str]) -> np.ndarray:
        """Compute TF-IDF similarity matrix."""
        vectorizer = self._get_tfidf_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)

    def _compute_lexical_score(
        self,
        text_a: str,
        text_b: str,
        method: str = None,
    ) -> float:
        """Compute lexical similarity using specified method."""
        method = method or self.config.lexical_method

        # Check cache
        cache_key = f"{method}:{hash(text_a)}:{hash(text_b)}"
        if cache_key in self._lexical_cache:
            return self._lexical_cache[cache_key]

        # Compute based on method
        if method == "tfidf":
            # Compute on the fly for single pair
            vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                stop_words="english" if self.config.use_stopwords else None,
                lowercase=True,
            )
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        elif method == "jaccard":
            stopwords = STOPWORDS if self.config.use_stopwords else None
            score = jaccard_similarity(text_a, text_b, stopwords=stopwords)

        elif method == "dice":
            stopwords = STOPWORDS if self.config.use_stopwords else None
            score = dice_coefficient(text_a, text_b, stopwords=stopwords)

        elif method == "overlap":
            stopwords = STOPWORDS if self.config.use_stopwords else None
            score = overlap_coefficient(text_a, text_b, stopwords=stopwords)

        elif method == "ngram":
            stopwords = STOPWORDS if self.config.use_stopwords else None
            score = n_gram_overlap(
                text_a, text_b, n=self.config.ngram_n, stopwords=stopwords
            )

        elif method == "char_ngram":
            score = compute_char_ngram_similarity(
                text_a, text_b, n=self.config.char_ngram_n
            )

        else:
            raise ValueError(f"Unknown lexical method: {method}")

        # Cache result
        self._lexical_cache[cache_key] = float(score)
        return float(score)

    def compute_hybrid_similarity(
        self,
        text_a: str,
        text_b: str,
        semantic_score: float,
        alpha: Optional[float] = None,
        lexical_method: Optional[str] = None,
    ) -> float:
        """
        Compute hybrid similarity for a single pair.

        Formula: hybrid = alpha * semantic + (1 - alpha) * lexical

        Args:
            text_a: First document text
            text_b: Second document text
            semantic_score: Semantic similarity score (0-1)
            alpha: Semantic weight (default: from config)
            lexical_method: Lexical method (default: from config)

        Returns:
            Hybrid similarity score (0-1)
        """
        alpha = alpha if alpha is not None else self.config.alpha
        lexical_method = lexical_method or self.config.lexical_method

        lexical_score = self._compute_lexical_score(text_a, text_b, lexical_method)
        hybrid_score = alpha * semantic_score + (1 - alpha) * lexical_score

        return min(1.0, max(0.0, hybrid_score))

    def compute_hybrid_matrix(
        self,
        texts: dict[str, str],
        semantic_matrix: Optional[pd.DataFrame] = None,
        alpha: Optional[float] = None,
        lexical_method: Optional[str] = None,
        normalize: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute hybrid similarity matrix for all document pairs.

        Args:
            texts: Dict mapping doc name to text content
            semantic_matrix: Pre-computed semantic matrix (optional)
            alpha: Semantic weight (default: from config)
            lexical_method: Lexical method (default: from config)
            normalize: Normalization method ('minmax', 'zscore', or None)

        Returns:
            Hybrid similarity DataFrame
        """
        from src.core.similarity import normalize_scores

        alpha = alpha if alpha is not None else self.config.alpha
        lexical_method = lexical_method or self.config.lexical_method
        normalize = normalize if normalize is not None else self.config.normalize

        doc_names = list(texts.keys())
        n = len(doc_names)

        # Compute semantic matrix if not provided
        if semantic_matrix is None:
            from src.core.similarity import document_similarity_matrix

            # Compute embeddings and matrix
            semantic_matrix = document_similarity_matrix(
                [texts[name] for name in doc_names]
            )
            if isinstance(semantic_matrix, np.ndarray):
                semantic_matrix = pd.DataFrame(
                    semantic_matrix, index=doc_names, columns=doc_names
                )

        # Compute lexical matrix
        if lexical_method == "tfidf" and n > 0:
            doc_texts = [texts[name] for name in doc_names]
            lexical_matrix = self._compute_tfidf_similarity(doc_texts)
        else:
            lexical_matrix = np.zeros((n, n))
            for i, doc_a in enumerate(doc_names):
                for j, doc_b in enumerate(doc_names):
                    if i == j:
                        lexical_matrix[i, j] = 1.0
                    elif j > i:
                        score = self._compute_lexical_score(
                            texts[doc_a], texts[doc_b], lexical_method
                        )
                        lexical_matrix[i, j] = score
                        lexical_matrix[j, i] = score

        lexical_df = pd.DataFrame(lexical_matrix, index=doc_names, columns=doc_names)

        sem_norm = normalize_scores(semantic_matrix, method=normalize)
        lex_norm = normalize_scores(lexical_df, method=normalize)

        # Combine matrices
        hybrid_matrix = alpha * sem_norm + (1 - alpha) * lex_norm
        if normalize in (None, "none", "minmax", "min-max"):
            hybrid_matrix = np.clip(hybrid_matrix, 0.0, 1.0)

        if isinstance(hybrid_matrix, pd.DataFrame):
            return hybrid_matrix
        return pd.DataFrame(hybrid_matrix, index=doc_names, columns=doc_names)

    def flag_plagiarism_hybrid(
        self,
        hybrid_df: pd.DataFrame,
        threshold: float = 0.59,
    ) -> list[dict[str, Any]]:
        """
        Flag plagiarism pairs using hybrid scores.

        Args:
            hybrid_df: Hybrid similarity DataFrame
            threshold: Flagging threshold

        Returns:
            List of flagged pairs with scores
        """
        flagged = []
        doc_names = hybrid_df.columns.tolist()

        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                score = float(hybrid_df.iloc[i, j])
                if score >= threshold:
                    flagged.append(
                        {
                            "doc_a": doc_names[i],
                            "doc_b": doc_names[j],
                            "hybrid_score": score,
                            "threshold": threshold,
                        }
                    )

        flagged.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return flagged

    def compute_pair_stats(
        self,
        text_a: str,
        text_b: str,
        semantic_score: float,
        alpha: float = 0.7,
        lexical_method: str = "tfidf",
    ) -> dict[str, Any]:
        """
        Compute detailed stats for a document pair.

        Args:
            text_a: First document text
            text_b: Second document text
            semantic_score: Semantic similarity score
            alpha: Semantic weight
            lexical_method: Lexical method

        Returns:
            Dictionary with detailed scores
        """
        lexical_score = self._compute_lexical_score(text_a, text_b, lexical_method)
        hybrid_score = alpha * semantic_score + (1 - alpha) * lexical_score

        # Compute individual lexical method scores for comparison
        methods = ["tfidf", "jaccard", "dice", "overlap", "ngram", "char_ngram"]
        method_scores = {}
        for method in methods:
            method_scores[method] = self._compute_lexical_score(text_a, text_b, method)

        return {
            "doc_a_preview": text_a[:100] + "..." if len(text_a) > 100 else text_a,
            "doc_b_preview": text_b[:100] + "..." if len(text_b) > 100 else text_b,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "hybrid_score": hybrid_score,
            "alpha": alpha,
            "lexical_method": lexical_method,
            "method_scores": method_scores,
        }

    def get_recommended_alpha(
        self, scores: list[float], labels: list[int], method: str = "f1"
    ) -> float:
        """
        Find optimal alpha value for a dataset.

        Args:
            scores: List of similarity scores
            labels: List of ground truth labels (1=plagiarism, 0=not)
            method: Optimization method ('f1', 'roc')

        Returns:
            Optimal alpha value
        """
        alphas = np.arange(0.0, 1.05, 0.05)
        best_alpha = 0.7
        best_score = 0.0

        for alpha in alphas:
            # Combine scores
            # This assumes scores are from both semantic and lexical
            # In practice, you'd need both sets of scores
            pass

        return best_alpha

    def get_stats(self) -> dict[str, Any]:
        """Get scoring statistics."""
        return {
            **self._stats,
            "config": {
                "alpha": self.config.alpha,
                "lexical_method": self.config.lexical_method,
                "use_stopwords": self.config.use_stopwords,
            },
            "cache_size": len(self._lexical_cache),
        }

    def clear_cache(self) -> None:
        """Clear lexical cache."""
        self._lexical_cache.clear()
        self._tfidf_vectorizer = None


def compute_hybrid_plagiarism_flags(
    similarity_df: pd.DataFrame,
    lexical_scores: Optional[pd.DataFrame] = None,
    texts: Optional[dict[str, str]] = None,
    alpha: float = 0.7,
    threshold: float = 0.59,
    lexical_method: str = "tfidf",
    normalize: Optional[str] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Compute hybrid plagiarism flags.

    Args:
        similarity_df: Semantic similarity DataFrame
        lexical_scores: Pre-computed lexical scores (optional)
        texts: Document texts (required if lexical_scores not provided)
        alpha: Semantic weight
        threshold: Flagging threshold
        lexical_method: Lexical method
        normalize: Score normalization method ('minmax', 'zscore', or None)

    Returns:
        Tuple of (hybrid_df, flagged_pairs)
    """
    from src.core.similarity import normalize_scores

    scorer = HybridScorer(
        HybridConfig(alpha=alpha, lexical_method=lexical_method, normalize=normalize)
    )

    if lexical_scores is not None:
        sem_norm = normalize_scores(similarity_df, method=normalize)
        lex_norm = normalize_scores(lexical_scores, method=normalize)
        hybrid_df = alpha * sem_norm + (1 - alpha) * lex_norm
        if normalize in (None, "none", "minmax", "min-max"):
            hybrid_df = np.clip(hybrid_df, 0.0, 1.0)
    elif texts is not None:
        hybrid_df = scorer.compute_hybrid_matrix(
            texts, similarity_df, alpha, lexical_method, normalize
        )
    else:
        raise ValueError("Either lexical_scores or texts must be provided")

    flagged = scorer.flag_plagiarism_hybrid(hybrid_df, threshold)

    return hybrid_df, flagged

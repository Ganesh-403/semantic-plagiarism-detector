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

"""lexical_similarity.py
---------------------
Computes lexical similarity between documents using TF-IDF vectorization, Jaccard similarity,
Dice coefficient, overlap coefficient, and n-gram overlap.

This module provides a TF-IDF based baseline and set-theoretic similarity metrics
for plagiarism detection, which excel at identifying identical lexical copy-pasting.

Mathematical Formulas:
    - Jaccard Similarity: J(A, B) = |A ∩ B| / |A ∪ B|
    - Sørensen-Dice Coefficient: Dice(A, B) = (2 * |A ∩ B|) / (|A| + |B|)
    - Overlap Coefficient: Overlap(A, B) = |A ∩ B| / min(|A|, |B|)
    - TF-IDF Cosine Similarity: Cosine(u, v) = (u · v) / (||u|| * ||v||)
"""

from __future__ import annotations

import functools
import hashlib
import logging
import re

logger = logging.getLogger(__name__)
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

from src.core.stopwords import get_stopword_manager, tokenize_filtered

# ── Stop-word handling (issue #222) ───────────────────────────────────────────

_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Compact fallback list — covers high-frequency English function words.
_FALLBACK_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "of",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "them",
    "his",
    "her",
    "their",
    "our",
    "my",
    "your",
    "me",
    "him",
    "us",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "just",
    "also",
    "not",
    "no",
    "nor",
    "only",
    "own",
    "same",
    "such",
    "more",
    "most",
    "other",
    "some",
    "any",
    "each",
    "few",
    "both",
    "all",
    "there",
    "here",
    "where",
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
}


def _load_stopwords() -> set[str]:
    """Resolve the English stop-word set from NLTK corpus with fallback.

    Returns
    -------
    Set[str]
        Set of lower-case English stop-words.
    """
    try:
        from nltk.corpus import stopwords as _nltk_stopwords  # type: ignore

        return set(_nltk_stopwords.words("english"))
    except Exception:
        return set(_FALLBACK_STOPWORDS)


#: Module-level stop-word set resolved once at import.
STOPWORDS: set[str] = _load_stopwords()


def _get_combined_stopwords(
    custom_stopwords: Optional[Iterable[str]] = None,
) -> set[str]:
    """Combine base module STOPWORDS with optional custom domain stop-words.

    Parameters
    ----------
    custom_stopwords : Optional[Iterable[str]], default=None
        Optional list or set of custom domain-specific stop-words.

    Returns
    -------
    Set[str]
        Combined set of normalized lower-case stop-words.
    """
    combined: set[str] = set(STOPWORDS)
    if custom_stopwords:
        combined.update(
            w.lower().strip() for w in custom_stopwords if isinstance(w, str)
        )
    return combined


@functools.lru_cache(maxsize=4096)
def _get_base_tokens(text: str) -> list[str]:
    """Extract and cache lower-cased base tokens from text, using nltk if available, or regex fallback."""
    if not text or not isinstance(text, str):
        return []

    try:
        from nltk.tokenize import word_tokenize as _nltk_word_tokenize

        raw_tokens = _nltk_word_tokenize(text)
        return [
            tok.lower()
            for tok in raw_tokens
            if tok.isalnum() or (any(c.isalnum() for c in tok) and "'" in tok)
        ]
    except Exception:
        return _TOKEN_RE.findall(text.lower())


@functools.lru_cache(maxsize=4096)
def _tokenize_cached(text: str, stopwords_frozenset: frozenset[str]) -> list[str]:
    """Helper to tokenize and filter stopwords using cached base tokens."""
    base_tokens = _get_base_tokens(text)
    return [tok for tok in base_tokens if tok not in stopwords_frozenset]


def remove_stopwords(
    text: str,
    stopwords: Optional[Iterable[str]] = None,
) -> str:
    """Filter out stop-words from input text while preserving token order.

    Parameters
    ----------
    text : str
        Input document text string.
    stopwords : Optional[Iterable[str]], default=None
        Optional iterable of stop-words to exclude. Defaults to module STOPWORDS.

    Returns
    -------
    str
        Space-separated string of non-stop-word tokens.
    """
    if not text or not isinstance(text, str):
        return ""
    stop_set = frozenset(stopwords) if stopwords is not None else frozenset(STOPWORDS)
    tokens = _tokenize_cached(text, stop_set)
    return " ".join(tokens)


def tokenize(
    text: str,
    stopwords: Optional[Iterable[str]] = None,
) -> set[str]:
    """Tokenize input text into a set of lower-cased non-stop-word tokens.

    Parameters
    ----------
    text : str
        Input text to tokenize.
    stopwords : Optional[Iterable[str]], default=None
        Optional iterable of stop-words to exclude. Defaults to module STOPWORDS.

    Returns
    -------
    Set[str]
        Set of unique lower-case non-stop-word token strings.
    """
    if not text or not isinstance(text, str):
        return set()
    stop_set = frozenset(stopwords) if stopwords is not None else frozenset(STOPWORDS)
    return set(_tokenize_cached(text, stop_set))


def get_ngrams(
    text: str,
    n: int = 3,
    stopwords: Optional[Iterable[str]] = None,
) -> set[tuple[str, ...]]:
    """Extract word-level n-grams from text after stop-word filtering.

    Mathematical Formula
    --------------------
    An n-gram G_i = (w_i, w_{i+1}, ..., w_{i+n-1}) is a sequence of n contiguous words.

    Parameters
    ----------
    text : str
        Input document text string.
    n : int, default=3
        N-gram sequence length (number of contiguous tokens).
    stopwords : Optional[Iterable[str]], default=None
        Optional iterable of stop-words to filter out prior to n-gram extraction.

    Returns
    -------
    Set[Tuple[str, ...]]
        Set of n-gram tuples extracted from the filtered token sequence.
    """
    if not text or not isinstance(text, str) or n < 1:
        return set()
    stop_set = frozenset(stopwords) if stopwords is not None else frozenset(STOPWORDS)
    tokens = _tokenize_cached(text, stop_set)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def n_gram_overlap(
    text_a: str,
    text_b: str,
    n: int = 3,
    stopwords: Optional[Iterable[str]] = None,
) -> float:
    """Compute word-level n-gram overlap Jaccard coefficient between two texts.

    Mathematical Formula
    --------------------
    .. math::

        Overlap_N(A, B) = \\frac{|G_n(A) \\cap G_n(B)|}{|G_n(A) \\cup G_n(B)|}

    where :math:`G_n(X)` represents the set of n-gram tuples for text X.

    Parameters
    ----------
    text_a : str
        First document text string.
    text_b : str
        Second document text string.
    n : int, default=3
        Order of n-grams to extract.
    stopwords : Optional[Iterable[str]], default=None
        Optional iterable of stop-words to filter out.

    Returns
    -------
    float
        Overlap score bounded between 0.0 and 1.0.
    
    Raises
    ------
    ValueError
        If n is not between 1 and 10 inclusive.
    """
    if not (1 <= n <= 10):
        raise ValueError(f"n must be between 1 and 10, got {n}")

    ngrams_a: set[tuple[str, ...]] = get_ngrams(text_a, n=n, stopwords=stopwords)
    ngrams_b: set[tuple[str, ...]] = get_ngrams(text_b, n=n, stopwords=stopwords)

    if not ngrams_a and not ngrams_b:
        return 0.0
    union: set[tuple[str, ...]] = ngrams_a | ngrams_b
    if not union:
        return 0.0
    return float(len(ngrams_a & ngrams_b) / len(union))


def jaccard_similarity(
    text_a: str,
    text_b: str,
    stopwords: Optional[set[str]] = None,
    use_stopwords: bool = True,
) -> float:
    """Compute Jaccard similarity with optional stopword filtering."""
    if not text_a or not text_b:
        return 0.0

    if use_stopwords:
        if stopwords is None:
            stopwords = get_stopword_manager().get_stopwords()
        set_a = tokenize_filtered(text_a, stopwords)
        set_b = tokenize_filtered(text_b, stopwords)
    else:
        from src.core.lexical_similarity import tokenize

        set_a = tokenize(text_a, stopwords)
        set_b = tokenize(text_b, stopwords)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def jaccard_index(
    text_a: str,
    text_b: str,
    stopwords: Optional[Iterable[str]] = None,
) -> float:
    """Alias helper for jaccard_similarity.

    Parameters
    ----------
    text_a : str
        First document text string.
    text_b : str
        Second document text string.
    stopwords : Optional[Iterable[str]], default=None
        Optional stop-word set.

    Returns
    -------
    float
        Jaccard similarity index between 0.0 and 1.0.
    """
    return jaccard_similarity(text_a, text_b, stopwords=stopwords)


def dice_coefficient(
    text_a: str,
    text_b: str,
    stopwords: Optional[Iterable[str]] = None,
) -> float:
    """Compute Sørensen-Dice coefficient similarity score between two texts.

    Mathematical Formula
    --------------------
    .. math::

        Dice(A, B) = \\frac{2 \\cdot |A \\cap B|}{|A| + |B|}

    where A and B are unique token sets extracted from text_a and text_b.

    Parameters
    ----------
    text_a : str
        First document text.
    text_b : str
        Second document text.
    stopwords : Optional[Iterable[str]], default=None
        Optional stop-words filter.

    Returns
    -------
    float
        Sørensen-Dice coefficient score bounded between 0.0 and 1.0.
    """
    set_a: set[str] = tokenize(text_a, stopwords=stopwords)
    set_b: set[str] = tokenize(text_b, stopwords=stopwords)
    total_len = len(set_a) + len(set_b)
    if total_len == 0:
        return 0.0
    intersection_len = len(set_a & set_b)
    return float((2.0 * intersection_len) / total_len)


def overlap_coefficient(
    text_a: str,
    text_b: str,
    stopwords: Optional[Iterable[str]] = None,
) -> float:
    """Compute Szymkiewicz-Simpson Overlap coefficient between two texts.

    Mathematical Formula
    --------------------
    .. math::

        Overlap(A, B) = \\frac{|A \\cap B|}{\\min(|A|, |B|)}

    where A and B are token sets. Evaluates subset inclusion robustly against length disparity.

    Parameters
    ----------
    text_a : str
        First document text.
    text_b : str
        Second document text.
    stopwords : Optional[Iterable[str]], default=None
        Optional stop-words filter.

    Returns
    -------
    float
        Overlap coefficient score bounded between 0.0 and 1.0.
    """
    set_a: set[str] = tokenize(text_a, stopwords=stopwords)
    set_b: set[str] = tokenize(text_b, stopwords=stopwords)
    min_len = min(len(set_a), len(set_b))
    if min_len == 0:
        return 0.0
    return float(len(set_a & set_b) / min_len)


def calculate_lexical_similarity(
    text_a: str,
    text_b: str,
    custom_stopwords: Optional[set[str]] = None,
) -> float:
    """Calculate lexical similarity between two text strings using TF-IDF cosine similarity.

    Mathematical Formula
    --------------------
    .. math::

        Cosine(u, v) = \\frac{u \\cdot v}{||u||_2 \\cdot ||v||_2}

    where u and v are L2-normalized TF-IDF vector representations of text_a and text_b.

    Parameters
    ----------
    text_a : str
        First document text string.
    text_b : str
        Second document text string.
    custom_stopwords : Optional[Set[str]], default=None
        Optional set of custom stop-words (e.g. 'ibid', 'figure') to merge with default stop-words.

    Returns
    -------
    float
        Cosine similarity score bounded between 0.0 and 1.0.
    """
    if not text_a or not text_b or not text_a.strip() or not text_b.strip():
        return 0.0

    stop_words_list: list[str] = list(_get_combined_stopwords(custom_stopwords))

    try:
        vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except ValueError:
        # Handles case where documents contain only stop-words or empty vocabulary
        return 0.0


def compute_tfidf_lexical_similarity(
    doc_a: str,
    doc_b: str,
    corpus: list[str],
) -> float:
    """Compute TF-IDF weighted lexical similarity between doc_a and doc_b across corpus vocabulary.

    Parameters
    ----------
    doc_a : str
        First document text string.
    doc_b : str
        Second document text string.
    corpus : list[str]
        Corpus document texts used to compute term frequencies and inverse document frequencies.

    Returns
    -------
    float
        Normalized similarity score bounded strictly between 0.0 and 1.0.
    """
    if (
        not doc_a
        or not doc_b
        or not isinstance(doc_a, str)
        or not isinstance(doc_b, str)
    ):
        return 0.0
    if not doc_a.strip() or not doc_b.strip():
        return 0.0

    combined_corpus = list(corpus) if corpus else []
    if doc_a not in combined_corpus:
        combined_corpus.append(doc_a)
    if doc_b not in combined_corpus:
        combined_corpus.append(doc_b)

    try:
        vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
        vectorizer.fit(combined_corpus)
        matrix = vectorizer.transform([doc_a, doc_b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except ValueError:
        return 0.0


def _make_documents_hash(
    documents: dict[str, str],
    custom_stopwords: Optional[set[str]] = None,
) -> str:
    """Create a deterministic SHA-256 hash for caching similarity matrix computations.

    Parameters
    ----------
    documents : Dict[str, str]
        Dictionary mapping document titles to text content.
    custom_stopwords : Optional[Set[str]], default=None
        Custom stop-words set.

    Returns
    -------
    str
        Hexadecimal SHA-256 hash string digest.
    """
    sorted_items: list[tuple[str, str]] = sorted(documents.items())
    sorted_custom: list[str] = sorted(custom_stopwords) if custom_stopwords else []
    hash_input: bytes = str((sorted_items, sorted_custom)).encode("utf-8")
    return hashlib.sha256(hash_input).hexdigest()


@functools.lru_cache(maxsize=32)
def _cached_lexical_similarity_matrix(
    documents_hash: str,
    documents_tuple: tuple[tuple[str, str], ...],
    custom_stopwords_tuple: tuple[str, ...],
) -> pd.DataFrame:
    """LRU-cached implementation of TF-IDF similarity matrix calculation.

    Parameters
    ----------
    documents_hash : str
        Deterministic SHA-256 hash of the input corpus.
    documents_tuple : Tuple[Tuple[str, str], ...]
        Immutable tuple representation of (doc_name, doc_text) pairs.
    custom_stopwords_tuple : Tuple[str, ...]
        Immutable tuple of custom stop-words.

    Returns
    -------
    pd.DataFrame
        N x N DataFrame matrix of similarity scores indexed by document titles.
    """
    documents: dict[str, str] = dict(documents_tuple)
    doc_names: list[str] = list(documents.keys())
    n: int = len(doc_names)

    if n == 0:
        return pd.DataFrame()

    texts: list[str] = [documents[name] for name in doc_names]
    stop_words_list: list[str] = list(
        _get_combined_stopwords(
            set(custom_stopwords_tuple) if custom_stopwords_tuple else None
        )
    )

    try:
        vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    except ValueError:
        sim_matrix = np.zeros((n, n), dtype=float)

    return pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)


def lexical_similarity_matrix(
    documents: dict[str, str],
    use_cache: bool = True,
    custom_stopwords: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build an N x N TF-IDF cosine similarity matrix across all document pairs in a corpus.

    Mathematical Formula
    --------------------
    For each pair of documents :math:`d_i` and :math:`d_j`:

    .. math::

        M_{i,j} = \\frac{\\mathbf{v}_i \\cdot \\mathbf{v}_j}{\\|\\mathbf{v}_i\\|_2 \\|\\mathbf{v}_j\\|_2}

    where :math:`\\mathbf{v}_i` is the TF-IDF vector of document :math:`d_i`.

    Parameters
    ----------
    documents : Dict[str, str]
        Dictionary mapping document filenames/identifiers to document text strings.
    use_cache : bool, default=True
        Whether to leverage LRU cache for identical corpus and stop-word inputs.
    custom_stopwords : Optional[Set[str]], default=None
        Optional set of domain-specific stop-words to exclude.

    Returns
    -------
    pd.DataFrame
        Square N x N pandas DataFrame containing similarity scores in range [0.0, 1.0].
    """
    custom_tuple: tuple[str, ...] = (
        tuple(sorted(custom_stopwords)) if custom_stopwords else ()
    )

    if use_cache:
        documents_tuple: tuple[tuple[str, str], ...] = tuple(sorted(documents.items()))
        documents_hash: str = _make_documents_hash(documents, custom_stopwords)
        return _cached_lexical_similarity_matrix(
            documents_hash, documents_tuple, custom_tuple
        )

    doc_names: list[str] = list(documents.keys())
    n: int = len(doc_names)

    if n == 0:
        return pd.DataFrame()

    texts: list[str] = [documents[name] for name in doc_names]
    stop_words_list: list[str] = list(_get_combined_stopwords(custom_stopwords))

    try:
        vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    except ValueError:
        sim_matrix = np.zeros((n, n), dtype=float)

    return pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)


# ── Soft-Max / Sigmoidal Normalization (#924) ──────────────────────────────────


def scale_lexical_score(
    score: float,
    steepness: float = 6.0,
    midpoint: float = 0.5,
) -> float:
    """Apply non-linear sigmoid/softmax normalization to lexical similarity scores.

    Raw Jaccard and Levenshtein similarity scores exhibit a linear distribution,
    which often causes mild word overlaps to appear overly severe. This function
    applies a tuned logistic sigmoid curve normalized such that:
    - Input 0.0 maps strictly to 0.0
    - Input 0.5 maps strictly to 0.5
    - Input 1.0 maps strictly to 1.0
    - Intermediate scores are smoothly suppressed in low ranges and enhanced in high ranges.

    Mathematical Formula
    --------------------
    .. math::

        \\sigma(s) = \\frac{1}{1 + e^{-k (s - m)}}

        f(s) = \\frac{\\sigma(s) - \\sigma(0)}{\\sigma(1) - \\sigma(0)}

    Parameters
    ----------
    score : float
        Raw lexical similarity score, typically in range [0.0, 1.0].
    steepness : float, default=6.0
        Logistic curve steepness parameter (k).
    midpoint : float, default=0.5
        Inflection point parameter (m).

    Returns
    -------
    float
        Scaled lexical similarity score strictly bounded in [0.0, 1.0].
    """
    try:
        val = float(score)
    except (TypeError, ValueError):
        return 0.0

    if np.isnan(val) or np.isinf(val):
        return 0.0

    if val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0

    def _raw_sig(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))

    sig_val = _raw_sig(val)
    sig_min = _raw_sig(0.0)
    sig_max = _raw_sig(1.0)

    if sig_max == sig_min:
        return float(np.clip(val, 0.0, 1.0))

    scaled = (sig_val - sig_min) / (sig_max - sig_min)
    return float(np.clip(scaled, 0.0, 1.0))


def softmax_normalize_scores(
    scores: Iterable[float] | np.ndarray,
    steepness: float = 6.0,
    midpoint: float = 0.5,
) -> np.ndarray:
    """Normalize a vector or array of lexical similarity scores using sigmoidal softmax scaling.

    Parameters
    ----------
    scores : Iterable[float] | np.ndarray
        Array or iterable of raw similarity scores.
    steepness : float, default=6.0
        Logistic curve steepness parameter.
    midpoint : float, default=0.5
        Inflection point parameter.

    Returns
    -------
    np.ndarray
        NumPy array of scaled scores bounded in [0.0, 1.0].
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return np.empty_like(arr, dtype=float)

    vectorized_scale = np.vectorize(
        lambda s: scale_lexical_score(s, steepness=steepness, midpoint=midpoint)
    )
    return vectorized_scale(arr)


def scale_lexical_matrix(
    matrix: pd.DataFrame | np.ndarray,
    steepness: float = 6.0,
    midpoint: float = 0.5,
) -> pd.DataFrame | np.ndarray:
    """Apply sigmoid/softmax scaling across a full similarity matrix or DataFrame.

    Parameters
    ----------
    matrix : pd.DataFrame | np.ndarray
        Similarity matrix or DataFrame.
    steepness : float, default=6.0
        Logistic curve steepness parameter.
    midpoint : float, default=0.5
        Inflection point parameter.

    Returns
    -------
    pd.DataFrame | np.ndarray
        Scaled matrix or DataFrame preserving input structure and column headers.
    """
    if isinstance(matrix, pd.DataFrame):
        scaled_vals = softmax_normalize_scores(
            matrix.values, steepness=steepness, midpoint=midpoint
        )
        return pd.DataFrame(scaled_vals, index=matrix.index, columns=matrix.columns)
    return softmax_normalize_scores(matrix, steepness=steepness, midpoint=midpoint)


def compute_vectorized_jaccard_matrix(documents: list[str]) -> np.ndarray:
    """Compute pairwise Jaccard similarity for a list of documents using vectorized operations.

    This function leverages SciPy sparse matrices to compute the Jaccard similarity
    matrix significantly faster than naive Python set intersections, which is
    essential for processing thousands of documents.

    Parameters
    ----------
    documents : list[str]
        A list of document texts.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, N) where N is the number of documents,
        representing the pairwise Jaccard similarity scores.
    """
    if not documents:
        return np.array([])
    
    # Vectorize documents into binary bag-of-words (Boolean arrays)
    vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
    try:
        X = vectorizer.fit_transform(documents)
    except ValueError:
        # Occurs if documents are entirely empty
        N = len(documents)
        return np.zeros((N, N))

    # X is shape (N, V). Intersection is dot product of binary matrices
    intersection = X.dot(X.T).toarray()
    
    # Sum across columns to get the number of unique words per document
    row_sums = X.sum(axis=1).A.flatten()
    
    # Union = |A| + |B| - |A ∩ B|
    # row_sums[:, None] creates a column vector, row_sums[None, :] a row vector
    union = row_sums[:, None] + row_sums[None, :] - intersection
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_matrix = np.where(union != 0, intersection / union, 0.0)
        
    return jaccard_matrix


def compute_char_ngram_similarity(text_a: str, text_b: str, n: int = 5) -> float:
    """Compute character-level sliding n-gram Jaccard similarity between two texts.

    Word-level Jaccard similarity misses obfuscations where words are misspelled,
    hyphenated, or slightly altered (e.g., OCR text typos). Character-level n-gram overlap (shingling)
    detects sub-word plagiarism by comparing sequences of `n` consecutive characters.

    Mathematical Formula
    --------------------
    .. math::

        J_{char}(A, B) = \frac{|N_n(A) \cap N_n(B)|}{|N_n(A) \cup N_n(B)|}

    where :math:`N_n(X)` represents the set of unique character n-grams for text X.
    Both texts are converted to lowercase and stripped leading/trailing whitespace
    before n-gram extraction to ensure case-insensitive comparison.

    Parameters
    ----------
    text_a : str
        First document text string.
    text_b : str
        Second document text string.
    n : int, default=5
        Length of the character sliding window (n-gram size). Must be >= 1.
        A value of 5 is recommended for detecting paraphrased or slightly
        obfuscated academic text.

    Returns
    -------
    float
        Jaccard similarity index bounded between 0.0 and 1.0.
        Returns 0.0 if either text is empty, None, or shorter than `n` characters
        after preprocessing.

    Examples
    --------
    >>> compute_char_ngram_similarity("plagiarism", "plagiarism", n=5)
    1.0
    >>> compute_char_ngram_similarity("plagiarism", "plagarism", n=5)
    0.75
    >>> compute_char_ngram_similarity("hello world", "goodbye moon", n=3)
    0.0
    """
    # Validate inputs and handle edge cases gracefully
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        logger.debug(
            "compute_char_ngram_similarity: non-string input provided, returning 0.0"
        )
        return 0.0

    if not text_a or not text_b:
        return 0.0

    if n < 1:
        logger.warning(
            "compute_char_ngram_similarity: n must be >= 1, received %d. Defaulting to 5.",
            n,
        )
        n = 5

    # Preprocess texts: lowercase and strip whitespace for consistent comparison
    processed_a = text_a.lower().strip()
    processed_b = text_b.lower().strip()

    # If either text is shorter than n after preprocessing, no n-grams can be formed
    if len(processed_a) < n or len(processed_b) < n:
        return 0.0

    # Extract sliding character n-grams using set comprehension for O(1) lookups
    # A sliding window of size n moves one character at a time across the string
    ngrams_a = {processed_a[i : i + n] for i in range(len(processed_a) - n + 1)}
    ngrams_b = {processed_b[i : i + n] for i in range(len(processed_b) - n + 1)}

    # Calculate Jaccard index: intersection over union
    intersection_len = len(ngrams_a & ngrams_b)
    union_len = len(ngrams_a | ngrams_b)

    if union_len == 0:
        return 0.0

    similarity = float(intersection_len / union_len)

    logger.debug(
        "compute_char_ngram_similarity: computed char %d-gram similarity=%.4f "
        "(intersection=%d, union=%d)",
        n,
        similarity,
        intersection_len,
        union_len,
    )

    return similarity

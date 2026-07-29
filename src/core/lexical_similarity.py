"""
lexical_similarity.py
---------------------
Computes lexical similarity between documents using TF-IDF vectorization.

This module provides a TF-IDF based baseline for plagiarism detection,
which excels at identifying identical lexical copy-pasting.

Issue #222: Stop-words (the, and, is, a, …) are filtered out before the
TF-IDF set intersection is computed, so common function words cannot
artificially inflate the lexical similarity between unrelated essays.
Filtering is applied both in the TF-IDF vectorizer (via ``stop_words``)
and in the standalone ``jaccard_similarity`` / ``remove_stopwords``
helpers, so any Jaccard-style fallback comparison benefits from the same
filtering.

Issue #845: Supports custom_stopwords parameter to extend stop-word sets
for domain-specific academic filler words (e.g. "ibid", "figure", "table").
"""

import functools
import hashlib
import re
from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Stop-word handling (issue #222) ───────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Compact fallback list — covers high-frequency English function words.
_FALLBACK_STOPWORDS: Set[str] = {
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


def _load_stopwords() -> Set[str]:
    """Resolve the English stop-word set."""
    try:
        from nltk.corpus import stopwords as _nltk_stopwords  # type: ignore

        return set(_nltk_stopwords.words("english"))
    except Exception:
        return set(_FALLBACK_STOPWORDS)


#: Module-level stop-word set resolved once at import.
STOPWORDS: Set[str] = _load_stopwords()


def _get_combined_stopwords(
    custom_stopwords: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Combine base STOPWORDS with optional custom stop-words."""
    combined = set(STOPWORDS)
    if custom_stopwords:
        combined.update(w.lower() for w in custom_stopwords)
    return combined


def remove_stopwords(text: str, stopwords: Optional[Iterable[str]] = None) -> str:
    """Return ``text`` with stop-words removed (case-insensitive)."""
    if not text:
        return ""
    stop_set = set(stopwords) if stopwords is not None else STOPWORDS
    tokens = [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in stop_set]
    return " ".join(tokens)


def tokenize(text: str, stopwords: Optional[Iterable[str]] = None) -> Set[str]:
    """Tokenize ``text`` into a set of lower-cased non-stop-word tokens."""
    if not text:
        return set()
    stop_set = set(stopwords) if stopwords is not None else STOPWORDS
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in stop_set}


def jaccard_similarity(
    text_a: str, text_b: str, stopwords: Optional[Iterable[str]] = None
) -> float:
    """Jaccard similarity over stop-word-filtered token sets."""
    set_a = tokenize(text_a, stopwords=stopwords)
    set_b = tokenize(text_b, stopwords=stopwords)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def calculate_lexical_similarity(
    text_a: str,
    text_b: str,
    custom_stopwords: Optional[Set[str]] = None,
) -> float:
    """
    Calculate lexical similarity between two text strings using TF-IDF cosine similarity.

    Args:
        text_a: First document text.
        text_b: Second document text.
        custom_stopwords: Optional set of custom stop-words (e.g. "ibid", "figure")
            to merge with default NLTK stop-words.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    stop_words_list = list(_get_combined_stopwords(custom_stopwords))

    try:
        vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except ValueError:
        # Handles case where documents contain only stop-words or empty vocabulary
        return 0.0


def _make_documents_hash(
    documents: Dict[str, str],
    custom_stopwords: Optional[Set[str]] = None,
) -> str:
    """Create a stable hash from document contents and custom stopwords for caching."""
    sorted_items = sorted(documents.items())
    sorted_custom = sorted(custom_stopwords) if custom_stopwords else []
    hash_input = str((sorted_items, sorted_custom)).encode("utf-8")
    return hashlib.sha256(hash_input).hexdigest()


@functools.lru_cache(maxsize=32)
def _cached_lexical_similarity_matrix(
    documents_hash: str,
    documents_tuple: tuple,
    custom_stopwords_tuple: tuple,
) -> pd.DataFrame:
    """Internal cached implementation of lexical similarity matrix."""
    documents = dict(documents_tuple)
    doc_names = list(documents.keys())
    n = len(doc_names)

    if n == 0:
        return pd.DataFrame()

    texts = [documents[name] for name in doc_names]
    stop_words_list = list(
        _get_combined_stopwords(set(custom_stopwords_tuple) if custom_stopwords_tuple else None)
    )

    try:
        vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    except ValueError:
        sim_matrix = np.zeros((n, n))

    return pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)


def lexical_similarity_matrix(
    documents: Dict[str, str],
    use_cache: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Build an N×N TF-IDF cosine similarity matrix between all document pairs."""
    custom_tuple = tuple(sorted(custom_stopwords)) if custom_stopwords else ()

    if use_cache:
        documents_tuple = tuple(sorted(documents.items()))
        documents_hash = _make_documents_hash(documents, custom_stopwords)
        return _cached_lexical_similarity_matrix(
            documents_hash, documents_tuple, custom_tuple
        )
    else:
        doc_names = list(documents.keys())
        n = len(doc_names)

        if n == 0:
            return pd.DataFrame()

        texts = [documents[name] for name in doc_names]
        stop_words_list = list(_get_combined_stopwords(custom_stopwords))

        try:
            vectorizer = TfidfVectorizer(stop_words=stop_words_list)
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
            sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
        except ValueError:
            sim_matrix = np.zeros((n, n))

        return pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
    
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
Stopword filtering module for lexical similarity.

Provides configurable stopword lists and filtering functions
to reduce false positives in lexical similarity matching.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT STOPWORD LISTS
# ============================================================================

# English stopwords - comprehensive list
ENGLISH_STOPWORDS = {
    # Articles
    "a",
    "an",
    "the",
    # Conjunctions
    "and",
    "or",
    "but",
    "for",
    "nor",
    "yet",
    "so",
    "both",
    "either",
    "neither",
    "whether",
    # Prepositions
    "of",
    "to",
    "in",
    "for",
    "on",
    "at",
    "from",
    "by",
    "with",
    "without",
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
    "up",
    "down",
    "off",
    "over",
    "under",
    "upon",
    "across",
    "among",
    "throughout",
    "toward",
    "within",
    "without",
    # Pronouns
    "i",
    "me",
    "my",
    "myself",
    "we",
    "us",
    "our",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "who",
    "whom",
    "whose",
    "which",
    "that",
    "this",
    "these",
    "those",
    # Auxiliary verbs
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
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
    # Common adverbs
    "very",
    "too",
    "so",
    "than",
    "then",
    "now",
    "just",
    "only",
    "also",
    "too",
    "very",
    "just",
    # Other common words
    "not",
    "no",
    "nor",
    "never",
    "none",
    "all",
    "each",
    "every",
    "few",
    "many",
    "more",
    "most",
    "other",
    "some",
    "any",
    "such",
    "there",
    "here",
    "where",
    "why",
    "how",
    "what",
    "when",
    "which",
    "who",
    "whom",
}

# Domain-specific stopwords (academic/plagiarism context)
ACADEMIC_STOPWORDS = {
    "figure",
    "table",
    "equation",
    "section",
    "chapter",
    "appendix",
    "reference",
    "bibliography",
    "citation",
    "abstract",
    "introduction",
    "conclusion",
    "discussion",
    "methodology",
    "results",
    "analysis",
    "finding",
    "study",
    "research",
    "paper",
    "article",
    "journal",
    "author",
    "year",
    "volume",
    "issue",
    "page",
    "doi",
    "url",
    "http",
    "https",
    "www",
    "et",
    "al",
    "ibid",
    "op",
    "cit",
    "see",
    "also",
    "e.g.",
    "i.e.",
    "etc",
    "fig",
    "tab",
    "eq",
    "sec",
}

# Custom stopwords - can be extended by users
CUSTOM_STOPWORDS: set[str] = set()


# ============================================================================
# STOPWORD MANAGER
# ============================================================================


class StopwordManager:
    """
    Manages stopword lists with filtering and customization options.

    Features:
    - Multiple stopword lists (English, Academic, Custom)
    - Configurable filtering (enable/disable specific lists)
    - Custom stopword addition/removal
    - Load/Save custom stopwords to file
    """

    def __init__(self):
        self.english = set(ENGLISH_STOPWORDS)
        self.academic = set(ACADEMIC_STOPWORDS)
        self.custom = set(CUSTOM_STOPWORDS)
        self._combined_cache: Optional[set[str]] = None
        self._enabled = {
            "english": True,
            "academic": True,
            "custom": True,
        }
        self._load_custom_from_file()

    def _load_custom_from_file(self) -> None:
        """Load custom stopwords from file if exists."""
        try:
            custom_path = Path("data/custom_stopwords.json")
            if custom_path.exists():
                with open(custom_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.custom = set(data)
                        logger.info(
                            f"Loaded {len(self.custom)} custom stopwords from file"
                        )
        except Exception as e:
            logger.warning(f"Failed to load custom stopwords: {e}")

    def save_custom_stopwords(self) -> bool:
        """Save custom stopwords to file."""
        try:
            custom_path = Path("data/custom_stopwords.json")
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            with open(custom_path, "w") as f:
                json.dump(list(self.custom), f, indent=2)
            logger.info(f"Saved {len(self.custom)} custom stopwords to file")
            return True
        except Exception as e:
            logger.error(f"Failed to save custom stopwords: {e}")
            return False

    def get_stopwords(self) -> set[str]:
        """Get combined stopword set based on enabled lists."""
        if self._combined_cache is not None:
            return self._combined_cache

        combined = set()
        if self._enabled.get("english", True):
            combined.update(self.english)
        if self._enabled.get("academic", True):
            combined.update(self.academic)
        if self._enabled.get("custom", True):
            combined.update(self.custom)

        self._combined_cache = combined
        return combined

    def enable_list(self, list_name: str) -> None:
        """Enable a stopword list."""
        if list_name in self._enabled:
            self._enabled[list_name] = True
            self._clear_cache()

    def disable_list(self, list_name: str) -> None:
        """Disable a stopword list."""
        if list_name in self._enabled:
            self._enabled[list_name] = False
            self._clear_cache()

    def add_stopword(self, word: str) -> None:
        """Add a custom stopword."""
        word = word.lower().strip()
        if word:
            self.custom.add(word)
            self._clear_cache()
            self.save_custom_stopwords()

    def remove_stopword(self, word: str) -> bool:
        """Remove a custom stopword."""
        word = word.lower().strip()
        if word in self.custom:
            self.custom.remove(word)
            self._clear_cache()
            self.save_custom_stopwords()
            return True
        return False

    def add_custom_list(self, words: list[str]) -> None:
        """Add multiple custom stopwords."""
        added = 0
        for word in words:
            word = word.lower().strip()
            if word and word not in self.custom:
                self.custom.add(word)
                added += 1
        if added > 0:
            self._clear_cache()
            self.save_custom_stopwords()
        return added

    def remove_custom_list(self, words: list[str]) -> int:
        """Remove multiple custom stopwords."""
        removed = 0
        for word in words:
            word = word.lower().strip()
            if word in self.custom:
                self.custom.remove(word)
                removed += 1
        if removed > 0:
            self._clear_cache()
            self.save_custom_stopwords()
        return removed

    def clear_custom(self) -> None:
        """Clear all custom stopwords."""
        self.custom.clear()
        self._clear_cache()
        self.save_custom_stopwords()

    def _clear_cache(self) -> None:
        """Clear the combined stopword cache."""
        self._combined_cache = None

    def get_stats(self) -> dict[str, Any]:
        """Get stopword statistics."""
        return {
            "english": len(self.english),
            "academic": len(self.academic),
            "custom": len(self.custom),
            "total": len(self.get_stopwords()),
            "enabled": self._enabled.copy(),
        }


# ============================================================================
# STOPWORD FILTERING FUNCTIONS
# ============================================================================


def filter_stopwords(
    text: str,
    stopwords: Optional[set[str]] = None,
    stopword_manager: Optional[StopwordManager] = None,
) -> str:
    """
    Filter stopwords from a text string.

    Args:
        text: Input text
        stopwords: Custom stopword set (overrides manager)
        stopword_manager: StopwordManager instance

    Returns:
        Text with stopwords removed
    """
    if not text or not text.strip():
        return text

    # Get stopwords
    if stopwords is not None:
        stop_set = stopwords
    elif stopword_manager is not None:
        stop_set = stopword_manager.get_stopwords()
    else:
        stop_set = ENGLISH_STOPWORDS

    # Tokenize and filter
    words = re.findall(r"\b[a-zA-Z0-9\']+\b", text.lower())
    filtered = [w for w in words if w not in stop_set]

    return " ".join(filtered)


def tokenize_filtered(
    text: str,
    stopwords: Optional[set[str]] = None,
    stopword_manager: Optional[StopwordManager] = None,
) -> set[str]:
    """
    Tokenize text and filter stopwords.

    Args:
        text: Input text
        stopwords: Custom stopword set
        stopword_manager: StopwordManager instance

    Returns:
        Set of filtered tokens
    """
    if not text or not text.strip():
        return set()

    # Get stopwords
    if stopwords is not None:
        stop_set = stopwords
    elif stopword_manager is not None:
        stop_set = stopword_manager.get_stopwords()
    else:
        stop_set = ENGLISH_STOPWORDS

    tokens = re.findall(r"\b[a-zA-Z0-9\']+\b", text.lower())
    return {t for t in tokens if t not in stop_set}


def filter_texts_batch(
    texts: list[str],
    stopword_manager: Optional[StopwordManager] = None,
) -> list[str]:
    """
    Filter stopwords from multiple texts.

    Args:
        texts: List of input texts
        stopword_manager: StopwordManager instance

    Returns:
        List of filtered texts
    """
    manager = stopword_manager or StopwordManager()
    stop_set = manager.get_stopwords()
    return [filter_stopwords(t, stop_set) for t in texts]


def get_token_overlap(
    text_a: str,
    text_b: str,
    stopword_manager: Optional[StopwordManager] = None,
) -> float:
    """
    Calculate token overlap after filtering stopwords.

    Returns:
        Overlap ratio (0-1)
    """
    tokens_a = tokenize_filtered(text_a, stopword_manager=stopword_manager)
    tokens_b = tokenize_filtered(text_b, stopword_manager=stopword_manager)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union > 0 else 0.0


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_stopword_manager: Optional[StopwordManager] = None


def get_stopword_manager() -> StopwordManager:
    """Get the global stopword manager instance."""
    global _stopword_manager
    if _stopword_manager is None:
        _stopword_manager = StopwordManager()
    return _stopword_manager


def get_stopwords() -> set[str]:
    """Get the global stopword set."""
    return get_stopword_manager().get_stopwords()

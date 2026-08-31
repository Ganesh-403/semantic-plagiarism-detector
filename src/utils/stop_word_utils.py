"""
Utility for filtering stop words in lexical similarity algorithms
Issue: #4017
"""


# Common English stop words
DEFAULT_STOP_WORDS = {
    "the", "and", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "to", "of", "in", "on", "at", "for", "with", "by",
    "it", "this", "that", "these", "those", "he", "she", "it",
    "we", "they", "i", "you", "not", "no", "yes", "or", "as"
}


def filter_stop_words(tokens, remove_stop_words=False):
    """
    Filters out stop words from a list of tokens.
    If remove_stop_words is False, returns the original list unchanged.
    """
    if not remove_stop_words:
        return tokens
    
    return [token for token in tokens if token.lower() not in DEFAULT_STOP_WORDS]


def get_stop_words():
    """
    Returns the set of default stop words used by the utility.
    """
    return DEFAULT_STOP_WORDS.copy()
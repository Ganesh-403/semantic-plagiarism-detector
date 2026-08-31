"""
Utility for case-insensitive search in PDF highlighting
Issue: #3973
"""


def find_case_insensitive(text: str, search_term: str):
    """
    Finds all indices of a search term in a text, ignoring case.
    Returns a list of start indices.
    """
    if not text or not search_term:
        return []
    
    lower_text = text.lower()
    lower_term = search_term.lower()
    indices = []
    start = 0
    
    while True:
        index = lower_text.find(lower_term, start)
        if index == -1:
            break
        indices.append(index)
        start = index + len(lower_term)
    
    return indices


def is_case_insensitive_match(text: str, search_term: str) -> bool:
    """
    Returns True if the search term exists in the text, ignoring case.
    """
    if not text or not search_term:
        return False
    return search_term.lower() in text.lower()
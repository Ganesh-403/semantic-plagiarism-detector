"""
Utility for handling unicode zero-width space joining in chunking
Issue: #4004
"""


def remove_zero_width_spaces(text: str) -> str:
    """
    Removes zero-width spaces (U+200B) and zero-width non-joiners (U+200C) from text.
    """
    if not isinstance(text, str):
        return ""
    return text.replace("\u200b", "").replace("\u200c", "")


def has_zero_width_spaces(text: str) -> bool:
    """
    Checks if a string contains zero-width spaces.
    """
    if not isinstance(text, str):
        return False
    return "\u200b" in text or "\u200c" in text
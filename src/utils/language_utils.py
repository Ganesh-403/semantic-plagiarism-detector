"""
Utility for validating ISO 639 language codes
Issue: #3985
"""

VALID_LANGUAGE_CODES = {
    "en", "fr", "es", "de", "it", "pt", "ja", "zh", "hi",
    "ar", "ru", "ko", "nl", "sv", "no", "da", "fi", "pl",
    "tr", "el", "he", "th", "vi", "id", "ms"
}


def validate_language_code(code: str) -> bool:
    """
    Validates if a language code is in the ISO 639 map.
    """
    if not isinstance(code, str):
        return False
    return code.lower() in VALID_LANGUAGE_CODES
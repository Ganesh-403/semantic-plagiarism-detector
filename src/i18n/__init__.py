"""
src/i18n
--------
Internationalization (i18n) package for multi-language UI support.
"""

from .translator import _SUPPORTED_LANGUAGES, get_text

__all__ = ["get_text", "_SUPPORTED_LANGUAGES"]

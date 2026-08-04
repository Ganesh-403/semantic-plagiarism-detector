"""
src/i18n/translator.py
---------------------

Translation manager for dynamic UI internationalization (i18n).
"""

# pylint: disable=streamlit-global-mutation

from __future__ import annotations

import html
import json
import logging
import os
from typing import Any, Dict

import streamlit as st


logger = logging.getLogger(__name__)

_I18N_DIR = os.path.dirname(os.path.abspath(__file__))
_SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
}
LANGUAGE_DISPLAY = _SUPPORTED_LANGUAGES
DISPLAY_TO_CODE = {
    display_name: code
    for code, display_name in _SUPPORTED_LANGUAGES.items()
}

_translations: Dict[str, Dict[str, str]] = {}


@st.cache_data(show_spinner=False)
def _load_translation_dictionary(
    file_path: str,
) -> Dict[str, str]:
    """Read and cache one translation JSON dictionary.

    The resolved file path is part of Streamlit's cache key, so each
    language file is cached independently. ``st.cache_data`` returns a
    deserialised copy to callers, preventing accidental mutation of the
    cached value.
    """
    with open(file_path, "r", encoding="utf-8") as translation_file:
        loaded = json.load(translation_file)

    if not isinstance(loaded, dict):
        raise ValueError(
            "Translation file must contain a JSON object: "
            f"{file_path}"
        )

    return {
        str(key): str(value)
        for key, value in loaded.items()
    }


def load_translations() -> None:
    """Load all supported dictionaries through the Streamlit cache.

    Missing or malformed non-English files are skipped with a warning.
    A malformed English dictionary is also logged; ``get_text`` then
    safely falls back to returning the requested key.
    """
    global _translations

    loaded_translations: Dict[str, Dict[str, str]] = {}

    for lang_code in _SUPPORTED_LANGUAGES:
        file_path = os.path.join(
            _I18N_DIR,
            f"{lang_code}.json",
        )

        if not os.path.isfile(file_path):
            logger.warning(
                "Translation file is missing for language %s: %s",
                lang_code,
                file_path,
            )
            continue

        try:
            loaded_translations[lang_code] = (
                _load_translation_dictionary(file_path)
            )
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as exception:
            logger.warning(
                "Unable to load translation file for %s: %s",
                lang_code,
                exception,
            )

    _translations = loaded_translations


def clear_translation_cache() -> None:
    """Clear cached dictionaries and reload them from disk on demand."""
    global _translations

    _load_translation_dictionary.clear()
    _translations = {}


# Preload translations on module import. Streamlit's cache prevents
# repeated disk I/O when this function is invoked during reruns.
load_translations()


def get_text(
    key: str,
    lang: str = "en",
    **kwargs: Any,
) -> str:
    """Return translated text with English and key fallbacks.

    Keyword values are HTML-escaped before ``str.format`` substitution
    to prevent untrusted values from injecting markup.
    """
    if not _translations:
        load_translations()

    language_dictionary = _translations.get(lang)
    if not language_dictionary:
        language_dictionary = _translations.get("en", {})

    text = language_dictionary.get(
        key,
        _translations.get("en", {}).get(key, key),
    )

    if kwargs:
        escaped_values = {
            name: html.escape(str(value))
            for name, value in kwargs.items()
        }
        text = text.format(**escaped_values)

    return text

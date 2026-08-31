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
import re
from typing import Any, Dict, List

import streamlit as st

logger = logging.getLogger(__name__)

_I18N_DIR = os.path.dirname(os.path.abspath(__file__))
_SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
    "zh": "中文",
}
LANGUAGE_DISPLAY = _SUPPORTED_LANGUAGES
DISPLAY_TO_CODE = {
    display_name: code for code, display_name in _SUPPORTED_LANGUAGES.items()
}

_translations: dict[str, dict[str, str]] = {}


@st.cache_data(show_spinner=False)
def _load_translation_dictionary(
    file_path: str,
) -> dict[str, str]:
    """Read and cache one translation JSON dictionary.

    The resolved file path is part of Streamlit's cache key, so each
    language file is cached independently. ``st.cache_data`` returns a
    deserialised copy to callers, preventing accidental mutation of the
    cached value.
    """
    with open(file_path, "r", encoding="utf-8") as translation_file:
        loaded = json.load(translation_file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Translation file must contain a JSON object: {file_path}")

    return {str(key): str(value) for key, value in loaded.items()}


def load_translations() -> None:
    """Load all supported dictionaries through the Streamlit cache.

    Missing or malformed non-English files are skipped with a warning.
    A malformed English dictionary is also logged; ``get_text`` then
    safely falls back to returning the requested key.
    """
    global _translations

    loaded_translations: dict[str, dict[str, str]] = {}

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
            loaded_translations[lang_code] = _load_translation_dictionary(file_path)
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


#: Matches the field name at the start of a ``str.format`` replacement field,
#: e.g. ``ai_a`` in ``{ai_a:.1%}``. Doubled braces are literal and skipped.
_FORMAT_FIELD_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)")

REFERENCE_LANGUAGE = "en"


def supported_language_codes() -> list[str]:
    """Return the language codes offered in the picker.

    Returns:
        The codes in ``_SUPPORTED_LANGUAGES``, in declaration order.
    """
    return list(_SUPPORTED_LANGUAGES)


def _format_fields(template: str) -> list[str]:
    """Return the sorted, de-duplicated field names used by *template*.

    Args:
        template: A ``str.format``-style string.

    Returns:
        Field names, e.g. ``["ai_a", "ai_b", "doc_a", "doc_b"]``.
    """
    return sorted(set(_FORMAT_FIELD_RE.findall(template)))


def missing_translation_keys(lang_code: str) -> list[str]:
    """Return the English keys that *lang_code* does not translate.

    A locale file that is missing a key is not broken — ``get_text`` falls
    back to English per key — but the language picker offers the language as
    though it were finished, and nothing else surfaces the gap. This makes it
    inspectable, and lets a test fail on it.

    Args:
        lang_code: The language code to inspect, e.g. ``"pt"``.

    Returns:
        Sorted keys present in English and absent from *lang_code*. Empty when
        the locale is complete, and empty for English itself.
    """
    if not _translations:
        load_translations()

    reference = _translations.get(REFERENCE_LANGUAGE, {})
    translated = _translations.get(lang_code, {})

    return sorted(set(reference) - set(translated))


def placeholder_mismatches(lang_code: str) -> dict[str, tuple]:
    """Return keys whose placeholders differ from the English original.

    A translation that renames ``{doc_a}`` or drops ``{ai_a:.1%}`` does not
    fail loudly: :func:`format_text` catches the ``KeyError`` and returns the
    raw template, so the user sees ``"{doc_a}"`` on screen. Comparing field
    names catches that before it ships.

    Args:
        lang_code: The language code to inspect.

    Returns:
        Mapping of key to ``(english_fields, translated_fields)`` for every
        key whose field names disagree. Empty when the locale is consistent.
    """
    if not _translations:
        load_translations()

    reference = _translations.get(REFERENCE_LANGUAGE, {})
    translated = _translations.get(lang_code, {})

    mismatches: dict[str, tuple] = {}
    for key, english_template in reference.items():
        if key not in translated:
            continue

        english_fields = _format_fields(english_template)
        local_fields = _format_fields(translated[key])
        if english_fields != local_fields:
            mismatches[key] = (english_fields, local_fields)

    return mismatches


def translation_coverage(lang_code: str) -> float:
    """Return the fraction of English keys that *lang_code* translates.

    Args:
        lang_code: The language code to inspect.

    Returns:
        A value in ``[0.0, 1.0]``. ``1.0`` when the locale is complete, and
        ``1.0`` when English carries no keys at all.
    """
    if not _translations:
        load_translations()

    reference = _translations.get(REFERENCE_LANGUAGE, {})
    if not reference:
        return 1.0

    return 1.0 - len(missing_translation_keys(lang_code)) / len(reference)


class _EscapedValue:
    """Wrapper that HTML-escapes a value *after* its format spec is applied.

    The previous implementation escaped values by calling ``html.escape(str(v))``
    before substitution. That coerced every value to ``str``, so a translation
    carrying a numeric format spec — ``"{ai_a:.1%}"``, which all four locale
    files use — raised ``ValueError: Unknown format code '%' for object of
    type 'str'``.

    Deferring the escape to ``__format__`` lets ``str.format`` apply the spec to
    the original object first, so ``{ai_a:.1%}`` still renders as ``85.0%`` and
    only the resulting text is escaped.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def __format__(self, format_spec: str) -> str:
        return html.escape(format(self._value, format_spec))

    def __str__(self) -> str:
        return html.escape(str(self._value))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_EscapedValue({self._value!r})"


def format_text(
    template: str,
    escape_html: bool = True,
    **kwargs: Any,
) -> str:
    """Substitute ``kwargs`` into ``template``, degrading instead of raising.

    Args:
        template: A ``str.format``-style template, typically from ``get_text``.
        escape_html: When True (the default, preserving existing behaviour),
            each substituted value is HTML-escaped after its format spec is
            applied. Set False only for sinks that must receive raw text and
            will never interpret entities -- for example a PDF report cell or
            a CSV field, where ``&amp;`` would be shown literally.
        **kwargs: Values to substitute.

    Returns:
        The formatted string, or the unmodified template if substitution fails.

    Notes:
        A translation file is data, not code: a missing placeholder, a stray
        brace or a spec that does not apply to the supplied value must not take
        down the page. Such failures are logged and the raw template is
        returned so the UI stays usable.
    """
    if not kwargs:
        return template

    values: dict[str, Any]
    if escape_html:
        values = {name: _EscapedValue(value) for name, value in kwargs.items()}
    else:
        values = dict(kwargs)

    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exception:
        logger.warning(
            "Unable to format translation template %r with keys %s: %s",
            template,
            sorted(kwargs),
            exception,
        )
        return template


def get_text(
    key: str,
    lang: str = "en",
    escape_html: bool = True,
    **kwargs: Any,
) -> str:
    """Return translated text with English and key fallbacks.

    Args:
        key: Translation key to look up.
        lang: Language code. Falls back to English, then to the key itself.
        escape_html: Forwarded to :func:`format_text`. See its docstring for
            when to enable it.
        **kwargs: Values substituted into the translated template.

    Returns:
        The translated and formatted string.
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

    return format_text(text, escape_html=escape_html, **kwargs)

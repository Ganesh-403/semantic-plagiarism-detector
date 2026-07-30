"""
tests/i18n/test_translator.py
------------------------------
Unit tests for the i18n translation engine.
"""

import json
import os
from unittest.mock import patch

from src.i18n.translator import _I18N_DIR, _SUPPORTED_LANGUAGES, _translations, get_text

# A key that genuinely exists in the English translations
_ENGLISH_KEY = "tab_health"
_ENGLISH_VALUE = "🖥️ System Health"


def test_translation_english():
    title = get_text("title", lang="en")
    assert "Semantic Plagiarism" in title


def test_translation_spanish():
    title = get_text("title", lang="es")
    assert "Plagio Semántico" in title


def test_translation_french():
    title = get_text("title", lang="fr")
    assert "Détection de Plagiat" in title


def test_french_supported_languages():
    assert "fr" in _SUPPORTED_LANGUAGES
    assert _SUPPORTED_LANGUAGES["fr"] == "Français"


def test_french_all_keys_match_english():
    """fr.json must have every key that en.json has."""
    en_path = os.path.join(_I18N_DIR, "en.json")
    fr_path = os.path.join(_I18N_DIR, "fr.json")
    with open(en_path, encoding="utf-8") as f:
        en_keys = set(json.load(f).keys())
    with open(fr_path, encoding="utf-8") as f:
        fr_keys = set(json.load(f).keys())
    missing = en_keys - fr_keys
    extra = fr_keys - en_keys
    assert not missing, f"Keys missing from fr.json: {missing}"
    assert not extra, f"Unexpected keys in fr.json: {extra}"


def test_spanish_missing_key_falls_back_to_english():
    """A key missing from the Spanish dict should return the English value."""
    spanish_missing = {
        k: v for k, v in _translations["es"].items() if k != _ENGLISH_KEY
    }
    modified = dict(_translations, es=spanish_missing)
    with patch.dict("src.i18n.translator._translations", modified, clear=False):
        result = get_text(_ENGLISH_KEY, lang="es")
    assert result == _ENGLISH_VALUE


def test_spanish_missing_key_returns_string_not_key():
    """Fallback should return the English translated string, not the raw key."""
    spanish_missing = {
        k: v for k, v in _translations["es"].items() if k != _ENGLISH_KEY
    }
    modified = dict(_translations, es=spanish_missing)
    with patch.dict("src.i18n.translator._translations", modified, clear=False):
        result = get_text(_ENGLISH_KEY, lang="es")
    assert "System Health" in result
    assert result != _ENGLISH_KEY


def test_nonexistent_key_returns_key_spanish():
    """A key that does not exist in any language should return itself (Spanish)."""
    result = get_text("non_existent_key_xyz", lang="es")
    assert result == "non_existent_key_xyz"


def test_nonexistent_key_returns_key_english():
    """A key that does not exist in any language should return itself (English)."""
    result = get_text("non_existent_key_xyz", lang="en")
    assert result == "non_existent_key_xyz"


def test_get_text_no_keyerror_for_missing_keys():
    """No exception should be raised for any missing-key scenario."""
    spanish_missing = {
        k: v for k, v in _translations["es"].items() if k != _ENGLISH_KEY
    }
    modified = dict(_translations, es=spanish_missing)
    with patch.dict("src.i18n.translator._translations", modified, clear=False):
        result_es = get_text(_ENGLISH_KEY, lang="es")
    result_missing = get_text("this_key_does_not_exist_at_all_42", lang="es")
    result_bad_lang = get_text("title", lang="xx")
    assert isinstance(result_es, str)
    assert isinstance(result_missing, str)
    assert isinstance(result_bad_lang, str)


def test_get_text_invalid_lang_falls_back_to_english():
    """An unsupported language code should fall back to English."""
    result = get_text("title", lang="xx")
    assert "Semantic Plagiarism" in result


def test_get_text_with_format_html_escapes_values():
    template = "Status: {label}"
    modified = dict(_translations, en={**_translations["en"], "_test_xss": template})
    with patch.dict("src.i18n.translator._translations", modified, clear=False):
        result = get_text("_test_xss", lang="en", label='<script>alert("xss")</script>')
    assert "Status: " in result
    assert "&lt;script&gt;" in result
    assert "<script>" not in result
    assert "&quot;" in result or "&#x27;" in result


def test_get_text_with_format_no_kwargs_leaves_placeholder_unchanged():
    template = "Hello {name}"
    modified = dict(_translations, en={**_translations["en"], "_test_ph": template})
    with patch.dict("src.i18n.translator._translations", modified, clear=False):
        result = get_text("_test_ph", lang="en")
    assert result == "Hello {name}"


def test_get_text_with_format_no_placeholder_ignores_unused_kwargs():
    result = get_text("title", lang="en", unused="<br>")
    assert result == "🔍 Semantic Plagiarism Detection System"

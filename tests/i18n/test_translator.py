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
tests/i18n/test_translator.py
------------------------------
Unit tests for the i18n translation engine.
"""

import json
import os
from unittest.mock import patch

import pytest

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


def test_get_language_name_known_codes():
    from src.core.translator import get_language_name

    assert get_language_name("en") == "English"
    assert get_language_name("es") == "Spanish"
    assert get_language_name("fr") == "French"
    assert get_language_name("de") == "German"
    assert get_language_name("zh") == "Chinese"
    assert get_language_name("ja") == "Japanese"
    assert get_language_name("ru") == "Russian"
    assert get_language_name("ar") == "Arabic"
    assert get_language_name("hi") == "Hindi"
    assert get_language_name("pt") == "Portuguese"
    assert get_language_name("it") == "Italian"
    assert get_language_name("nl") == "Dutch"
    assert get_language_name("ko") == "Korean"
    assert get_language_name("pl") == "Polish"
    assert get_language_name("tr") == "Turkish"
    assert get_language_name("sv") == "Swedish"
    assert get_language_name("da") == "Danish"
    assert get_language_name("fi") == "Finnish"
    assert get_language_name("no") == "Norwegian"
    assert get_language_name("uk") == "Ukrainian"
    assert get_language_name("vi") == "Vietnamese"
    assert get_language_name("cs") == "Czech"
    assert get_language_name("el") == "Greek"
    assert get_language_name("he") == "Hebrew"
    assert get_language_name("hu") == "Hungarian"
    assert get_language_name("id") == "Indonesian"
    assert get_language_name("ro") == "Romanian"
    assert get_language_name("sk") == "Slovak"


def test_get_language_name_unknown_fallback():
    from src.core.translator import get_language_name

    assert get_language_name("xyz") == "XYZ"
    assert get_language_name("unknown_code") == "UNKNOWN_CODE"
    assert get_language_name("abc123") == "ABC123"
    assert get_language_name("") == ""
    assert get_language_name(None) == ""
    assert get_language_name(123) == ""


def test_get_language_name_case_and_whitespace():
    from src.core.translator import get_language_name

    assert get_language_name(" EN ") == "English"
    assert get_language_name("ES") == "Spanish"
    assert get_language_name("  fr  ") == "French"
    assert get_language_name("De") == "German"


def test_get_language_native_name():
    from src.core.translator import get_language_native_name

    assert get_language_native_name("en") == "English"
    assert get_language_native_name("es") == "Español"
    assert get_language_native_name("fr") == "Français"
    assert get_language_native_name("de") == "Deutsch"
    assert get_language_native_name("zh") == "中文"
    assert get_language_native_name("ja") == "日本語"
    assert get_language_native_name("hi") == "हिन्दी"
    assert get_language_native_name("unknown") == "UNKNOWN"
    assert get_language_native_name("") == ""


def test_get_language_info():
    from src.core.translator import get_language_info

    info_en = get_language_info("en")
    assert info_en == {"code": "en", "name": "English", "native": "English"}

    info_es = get_language_info("ES")
    assert info_es == {"code": "es", "name": "Spanish", "native": "Español"}

    assert get_language_info("nonexistent") is None
    assert get_language_info("") is None
    assert get_language_info(None) is None


def test_is_valid_language_code():
    from src.core.translator import is_valid_language_code

    assert is_valid_language_code("en") is True
    assert is_valid_language_code("ES") is True
    assert is_valid_language_code("  fr  ") is True
    assert is_valid_language_code("xyz") is False
    assert is_valid_language_code("") is False
    assert is_valid_language_code(None) is False


def test_validate_target_language_code_valid():
    from src.core.translator import validate_target_language_code

    assert validate_target_language_code("en") is True
    assert validate_target_language_code("es") is True
    assert validate_target_language_code("fr") is True
    assert validate_target_language_code("ES") is True
    assert validate_target_language_code("  de  ") is True


def test_validate_target_language_code_invalid():
    from src.core.translator import validate_target_language_code

    with pytest.raises(ValueError) as excinfo:
        validate_target_language_code("xyz")
    assert "Unsupported target language code: xyz" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        validate_target_language_code("")
    assert "Unsupported target language code:" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        validate_target_language_code(None)
    assert "Unsupported target language code: None" in str(excinfo.value)


def test_get_supported_language_codes():
    from src.core.translator import get_supported_language_codes

    codes = get_supported_language_codes()
    assert isinstance(codes, list)
    assert len(codes) > 100
    assert "en" in codes
    assert "es" in codes
    assert "fr" in codes
    assert codes == sorted(codes)


def test_get_all_languages():
    from src.core.translator import get_all_languages

    all_langs = get_all_languages()
    assert isinstance(all_langs, dict)
    assert len(all_langs) > 100
    assert all_langs["en"] == "English"
    assert all_langs["es"] == "Spanish"
    assert all_langs["de"] == "German"


def test_normalize_language_code():
    from src.core.translator import normalize_language_code

    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("es_ES") == "es"
    assert normalize_language_code("FR") == "fr"
    assert normalize_language_code("invalid_code_123") == "invalid_code_123"
    assert normalize_language_code("") == "en"
    assert normalize_language_code(None) == "en"


def test_search_languages_by_name():
    from src.core.translator import search_languages_by_name

    results_span = search_languages_by_name("Spanish")
    assert any(code == "es" for code, name in results_span)

    results_es = search_languages_by_name("Español")
    assert any(code == "es" for code, name in results_es)

    results_eng = search_languages_by_name("English")
    assert any(code == "en" for code, name in results_eng)

    assert search_languages_by_name("NonExistentLanguageQueryXYZ") == []
    assert search_languages_by_name("") == []
    assert search_languages_by_name(None) == []


def test_batch_convert_language_codes():
    from src.core.translator import batch_convert_language_codes

    batch = ["en", "es", "fr", "unknown"]
    converted = batch_convert_language_codes(batch)
    assert converted == {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "unknown": "UNKNOWN",
    }
    assert batch_convert_language_codes([]) == {}
    assert batch_convert_language_codes(None) == {}


def test_format_language_display():
    from src.core.translator import format_language_display

    assert format_language_display("es", include_native=True) == "Spanish (Español)"
    assert format_language_display("es", include_native=False) == "Spanish"
    assert format_language_display("en", include_native=True) == "English"
    assert format_language_display("") == ""


def test_get_common_translation_pairs():
    from src.core.translator import get_common_translation_pairs

    pairs = get_common_translation_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) >= 5
    assert ("es", "en") in pairs
    assert ("fr", "en") in pairs


def test_get_language_display_name():
    from src.core.translator import get_language_display_name

    assert get_language_display_name("de") == "German"
    assert get_language_display_name("en") == "English"
    assert get_language_display_name("xyz") == "XYZ"
    assert get_language_display_name("") == ""
    assert get_language_display_name(None) == ""


def test_new_languages_supported():
    """Test that Portuguese and Chinese are present in supported languages and translations load correctly.

    This used to pin ``app_title``, which was the only key either stub file
    carried. ``app_title`` is not an i18n key at all -- it appears in no other
    locale, the window title comes from ``get_app_title()``, and no production
    code ever looked it up -- so the assertion proved only that the stub was
    still a stub. It now checks a key the UI actually renders (Issue #3048).
    """
    from src.i18n.translator import _SUPPORTED_LANGUAGES, get_text

    assert "pt" in _SUPPORTED_LANGUAGES
    assert "zh" in _SUPPORTED_LANGUAGES

    assert _SUPPORTED_LANGUAGES["pt"] == "Português"
    assert _SUPPORTED_LANGUAGES["zh"] == "中文"

    # Both locales translate the title rather than falling back to English.
    assert get_text("title", lang="pt") == "🔍 Sistema de Detecção de Plágio Semântico"
    assert get_text("title", lang="zh") == "🔍 语义查重检测系统"

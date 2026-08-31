"""
tests/i18n/test_locale_completeness.py
--------------------------------------
Guards that every language the picker offers is actually translated.

``pt.json`` and ``zh.json`` each shipped a single key while
``_SUPPORTED_LANGUAGES`` advertised both as available. ``get_text`` falls back
to English per key, so a user who chose *Português* or *中文* got a translated
title bar and 77 English strings with nothing indicating the language was
unfinished (Issue #3048).

Nothing compared key sets across locales, so the reverse drift — a key added to
``en.json`` and forgotten everywhere else — was equally invisible. These tests
close both directions.
"""

import json
import os

import pytest

from src.i18n.translator import (
    _I18N_DIR,
    _SUPPORTED_LANGUAGES,
    REFERENCE_LANGUAGE,
    load_translations,
    missing_translation_keys,
    placeholder_mismatches,
    supported_language_codes,
    translation_coverage,
)

NON_ENGLISH_CODES = [
    code for code in _SUPPORTED_LANGUAGES if code != REFERENCE_LANGUAGE
]


def _load_locale(code: str) -> dict:
    """Read one locale file straight from disk, bypassing the cache."""
    with open(os.path.join(_I18N_DIR, f"{code}.json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(autouse=True)
def _ensure_translations_loaded():
    """The module preloads on import, but ordering across tests is not fixed."""
    load_translations()


def test_every_advertised_language_has_a_file():
    """A code in the picker with no file renders an entirely English UI."""
    for code in supported_language_codes():
        path = os.path.join(_I18N_DIR, f"{code}.json")

        assert os.path.isfile(path), f"{code} is offered but {path} does not exist"


@pytest.mark.parametrize("code", NON_ENGLISH_CODES)
def test_locale_translates_every_english_key(code):
    """No advertised language may fall back to English for any key."""
    missing = missing_translation_keys(code)

    assert not missing, (
        f"{code}.json is missing {len(missing)} of the English keys: "
        f"{missing[:10]}{'…' if len(missing) > 10 else ''}"
    )


@pytest.mark.parametrize("code", NON_ENGLISH_CODES)
def test_locale_has_no_keys_english_does_not(code):
    """A key with no English counterpart is dead weight or a typo."""
    english = _load_locale(REFERENCE_LANGUAGE)
    extra = sorted(set(_load_locale(code)) - set(english))

    assert not extra, f"{code}.json defines keys absent from en.json: {extra}"


@pytest.mark.parametrize("code", NON_ENGLISH_CODES)
def test_locale_placeholders_match_english(code):
    """A renamed or dropped placeholder degrades to a raw template on screen.

    ``format_text`` catches the resulting ``KeyError`` and returns the
    template unchanged, so the user sees a literal ``{doc_a}`` rather than an
    error. That is a silent failure worth catching in CI.
    """
    mismatches = placeholder_mismatches(code)

    assert not mismatches, (
        f"{code}.json placeholders diverge from English: "
        + ", ".join(
            f"{key}: expected {expected}, found {found}"
            for key, (expected, found) in sorted(mismatches.items())
        )
    )


@pytest.mark.parametrize("code", NON_ENGLISH_CODES)
def test_locale_coverage_is_complete(code):
    """The coverage helper agrees with the key comparison."""
    assert translation_coverage(code) == pytest.approx(1.0)


@pytest.mark.parametrize("code", NON_ENGLISH_CODES)
def test_locale_values_are_non_empty_strings(code):
    """An empty string is worse than a missing key: it defeats the fallback."""
    empty = sorted(
        key for key, value in _load_locale(code).items() if not str(value).strip()
    )

    assert not empty, f"{code}.json has empty values for: {empty}"


@pytest.mark.parametrize("code", NON_ENGLISH_CODES)
def test_locale_is_not_a_verbatim_copy_of_english(code):
    """A file that only duplicates English is a stub wearing a locale's name.

    Some values legitimately match — bare emoji, "FAISS", "MD". The threshold
    catches a wholesale copy without policing individual strings.
    """
    english = _load_locale(REFERENCE_LANGUAGE)
    locale = _load_locale(code)

    identical = [key for key, value in locale.items() if english.get(key) == value]

    assert len(identical) < len(english) / 2, (
        f"{len(identical)} of {len(english)} values in {code}.json are identical "
        "to English"
    )


def test_english_reports_no_missing_keys():
    """The reference language is trivially complete."""
    assert missing_translation_keys(REFERENCE_LANGUAGE) == []
    assert translation_coverage(REFERENCE_LANGUAGE) == pytest.approx(1.0)


def test_unknown_language_reports_every_key_missing():
    """An unregistered code has translated nothing, not everything."""
    english = _load_locale(REFERENCE_LANGUAGE)

    assert missing_translation_keys("xx") == sorted(english)
    assert translation_coverage("xx") == pytest.approx(0.0)


def test_portuguese_renders_in_portuguese():
    """Spot-check the language this issue was raised for."""
    from src.i18n.translator import get_text

    assert get_text("tab_trash", lang="pt") == "🗑️ Lixeira"
    assert get_text("warn_page", lang="pt") == "Página"


def test_chinese_renders_in_chinese():
    """Spot-check the other stub locale."""
    from src.i18n.translator import get_text

    assert get_text("tab_trash", lang="zh") == "🗑️ 回收站"
    assert get_text("warn_page", lang="zh") == "页"


def test_numeric_format_spec_still_applies_in_new_locales():
    """``{ai_a:.1%}`` must survive translation in pt and zh."""
    from src.i18n.translator import get_text

    for code in ("pt", "zh"):
        rendered = get_text(
            "warn_ai_prob",
            lang=code,
            doc_a="a.pdf",
            ai_a=0.85,
            doc_b="b.pdf",
            ai_b=0.12,
        )

        assert "85.0%" in rendered, f"{code} lost the percent format spec"
        assert "12.0%" in rendered
        assert "{" not in rendered, f"{code} fell back to the raw template"

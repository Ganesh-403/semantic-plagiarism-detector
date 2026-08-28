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
tests/i18n/test_translator_formatting.py
----------------------------------------
Tests for the keyword-substitution path of the i18n layer (issue #2179).

``get_text(key, **kwargs)`` used to escape every value with
``html.escape(str(value))`` *before* substitution. That coerced values to
``str``, so any translation carrying a numeric format spec raised
``ValueError``. ``warn_ai_prob`` uses ``{ai_a:.1%}`` in all four locale files,
which made the documented kwargs API unusable -- and every call site worked
around it by calling ``.format()`` on the returned string instead.
"""

import json
import logging
import os

import pytest

from src.i18n.translator import _I18N_DIR, _SUPPORTED_LANGUAGES, format_text, get_text

# ── format_text: numeric format specs ──────────────────────────────────────────


def test_percentage_spec_is_applied_to_the_original_value():
    """Regression: this raised ValueError when values were pre-stringified."""
    result = format_text("{score:.1%}", score=0.856)
    assert result == "85.6%"


def test_percentage_spec_works_with_escaping_disabled():
    result = format_text("{score:.1%}", escape_html=False, score=0.856)
    assert result == "85.6%"


@pytest.mark.parametrize(
    "template, value, expected",
    [
        ("{n:.2f}", 3.14159, "3.14"),
        ("{n:,}", 1234567, "1,234,567"),
        ("{n:05d}", 42, "00042"),
        ("{n:+.1f}", 2.5, "+2.5"),
        ("{n:.1%}", 0.5, "50.0%"),
    ],
)
def test_numeric_specs_survive(template, value, expected):
    assert format_text(template, n=value) == expected
    assert format_text(template, escape_html=False, n=value) == expected


# ── format_text: escaping ──────────────────────────────────────────────────────


def test_escaping_is_on_by_default():
    """The pre-existing guarantee: substituted values cannot inject markup."""
    result = format_text("Search: {query}", query="<b>x</b>")
    assert result == "Search: &lt;b&gt;x&lt;/b&gt;"


def test_escaping_can_be_disabled_for_raw_text_sinks():
    result = format_text("Search: {query}", escape_html=False, query="O'Brien & Co")
    assert result == "Search: O'Brien & Co"


def test_escaping_covers_quotes_and_ampersands():
    result = format_text("{v}", v='O\'Brien & "Co"')
    assert "&amp;" in result
    assert "&#x27;" in result
    assert "&quot;" in result


def test_escaping_applies_after_the_format_spec():
    """The spec runs on the real value; only the rendered text is escaped."""
    result = format_text("{v:>12}", v="<i>")
    assert result == "         &lt;i&gt;"


# ── format_text: graceful degradation ──────────────────────────────────────────


def test_missing_placeholder_returns_the_template(caplog):
    """A KeyError used to propagate and take down the page."""
    with caplog.at_level(logging.WARNING):
        result = format_text("Threshold: >{pct}%", other=1)

    assert result == "Threshold: >{pct}%"
    assert "Unable to format translation template" in caplog.text


def test_stray_brace_returns_the_template():
    assert format_text("100% sure {", value=1) == "100% sure {"


def test_spec_incompatible_with_the_value_returns_the_template():
    assert format_text("{v:.1%}", v="not-a-number") == "{v:.1%}"


def test_positional_placeholder_returns_the_template():
    assert format_text("{0} and {name}", name="x") == "{0} and {name}"


def test_no_kwargs_returns_the_template_verbatim():
    """Braces must be left alone when there is nothing to substitute."""
    assert format_text("Literal {braces} kept") == "Literal {braces} kept"


def test_extra_kwargs_are_ignored():
    assert format_text("Hello {name}", name="Ana", unused=1) == "Hello Ana"


# ── get_text integration ───────────────────────────────────────────────────────


@pytest.mark.parametrize("lang", sorted(_SUPPORTED_LANGUAGES))
def test_warn_ai_prob_formats_in_every_locale(lang):
    """The real translation that triggered the bug, in all shipped locales."""
    result = get_text(
        "warn_ai_prob",
        lang=lang,
        doc_a="essay_a.docx",
        ai_a=0.85,
        doc_b="essay_b.docx",
        ai_b=0.42,
    )

    assert "85.0%" in result
    assert "42.0%" in result
    assert "essay_a.docx" in result
    assert "{" not in result


def test_get_text_substitutes_without_external_format():
    result = get_text("warn_filter_threshold", lang="en", pct="80")
    assert "80" in result
    assert "{pct}" not in result


def test_get_text_missing_kwargs_degrades_instead_of_raising():
    result = get_text("warn_filter_threshold", lang="en")
    assert "{pct}" in result  # untranslated placeholder, but no exception


def test_get_text_unknown_key_returns_the_key():
    assert (
        get_text("definitely_not_a_real_key", lang="en") == "definitely_not_a_real_key"
    )


def test_get_text_unknown_language_falls_back_to_english():
    english = get_text("warn_filter_threshold", lang="en", pct="80")
    assert get_text("warn_filter_threshold", lang="zz", pct="80") == english


def test_get_text_escapes_by_default_and_can_opt_out():
    escaped = get_text("warn_filter_document", lang="en", doc="A&B")
    raw = get_text("warn_filter_document", lang="en", escape_html=False, doc="A&B")

    assert "A&amp;B" in escaped
    assert "A&B" in raw


# ── locale file consistency ────────────────────────────────────────────────────


@pytest.mark.parametrize("lang", sorted(_SUPPORTED_LANGUAGES))
def test_every_locale_formats_with_the_same_keys_as_english(lang):
    """A translator adding or renaming a placeholder would break substitution."""
    import string

    def placeholders(text):
        return {
            name for _, name, _, _ in string.Formatter().parse(text) if name is not None
        }

    with open(os.path.join(_I18N_DIR, "en.json"), encoding="utf-8") as handle:
        english = json.load(handle)
    with open(os.path.join(_I18N_DIR, f"{lang}.json"), encoding="utf-8") as handle:
        translated = json.load(handle)

    mismatched = {
        key: (placeholders(value), placeholders(translated[key]))
        for key, value in english.items()
        if key in translated and placeholders(value) != placeholders(translated[key])
    }

    assert not mismatched, f"{lang}.json placeholder mismatch: {mismatched}"

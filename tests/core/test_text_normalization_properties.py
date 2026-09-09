"""Properties of the production cleaners, rather than test-only stand-ins."""
import unicodedata

import pytest
from hypothesis import example, given, settings, strategies as st
from src.core import document_parser as legacy
from src.core.parsers import cleaners


# Exercise both supported import paths while the parser extraction is maintained.
@pytest.mark.parametrize("module", [legacy, cleaners])
@given(st.text())
def test_unicode_normalization_is_canonical_and_idempotent(module, text):
    result = module.normalize_unicode_nfc(text)
    assert unicodedata.is_normalized("NFC", result)
    assert module.normalize_unicode_nfc(result) == result
    assert unicodedata.normalize("NFD", result) == unicodedata.normalize("NFD", text)


@pytest.mark.parametrize("module", [legacy, cleaners])
@settings(max_examples=1000)
@example("0\u00a0\u00a00")
@example("0 \u200b 0")
@example("0\t\xa00")
@given(st.text())
def test_cleaning_is_idempotent(module, text):
    result = module.clean_text(text)
    assert result == result.strip()
    assert module.clean_text(result) == result
    assert "  " not in result
    assert "\t" not in result
    assert "\n\n\n" not in result


@pytest.mark.parametrize("module", [legacy, cleaners])
@given(st.text(alphabet=st.characters(categories=("L", "N"))))
def test_cleaning_preserves_letters_and_numbers(module, text):
    assert module.clean_text(text) == text


@pytest.mark.parametrize("module", [legacy, cleaners])
@given(st.text())
def test_zero_width_sanitization_is_idempotent(module, text):
    result = module.sanitize_zero_width_characters(text)
    assert not module.ZERO_WIDTH_CHARS_PATTERN.search(result)
    assert module.sanitize_zero_width_characters(result) == result


@pytest.mark.parametrize("module", [legacy, cleaners])
@given(st.text())
def test_punctuation_normalization_is_idempotent(module, text):
    result = module.normalize_extended_punctuation(text)
    assert module.normalize_extended_punctuation(result) == result
    assert not set(result) & set("“”‘’—…")


@pytest.mark.parametrize("module", [legacy, cleaners])
@pytest.mark.parametrize("text", ["", "   ", "\n\t", "Café déjà vu", "学生的学术文章", "مرحبا بالعالم", "👩‍💻 works"])
def test_unicode_cleaning_examples(module, text):
    result = module.clean_text(text)
    assert isinstance(result, str)
    assert module.clean_text(result) == result


@pytest.mark.parametrize("module", [legacy, cleaners])
@pytest.mark.parametrize("header", ["References", "Bibliography", "Works Cited", "Citations"])
def test_bibliography_removal_respects_standalone_headers(module, header):
    body = "The researcher discusses evidence and conclusions."
    assert module.strip_bibliography(f"{body}\n\n{header}\nPublished source") == body
    assert module.strip_bibliography(f"{body} The {header} are discussed here.") == f"{body} The {header} are discussed here."


@pytest.mark.parametrize("module", [legacy, cleaners])
def test_custom_stopwords_and_punctuation(module, monkeypatch):
    monkeypatch.setattr(module, "get_stopwords", lambda: frozenset({"the", "domain"}))
    assert module.clean_text("The DOMAIN, specific words remain.", remove_stopwords=True) == "specific words remain."


@pytest.mark.parametrize("module", [legacy, cleaners])
def test_non_latin_letters_are_not_stripped_by_embedding_preparation(module, monkeypatch):
    monkeypatch.setattr(module, "detect_text_language", lambda _: "unknown")
    monkeypatch.setattr(module, "translate_text", lambda *args, **kwargs: pytest.fail("Unknown language must not be sent for translation"))
    text = "  Ελληνικά, العربية, 中文, café.  "
    result = module.prepare_text_for_embedding(text)
    assert result["original_text"] == text.strip()
    assert result["embedding_text"] == text.strip()
    assert result["was_translated"] is False

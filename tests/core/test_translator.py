import pytest

from src.core.translator import get_language_display_name, translate_text


def test_none_is_preserved():
    assert translate_text(None) is None


def test_empty_string_is_preserved():
    assert translate_text("") == ""


def test_invalid_language_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        translate_text("Hello", target_lang="invalid_lang")
    assert "Unsupported target language code: invalid_lang" in str(excinfo.value)


def test_get_language_display_name():
    assert get_language_display_name("es") == "Spanish (Español)"
    assert get_language_display_name("en") == "English"
    assert get_language_display_name("INVALID") == "INVALID"


def test_translate_text_basic():
    # Translate a simple French sentence to English
    result = translate_text("Bonjour tout le monde", target_lang="en")
    assert "hello" in result.lower() or "everyone" in result.lower()


def test_translate_text_empty():
    # Empty inputs should be returned as-is
    assert translate_text("") == ""
    assert translate_text("   ") == "   "
    assert translate_text(None) is None


def test_translate_text_error_handling():
    # An unsupported target language code is rejected before reaching the model
    with pytest.raises(ValueError) as excinfo:
        translate_text("Hello", target_lang="invalid_lang")
    assert "Unsupported target language code: invalid_lang" in str(excinfo.value)

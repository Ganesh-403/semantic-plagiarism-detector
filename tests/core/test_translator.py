import pytest
from unittest.mock import patch

from src.core.translator import get_language_display_name, translate_text, validate_target_language_code


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


@patch("src.core.translator.GoogleTranslator.translate")
def test_translate_text_network_exception(mock_translate):
    # Force the mock to raise a simulated network timeout
    mock_translate.side_effect = Exception("Connection timed out")
    
    # Call the function
    result = translate_text("Hello, world!", target_lang="es")
    
    # Verify it didn't crash and returned the expected error string
    assert result.startswith("(Translation Error:")
    assert "Connection timed out" in result


def test_validate_target_language_code_valid():
    valid_codes = ["en", "es", "fr"]
    for code in valid_codes:
        validate_target_language_code(code)


def test_validate_target_language_code_invalid():
    invalid_codes = ["invalid", "", None]
    for code in invalid_codes:
        with pytest.raises(ValueError):
            validate_target_language_code(code)
            
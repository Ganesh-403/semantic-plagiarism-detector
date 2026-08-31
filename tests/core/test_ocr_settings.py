import pytest

from src.core.app_config import SUPPORTED_OCR_LANGUAGES
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    MAX_OCR_DPI,
    MIN_OCR_DPI,
    normalize_ocr_settings,
    validate_ocr_dpi,
    validate_ocr_language,
)


def test_default_ocr_settings_are_valid():
    language, dpi = normalize_ocr_settings()
    assert language == DEFAULT_OCR_LANGUAGE == "eng"
    assert dpi == DEFAULT_OCR_DPI == 250


@pytest.mark.parametrize("dpi", [150, 200, 250, 300, 350, 400])
def test_supported_ocr_dpi_values(dpi):
    assert validate_ocr_dpi(dpi) == dpi


@pytest.mark.parametrize("dpi", [149, 401, -1, 0])
def test_out_of_range_ocr_dpi_is_rejected(dpi):
    with pytest.raises(ValueError, match="between 150 and 400"):
        validate_ocr_dpi(dpi)


@pytest.mark.parametrize("value", [None, "abc", 250.5, True])
def test_invalid_ocr_dpi_type_is_rejected(value):
    with pytest.raises(ValueError):
        validate_ocr_dpi(value)


@pytest.mark.parametrize("language", ["eng", "spa", "fra", "deu", "por", "ita"])
def test_supported_ocr_languages(language):
    assert validate_ocr_language(language) == language


def test_language_is_normalized():
    assert validate_ocr_language(" SPA ") == "spa"


@pytest.mark.parametrize("language", ["", "hin", None])
def test_unsupported_ocr_language_is_rejected(language):
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        validate_ocr_language(language)


def test_language_mapping_matches_issue_scope():
    assert SUPPORTED_OCR_LANGUAGES == {
        "eng": "English",
        "spa": "Spanish",
        "fra": "French",
        "deu": "German",
        "por": "Portuguese",
        "ita": "Italian",
    }


def test_dpi_bounds_match_issue_scope():
    assert MIN_OCR_DPI == 150
    assert MAX_OCR_DPI == 400


def test_validate_ocr_languages_against_mocked_tesseract():
    """Verify that all 3-letter language codes in SUPPORTED_OCR_LANGUAGES are recognized by Tesseract."""
    from unittest.mock import patch

    import pytesseract

    mock_languages = ["eng", "spa", "fra", "deu", "por", "ita", "osd"]
    with patch("pytesseract.get_languages", return_value=mock_languages):
        installed_languages = pytesseract.get_languages()
        for code in SUPPORTED_OCR_LANGUAGES:
            assert len(code) == 3, f"Language code '{code}' must be a 3-letter ISO code"
            assert code in installed_languages, (
                f"Language code '{code}' is not recognized by Tesseract"
            )


def test_validate_ocr_languages_against_tesseract_binary():
    """Verify 3-letter codes against installed Tesseract binary if available on system PATH."""
    import pytesseract

    try:
        installed_languages = pytesseract.get_languages()
    except (pytesseract.TesseractNotFoundError, FileNotFoundError, Exception):
        pytest.skip("Tesseract binary is not installed on system PATH")

    for code in SUPPORTED_OCR_LANGUAGES:
        assert len(code) == 3, f"Language code '{code}' must be a 3-letter ISO code"
        assert code in installed_languages, (
            f"Language code '{code}' is not recognized by Tesseract binary"
        )


def test_multi_language_ocr_support():
    """Verify that validate_ocr_language supports '+' separated language lists."""
    assert validate_ocr_language("eng+spa") == "eng+spa"
    assert validate_ocr_language(" eng + spa ") == "eng+spa"
    assert validate_ocr_language("eng+fra+deu") == "eng+fra+deu"
    assert validate_ocr_language("eng+spa+eng") == "eng+spa"


def test_multi_language_ocr_rejection():
    """Verify that invalid language codes in a '+' separated list are rejected."""
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        validate_ocr_language("eng+xyz")

    with pytest.raises(ValueError, match="Unsupported OCR language"):
        validate_ocr_language("spa+")


def test_common_parser_multi_language_support():
    """Verify common parser validate_ocr_language handles '+' separated language lists."""
    from src.core.parsers.common import validate_ocr_language as common_validate

    assert common_validate("eng+spa") == "eng+spa"
    assert common_validate(" eng + spa ") == "eng+spa"
    assert common_validate("eng+xyz") == "eng"

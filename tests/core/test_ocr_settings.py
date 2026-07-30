import pytest

from src.core.app_config import SUPPORTED_OCR_LANGUAGES
from src.core.document_parser import (DEFAULT_OCR_DPI, DEFAULT_OCR_LANGUAGE,
                                      MAX_OCR_DPI, MIN_OCR_DPI,
                                      normalize_ocr_settings, validate_ocr_dpi,
                                      validate_ocr_language)


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


@pytest.mark.parametrize("language", ["eng", "spa", "fra"])
def test_supported_ocr_languages(language):
    assert validate_ocr_language(language) == language


def test_language_is_normalized():
    assert validate_ocr_language(" SPA ") == "spa"


@pytest.mark.parametrize("language", ["", "deu", "hin", None])
def test_unsupported_ocr_language_is_rejected(language):
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        validate_ocr_language(language)


def test_language_mapping_matches_issue_scope():
    assert SUPPORTED_OCR_LANGUAGES == {
        "eng": "English",
        "spa": "Spanish",
        "fra": "French",
    }


def test_dpi_bounds_match_issue_scope():
    assert MIN_OCR_DPI == 150
    assert MAX_OCR_DPI == 400

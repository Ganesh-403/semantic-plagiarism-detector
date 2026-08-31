from unittest.mock import MagicMock, patch
import pytest
from src.core.translator import translate_text, translate_text_batch


def test_translate_text_uses_deepl_when_key_present(monkeypatch):
    """Verify DeepL is used as primary engine when DEEPL_API_KEY is configured."""
    monkeypatch.setenv("DEEPL_API_KEY", "fake-deepl-key-123")

    mock_deepl = MagicMock()
    mock_deepl.return_value.translate.return_value = "Texto traducido por DeepL"

    with patch("src.core.translator.DeeplTranslator", mock_deepl):
        res = translate_text("Text translated by DeepL", target_lang="es")
        assert res == "Texto traducido por DeepL"
        mock_deepl.assert_called_once_with(
            api_key="fake-deepl-key-123",
            source="auto",
            target="es",
        )


def test_translate_text_falls_back_to_google_on_deepl_error(monkeypatch):
    """Verify GoogleTranslator fallback when DeepL fails with an exception."""
    monkeypatch.setenv("DEEPL_API_KEY", "invalid-deepl-key")

    mock_deepl = MagicMock()
    mock_deepl.return_value.translate.side_effect = RuntimeError("DeepL quota exceeded")

    mock_google = MagicMock()
    mock_google.return_value.translate.return_value = "Texto traducido por Google"

    with patch("src.core.translator.DeeplTranslator", mock_deepl), \
         patch("src.core.translator.GoogleTranslator", mock_google):
        res = translate_text("Hello world", target_lang="es")
        assert res == "Texto traducido por Google"
        mock_google.assert_called_once()


def test_translate_text_batch_uses_deepl_when_key_present(monkeypatch):
    """Verify batch translation uses DeepL when DEEPL_API_KEY is configured."""
    monkeypatch.setenv("DEEPL_API_KEY", "fake-deepl-key-123")

    mock_deepl = MagicMock()
    mock_deepl.return_value.translate_batch.return_value = ["Uno", "Dos"]

    with patch("src.core.translator.DeeplTranslator", mock_deepl):
        res = translate_text_batch(["One", "Two"], target_lang="es")
        assert res == ["Uno", "Dos"]
        mock_deepl.assert_called_once_with(
            api_key="fake-deepl-key-123",
            source="auto",
            target="es",
        )


def test_translate_text_batch_falls_back_to_google_on_deepl_error(monkeypatch):
    """Verify batch translation fallback to Google when DeepL fails."""
    monkeypatch.setenv("DEEPL_API_KEY", "fake-deepl-key-123")

    mock_deepl = MagicMock()
    mock_deepl.return_value.translate_batch.side_effect = RuntimeError("DeepL error")

    mock_google = MagicMock()
    mock_google.return_value.translate_batch.return_value = ["Uno", "Dos"]

    with patch("src.core.translator.DeeplTranslator", mock_deepl), \
         patch("src.core.translator.GoogleTranslator", mock_google):
        res = translate_text_batch(["One", "Two"], target_lang="es")
        assert res == ["Uno", "Dos"]
        mock_google.assert_called_once()

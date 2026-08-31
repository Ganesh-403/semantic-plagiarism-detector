import sys
from unittest.mock import MagicMock, patch
import pytest
from src.core.translator import translate_text_marian, translate_text, _MARIAN_PIPELINES


def setup_function():
    _MARIAN_PIPELINES.clear()


def test_translate_text_marian_invokes_transformers_pipeline():
    """Verify translate_text_marian instantiates and calls transformers translation pipeline."""
    mock_pipeline_func = MagicMock()
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [{"translation_text": "This is a test translated offline."}]
    mock_pipeline_func.return_value = mock_pipe_instance

    mock_transformers = MagicMock()
    mock_transformers.pipeline = mock_pipeline_func

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = translate_text_marian("Este es un texto de prueba.", source_lang="es", target_lang="en")
        assert result == "This is a test translated offline."
        mock_pipeline_func.assert_called_once_with("translation", model="Helsinki-NLP/opus-mt-es-en")
        mock_pipe_instance.assert_called_once_with("Este es un texto de prueba.")


def test_translate_text_marian_caches_pipeline():
    """Verify multiple calls reuse cached pipeline instance."""
    mock_pipeline_func = MagicMock()
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [{"translation_text": "Hello"}]
    mock_pipeline_func.return_value = mock_pipe_instance

    mock_transformers = MagicMock()
    mock_transformers.pipeline = mock_pipeline_func

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        res1 = translate_text_marian("Hola", source_lang="es", target_lang="en")
        res2 = translate_text_marian("Mundo", source_lang="es", target_lang="en")
        assert res1 == "Hello"
        assert res2 == "Hello"
        # Pipeline constructor only called once
        assert mock_pipeline_func.call_count == 1
        assert mock_pipe_instance.call_count == 2


def test_translate_text_marian_handles_missing_transformers():
    """Verify graceful handling when transformers or model fails to load."""
    mock_transformers = MagicMock()
    mock_transformers.pipeline.side_effect = ImportError("No module named 'transformers'")

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = translate_text_marian("Bonjour", source_lang="fr", target_lang="en")
        assert result.startswith("(Translation Error: MarianMT")


def test_translate_text_respects_offline_translation_enabled_flag(monkeypatch):
    """Verify translate_text routes to translate_text_marian when OFFLINE_TRANSLATION_ENABLED=true."""
    monkeypatch.setenv("OFFLINE_TRANSLATION_ENABLED", "true")

    mock_marian = MagicMock(return_value="Offline Translation Result")
    with patch("src.core.translator.translate_text_marian", mock_marian):
        res = translate_text("Texto de entrada", target_lang="en", source_lang="es")
        assert res == "Offline Translation Result"
        mock_marian.assert_called_once_with("Texto de entrada", source_lang="es", target_lang="en")

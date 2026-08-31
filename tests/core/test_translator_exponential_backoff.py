from unittest.mock import MagicMock, patch
import pytest
from src.core.translator import translate_text


def test_translate_text_succeeds_on_first_attempt():
    """Verify translate_text succeeds immediately on first attempt."""
    mock_gt = MagicMock()
    mock_gt.return_value.translate.return_value = "Hola"

    with patch("src.core.translator.GoogleTranslator", mock_gt):
        res = translate_text("Hello", target_lang="es")
        assert res == "Hola"
        assert mock_gt.return_value.translate.call_count == 1


def test_translate_text_retries_with_exponential_backoff_and_succeeds():
    """Verify translate_text retries on transient errors and succeeds on 3rd attempt."""
    mock_gt = MagicMock()
    # Fails twice with rate limit, then succeeds on 3rd attempt
    mock_gt.return_value.translate.side_effect = [
        RuntimeError("429 Too Many Requests"),
        RuntimeError("Connection timeout"),
        "Hola Mundo",
    ]

    mock_sleep = MagicMock()

    with patch("src.core.translator.GoogleTranslator", mock_gt), \
         patch("time.sleep", mock_sleep):
        res = translate_text("Hello World", target_lang="es")
        assert res == "Hola Mundo"
        assert mock_gt.return_value.translate.call_count == 3
        # Verified sleep calls: 2^0 = 1s, 2^1 = 2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)


def test_translate_text_exhausts_retries_and_returns_error():
    """Verify translate_text returns error after exhausting all 3 attempts."""
    mock_gt = MagicMock()
    mock_gt.return_value.translate.side_effect = RuntimeError("Persistent rate limit")

    mock_sleep = MagicMock()

    with patch("src.core.translator.GoogleTranslator", mock_gt), \
         patch("time.sleep", mock_sleep):
        res = translate_text("Hello World", target_lang="es")
        assert res.startswith("(Translation Error: Persistent rate limit)")
        assert mock_gt.return_value.translate.call_count == 3
        # Sleep called twice: 1s, 2s (no sleep after final attempt)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1
        assert mock_sleep.call_args_list[1][0][0] == 2

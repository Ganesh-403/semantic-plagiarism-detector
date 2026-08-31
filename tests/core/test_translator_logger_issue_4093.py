"""
tests/core/test_translator_logger_issue_4093.py
-----------------------------------------------
Regression tests for the undefined ``logger`` in ``src.core.translator``
(issue #4093).

``translate_text_batch()`` wraps its provider call in a ``try/except``. The
handler logs the failure and then falls back to translating each string on its
own, so one bad batch degrades into slower-but-working individual calls rather
than losing every result.

That handler referenced a module-level ``logger`` that was never defined. The
first statement inside ``except`` therefore raised ``NameError`` and took the
whole handler with it: the per-text fallback loop underneath was unreachable
dead code, and the ``NameError`` masked the real provider error on the way out.

The bug only showed up when the provider actually failed, which is exactly the
path nobody exercises on a good day. These tests force that path.
"""

import importlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.core import translator
from src.core.translator import translate_text_batch


# ── module-level logger ────────────────────────────────────────────────────────


def test_module_defines_a_logger():
    """``translator`` must expose a module-level ``logger``.

    This is the direct regression assert. Before the fix the name simply did
    not exist and ``translator.logger`` raised ``AttributeError``.
    """
    assert hasattr(translator, "logger"), (
        "src.core.translator must define a module-level logger; the except "
        "block in translate_text_batch() references it."
    )
    assert isinstance(translator.logger, logging.Logger)


def test_logger_is_namespaced_to_the_module():
    """The logger must use ``__name__`` so log records are attributable.

    A bare ``logging.getLogger()`` would emit on the root logger and make
    translation failures indistinguishable from anything else in the process.
    """
    assert translator.logger.name == "src.core.translator"


def test_logger_survives_module_reload():
    """The logger is defined at import time, not lazily inside a function."""
    reloaded = importlib.reload(translator)
    assert isinstance(reloaded.logger, logging.Logger)
    assert reloaded.logger.name == "src.core.translator"


# ── the fallback path that the NameError used to kill ──────────────────────────


@pytest.fixture
def failing_batch():
    """Patch GoogleTranslator so ``translate_batch`` always raises."""
    with patch("src.core.translator.GoogleTranslator") as gt:
        instance = MagicMock()
        instance.translate_batch.side_effect = RuntimeError("rate limited")
        gt.return_value = instance
        yield gt


def test_batch_failure_does_not_raise(failing_batch):
    """A provider failure must be handled, not propagated.

    Before the fix this raised ``NameError: name 'logger' is not defined``.
    """
    result = translate_text_batch(["hello", "world"], target_lang="es")
    assert isinstance(result, list)


def test_batch_failure_falls_back_per_text(failing_batch):
    """Every input string must still produce an output slot.

    The fallback loop calls ``translate_text`` once per item. It never ran
    while the handler was dying on its first statement.
    """
    with patch("src.core.translator.translate_text") as single:
        single.side_effect = lambda text, **kw: f"translated::{text}"
        result = translate_text_batch(["alpha", "beta", "gamma"], target_lang="es")

    assert result == [
        "translated::alpha",
        "translated::beta",
        "translated::gamma",
    ]
    assert single.call_count == 3


def test_fallback_preserves_input_length(failing_batch):
    """Output length must always match input length.

    Callers zip the result back against the source documents, so a short list
    would silently misalign translations with their originals.
    """
    texts = [f"sentence number {i}" for i in range(12)]
    with patch("src.core.translator.translate_text") as single:
        single.side_effect = lambda text, **kw: text.upper()
        result = translate_text_batch(texts, target_lang="fr")

    assert len(result) == len(texts)


def test_fallback_forwards_language_arguments(failing_batch):
    """The per-text retry must use the same language pair as the batch call."""
    with patch("src.core.translator.translate_text") as single:
        single.return_value = "ok"
        translate_text_batch(["x"], target_lang="de", source_lang="en")

    _, kwargs = single.call_args
    assert kwargs["target_lang"] == "de"
    assert kwargs["source_lang"] == "en"


def test_fallback_substitutes_a_marker_when_a_single_text_also_fails(failing_batch):
    """If the individual retry fails too, the slot gets an error marker.

    The marker keeps positional alignment intact and carries the
    ``"Translation Error"`` prefix that the cross-lingual preprocessing layer
    already knows to detect and skip before embedding.
    """
    with patch("src.core.translator.translate_text") as single:
        single.side_effect = RuntimeError("still down")
        result = translate_text_batch(["one", "two"], target_lang="es")

    assert len(result) == 2
    for slot in result:
        assert "Translation Error" in slot


def test_fallback_coerces_none_results_to_empty_strings(failing_batch):
    """``translate_text`` returns ``None`` for ``None`` input; slots stay strings."""
    with patch("src.core.translator.translate_text") as single:
        single.return_value = None
        result = translate_text_batch(["a", "b"], target_lang="es")

    assert result == ["", ""]


# ── the failure is actually reported ───────────────────────────────────────────


def test_batch_failure_is_logged(failing_batch, caplog):
    """The provider error must reach the log at ERROR level.

    This is the whole point of the handler's first statement. It previously
    raised instead of logging, so the real cause was never recorded anywhere.
    """
    with caplog.at_level(logging.ERROR, logger="src.core.translator"):
        with patch("src.core.translator.translate_text", return_value="ok"):
            translate_text_batch(["hello"], target_lang="es")

    assert caplog.records, "translate_text_batch() logged nothing on failure"
    assert any("rate limited" in record.getMessage() for record in caplog.records)


def test_log_record_carries_the_original_exception_text(failing_batch, caplog):
    """The logged message must name the underlying error, not a generic string."""
    with caplog.at_level(logging.ERROR, logger="src.core.translator"):
        with patch("src.core.translator.translate_text", return_value="ok"):
            translate_text_batch(["hello"], target_lang="es")

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Batch translation error" in messages


def test_nothing_is_logged_on_the_happy_path(caplog):
    """A successful batch must stay silent at ERROR level."""
    with patch("src.core.translator.GoogleTranslator") as gt:
        instance = MagicMock()
        instance.translate_batch.return_value = ["hola", "mundo"]
        gt.return_value = instance

        with caplog.at_level(logging.ERROR, logger="src.core.translator"):
            result = translate_text_batch(["hello", "world"], target_lang="es")

    assert result == ["hola", "mundo"]
    assert not caplog.records


# ── behaviour that must not regress alongside the fix ──────────────────────────


def test_empty_input_short_circuits():
    """An empty list returns early without touching the provider."""
    with patch("src.core.translator.GoogleTranslator") as gt:
        assert translate_text_batch([], target_lang="es") == []
        gt.assert_not_called()


def test_invalid_target_language_still_raises():
    """Validation runs before the provider call and must keep raising.

    The handler only guards provider failures. A bad language code is a caller
    error and should surface as ``ValueError``, not be swallowed into markers.
    """
    with pytest.raises(ValueError):
        translate_text_batch(["hello"], target_lang="not-a-language")


def test_successful_batch_results_are_stripped():
    """Provider whitespace is normalised on the happy path."""
    with patch("src.core.translator.GoogleTranslator") as gt:
        instance = MagicMock()
        instance.translate_batch.return_value = ["  hola  ", "\tmundo\n"]
        gt.return_value = instance
        result = translate_text_batch(["hello", "world"], target_lang="es")

    assert result == ["hola", "mundo"]


def test_successful_batch_coerces_none_entries():
    """A ``None`` inside the provider response becomes an empty string."""
    with patch("src.core.translator.GoogleTranslator") as gt:
        instance = MagicMock()
        instance.translate_batch.return_value = ["hola", None]
        gt.return_value = instance
        result = translate_text_batch(["hello", "world"], target_lang="es")

    assert result == ["hola", ""]

"""Tests for the configurable translation socket timeout (Issue #3990)."""

from __future__ import annotations

import importlib

import pytest
import requests

import src.core.translator as translator


def test_default_timeout_is_ten_seconds():
    assert translator.TRANSLATION_TIMEOUT_SECONDS == 10.0


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("TRANSLATION_TIMEOUT_SECONDS", "3.5")
    try:
        reloaded = importlib.reload(translator)
        assert reloaded.TRANSLATION_TIMEOUT_SECONDS == 3.5
    finally:
        monkeypatch.delenv("TRANSLATION_TIMEOUT_SECONDS", raising=False)
        importlib.reload(translator)


def test_request_timeout_injects_default_when_absent(monkeypatch):
    captured: dict = {}

    def fake_request(self, *args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("short-circuit before any network call")

    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)

    with pytest.raises(RuntimeError):
        with translator._request_timeout(7.25):
            requests.get("http://translation.invalid")

    assert captured["timeout"] == 7.25


def test_request_timeout_preserves_explicit_timeout(monkeypatch):
    captured: dict = {}

    def fake_request(self, *args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("short-circuit")

    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)

    with pytest.raises(RuntimeError):
        with translator._request_timeout(7.25):
            requests.get("http://translation.invalid", timeout=1.0)

    assert captured["timeout"] == 1.0


def test_request_timeout_uses_module_constant_by_default(monkeypatch):
    captured: dict = {}

    def fake_request(self, *args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("short-circuit")

    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)
    monkeypatch.setattr(translator, "TRANSLATION_TIMEOUT_SECONDS", 4.0)

    with pytest.raises(RuntimeError):
        with translator._request_timeout():
            requests.get("http://translation.invalid")

    assert captured["timeout"] == 4.0


def test_request_timeout_restores_original_on_exit():
    original = requests.sessions.Session.request
    with translator._request_timeout(5):
        assert requests.sessions.Session.request is not original
    assert requests.sessions.Session.request is original


def test_request_timeout_restores_original_on_exception():
    original = requests.sessions.Session.request
    with pytest.raises(ValueError):
        with translator._request_timeout(5):
            raise ValueError("boom")
    assert requests.sessions.Session.request is original


def test_translate_text_applies_timeout_to_the_request(monkeypatch):
    """translate_text() runs the provider call inside the timeout context."""
    seen: dict = {}

    def fake_request(self, *args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise RuntimeError("no outbound network in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    monkeypatch.delenv("OFFLINE_TRANSLATION_ENABLED", raising=False)
    monkeypatch.setattr(translator, "TRANSLATION_TIMEOUT_SECONDS", 4.0)
    monkeypatch.setattr(translator.time, "sleep", lambda *_a, **_k: None)  # skip backoff

    result = translator.translate_text("Hola mundo", target_lang="en", source_lang="es")

    assert "Translation Error" in result  # network was short-circuited
    assert seen["timeout"] == 4.0  # timeout was applied to the outbound request

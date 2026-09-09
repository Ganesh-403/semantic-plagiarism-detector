"""Provider protocol, timeout isolation, and failure handling without network calls."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import requests

from src.core.translator import _request_timeout
from src.utils import translation_providers as providers


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.closed = data, status, False

    def json(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def test_deepl_batches_authenticates_and_preserves_order(monkeypatch):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response({"translations": [{"text": text + " translated"} for text in kwargs["json"]["text"]]})
    monkeypatch.setattr(requests, "post", post)
    texts = [str(i) for i in range(51)]
    with _request_timeout(3.5):
        output = providers.DeeplTranslator("secret:fx", source="auto", target="de").translate_batch(texts)
    assert output == [text + " translated" for text in texts]
    assert [len(call[1]["json"]["text"]) for call in calls] == [50, 1]
    assert calls[0][0] == "https://api-free.deepl.com/v2/translate"
    assert calls[0][1]["headers"] == {"Authorization": "DeepL-Auth-Key secret:fx"}
    assert calls[0][1]["timeout"] == 3.5
    assert "source_lang" not in calls[0][1]["json"]


def test_google_api_protocol_and_html_decoding(monkeypatch):
    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "test-key")
    response = Response({"data": {"translations": [{"translatedText": "Tom &amp; Jerry"}]}})
    def post(url, **kwargs):
        assert url == "https://translation.googleapis.com/language/translate/v2"
        assert kwargs["json"] == {"q": "Tom y Jerry", "source": "es", "target": "en", "format": "text"}
        assert kwargs["params"] == {"key": "test-key"}
        return response
    monkeypatch.setattr(requests, "post", post)
    assert providers.GoogleTranslator(source="es").translate("Tom y Jerry") == "Tom & Jerry"
    assert response.closed


def test_mymemory_bounds_utf8_requests_and_checks_quota(monkeypatch):
    chunks = []
    def get(url, **kwargs):
        chunks.append(kwargs["params"]["q"])
        assert len(chunks[-1].encode()) <= 500
        return Response({"responseStatus": 200, "responseData": {"translatedText": "translated"}})
    monkeypatch.setattr(requests, "get", get)
    text = "é" * 501
    assert providers.MyMemoryTranslator(source="fr").translate(text) == "translated translated translated"
    assert "".join(chunks) == text
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response({"responseStatus": 200, "quotaFinished": True}))
    with pytest.raises(providers.TranslationError, match="quota"):
        providers.MyMemoryTranslator(source="fr").translate("bonjour")


@pytest.mark.parametrize("response", [Response({}, 403), Response({"translations": []}), Response({"translations": [{"text": ""}]})])
def test_provider_failures_are_not_translations(monkeypatch, response):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    with pytest.raises(providers.TranslationError):
        providers.DeeplTranslator("test-key").translate("bonjour")
    assert response.closed


def test_transport_error_does_not_disclose_key_or_source(monkeypatch):
    def fail(*a, **kw):
        raise requests.Timeout("url?key=secret&text=private")
    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(providers.TranslationError) as error:
        providers.DeeplTranslator("secret").translate("private")
    assert "secret" not in str(error.value) and "private" not in str(error.value)


def test_concurrent_timeouts_do_not_affect_other_requests():
    barrier = Barrier(2)
    original = requests.sessions.Session.request
    def worker(timeout):
        with _request_timeout(timeout):
            barrier.wait(timeout=5)
            assert requests.sessions.Session.request is original
            return providers.request_timeout.get()
    with ThreadPoolExecutor(2) as executor:
        assert list(executor.map(worker, [2, 7])) == [2, 7]
    assert providers.request_timeout.get() == 10

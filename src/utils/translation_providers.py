"""Small HTTP clients for translation, with request-local timeouts.

Google Cloud requires GOOGLE_TRANSLATE_API_KEY. Without it, the existing
secondary MyMemory service is used. No provider error is treated as a translation.
"""

from __future__ import annotations

import html
import os
from contextvars import ContextVar

import requests

request_timeout: ContextVar[float] = ContextVar("translation_request_timeout", default=10.0)


class TranslationError(RuntimeError):
    pass


def _send(method, url, **kwargs):
    try:
        return method(url, **kwargs)
    except requests.RequestException as exc:
        raise TranslationError("Translation provider connection failed") from exc


def _response(response):
    # Do not include request URLs or bodies (which can contain text or API keys)
    # in exceptions displayed by the app.
    if response.status_code != 200:
        raise TranslationError(f"Translation provider returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise TranslationError("Translation provider returned invalid JSON") from exc


def _translated(value):
    if not isinstance(value, str) or not value.strip():
        raise TranslationError("Translation provider returned empty text")
    return html.unescape(value).strip()


class MyMemoryTranslator:
    def __init__(self, source="auto", target="en"):
        self.source, self.target = source, target

    def translate(self, text):
        if not text.strip():
            return text
        source = self.source
        if source == "auto":
            from langdetect import detect
            source = detect(text)
        if source == self.target:
            return text
        # MyMemory limits each query to 500 UTF-8 bytes; never split a codepoint.
        chunks, current = [], ""
        for char in text:
            if len((current + char).encode("utf-8")) > 500:
                chunks.append(current)
                current = ""
            current += char
        if current:
            chunks.append(current)
        result = []
        for chunk in chunks:
            with _send(requests.get,
                "https://api.mymemory.translated.net/get",
                params={"q": chunk, "langpair": f"{source}|{self.target}"},
                timeout=request_timeout.get(),
            ) as response:
                data = _response(response)
            if str(data.get("responseStatus")) != "200" or data.get("quotaFinished"):
                raise TranslationError("MyMemory translation failed or its quota was exhausted")
            result.append(_translated(data.get("responseData", {}).get("translatedText")))
        return " ".join(result)

    def translate_batch(self, texts):
        return [self.translate(text) for text in texts]


class GoogleTranslator(MyMemoryTranslator):
    def translate(self, text):
        key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        if not key:
            return super().translate(text)
        body = {"q": text, "target": self.target, "format": "text"}
        if self.source != "auto":
            body["source"] = self.source
        with _send(requests.post,
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": key}, json=body, timeout=request_timeout.get(),
        ) as response:
            data = _response(response)
        translations = data.get("data", {}).get("translations", [])
        if len(translations) != 1:
            raise TranslationError("Google returned an unexpected translation count")
        return _translated(translations[0].get("translatedText"))


class DeeplTranslator(MyMemoryTranslator):
    def __init__(self, api_key, source="auto", target="en"):
        super().__init__(source, target)
        self.api_key = api_key

    def translate(self, text):
        return self.translate_batch([text])[0]

    def translate_batch(self, texts):
        if not texts:
            return []
        endpoint = "https://api-free.deepl.com" if self.api_key.endswith(":fx") else "https://api.deepl.com"
        results = []
        # DeepL accepts at most 50 texts in each request.
        for start in range(0, len(texts), 50):
            batch = texts[start:start + 50]
            body = {"text": batch, "target_lang": self.target.upper()}
            if self.source != "auto":
                body["source_lang"] = self.source.upper()
            with _send(requests.post,
                endpoint + "/v2/translate", json=body,
                headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                timeout=request_timeout.get(),
            ) as response:
                data = _response(response)
            translations = data.get("translations", [])
            if len(translations) != len(batch):
                raise TranslationError("DeepL returned an unexpected translation count")
            results.extend(_translated(item.get("text")) for item in translations)
        return results

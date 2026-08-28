# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Verifies that vectors use translated text while the registry keeps source text."""

from src.core.cross_lingual import prepare_documents_for_embedding


def test_translation_indexing_contract(monkeypatch):
    source_chunks = {
        "english.pdf": ["Artificial intelligence supports education."],
        "hindi.pdf": ["कृत्रिम बुद्धिमत्ता शिक्षा का समर्थन करती है।"],
    }

    def fake_prepare(text):
        is_hindi = text.startswith("कृत्रिम")
        return {
            "original_text": text,
            "embedding_text": (
                "Artificial intelligence supports education." if is_hindi else text
            ),
            "detected_language": "hi" if is_hindi else "en",
            "translated": is_hindi,
            "translation_failed": False,
        }

    monkeypatch.setattr(
        "src.core.cross_lingual.prepare_text_for_embedding",
        fake_prepare,
    )

    aligned, metadata = prepare_documents_for_embedding(source_chunks)

    # Embedding input is aligned in English.
    assert aligned["english.pdf"] == aligned["hindi.pdf"]

    # Display/database source remains in its original language.
    assert source_chunks["hindi.pdf"][0].startswith("कृत्रिम")
    assert metadata["hindi.pdf"][0]["original_text"].startswith("कृत्रिम")

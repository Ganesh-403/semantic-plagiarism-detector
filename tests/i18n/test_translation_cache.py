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

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.i18n import translator


@pytest.fixture(autouse=True)
def reset_translation_cache():
    translator.clear_translation_cache()
    yield
    translator.clear_translation_cache()
    translator.load_translations()


def write_dictionary(
    directory: Path,
    language: str,
    payload,
) -> Path:
    destination = directory / f"{language}.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def test_repeated_dictionary_load_reads_disk_once(tmp_path):
    file_path = write_dictionary(
        tmp_path,
        "en",
        {"title": "Cached title"},
    )

    with patch(
        "builtins.open",
        wraps=open,
    ) as mocked_open:
        first = translator._load_translation_dictionary(str(file_path))
        second = translator._load_translation_dictionary(str(file_path))

    assert first == {"title": "Cached title"}
    assert second == first
    assert mocked_open.call_count == 1


def test_cache_is_separate_for_each_language(tmp_path):
    english = write_dictionary(
        tmp_path,
        "en",
        {"title": "English"},
    )
    spanish = write_dictionary(
        tmp_path,
        "es",
        {"title": "Español"},
    )

    with patch(
        "builtins.open",
        wraps=open,
    ) as mocked_open:
        translator._load_translation_dictionary(str(english))
        translator._load_translation_dictionary(str(spanish))
        translator._load_translation_dictionary(str(english))
        translator._load_translation_dictionary(str(spanish))

    assert mocked_open.call_count == 2


def test_cached_result_is_not_mutated_by_caller(tmp_path):
    file_path = write_dictionary(
        tmp_path,
        "en",
        {"title": "Original"},
    )

    first = translator._load_translation_dictionary(str(file_path))
    first["title"] = "Changed by caller"

    second = translator._load_translation_dictionary(str(file_path))

    assert second["title"] == "Original"


def test_clear_translation_cache_forces_new_disk_read(tmp_path):
    file_path = write_dictionary(
        tmp_path,
        "en",
        {"title": "First"},
    )

    first = translator._load_translation_dictionary(str(file_path))
    file_path.write_text(
        json.dumps({"title": "Second"}),
        encoding="utf-8",
    )

    cached = translator._load_translation_dictionary(str(file_path))
    translator.clear_translation_cache()
    refreshed = translator._load_translation_dictionary(str(file_path))

    assert first["title"] == "First"
    assert cached["title"] == "First"
    assert refreshed["title"] == "Second"


def test_non_object_json_is_rejected(tmp_path):
    file_path = write_dictionary(
        tmp_path,
        "en",
        ["not", "an", "object"],
    )

    with pytest.raises(
        ValueError,
        match="must contain a JSON object",
    ):
        translator._load_translation_dictionary(str(file_path))


def test_load_translations_uses_cached_file_loader(
    tmp_path,
    monkeypatch,
):
    write_dictionary(
        tmp_path,
        "en",
        {"title": "English title"},
    )
    write_dictionary(
        tmp_path,
        "es",
        {"title": "Título español"},
    )
    write_dictionary(
        tmp_path,
        "fr",
        {"title": "Titre français"},
    )
    monkeypatch.setattr(
        translator,
        "_I18N_DIR",
        str(tmp_path),
    )

    with patch(
        "builtins.open",
        wraps=open,
    ) as mocked_open:
        translator.load_translations()
        translator.load_translations()

    assert mocked_open.call_count == 3
    assert (
        translator.get_text(
            "title",
            lang="es",
        )
        == "Título español"
    )


def test_missing_language_file_is_skipped(
    tmp_path,
    monkeypatch,
):
    write_dictionary(
        tmp_path,
        "en",
        {"title": "English title"},
    )
    monkeypatch.setattr(
        translator,
        "_I18N_DIR",
        str(tmp_path),
    )

    translator.load_translations()

    assert (
        translator.get_text(
            "title",
            lang="es",
        )
        == "English title"
    )


def test_malformed_language_file_is_skipped(
    tmp_path,
    monkeypatch,
):
    write_dictionary(
        tmp_path,
        "en",
        {"title": "English title"},
    )
    (tmp_path / "es.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        translator,
        "_I18N_DIR",
        str(tmp_path),
    )

    translator.load_translations()

    assert (
        translator.get_text(
            "title",
            lang="es",
        )
        == "English title"
    )


def test_existing_html_safe_formatting_is_preserved(
    tmp_path,
    monkeypatch,
):
    write_dictionary(
        tmp_path,
        "en",
        {"welcome": "Welcome, {name}!"},
    )
    monkeypatch.setattr(
        translator,
        "_I18N_DIR",
        str(tmp_path),
    )

    translator.load_translations()

    assert translator.get_text(
        "welcome",
        name="<script>alert(1)</script>",
    ) == ("Welcome, " "&lt;script&gt;alert(1)&lt;/script&gt;!")

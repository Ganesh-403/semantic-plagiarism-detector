from pathlib import Path


TRANSLATOR_PATH = Path("src/i18n/translator.py")


def test_translation_loader_uses_streamlit_cache_data():
    source = TRANSLATOR_PATH.read_text(encoding="utf-8")

    assert "@st.cache_data(show_spinner=False)" in source
    assert "def _load_translation_dictionary(" in source


def test_load_translations_delegates_to_cached_loader():
    source = TRANSLATOR_PATH.read_text(encoding="utf-8")

    assert (
        "_load_translation_dictionary(file_path)"
        in source
    )


def test_cache_can_be_cleared_explicitly():
    source = TRANSLATOR_PATH.read_text(encoding="utf-8")

    assert "def clear_translation_cache()" in source
    assert "_load_translation_dictionary.clear()" in source

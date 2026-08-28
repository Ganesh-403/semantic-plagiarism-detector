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

from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")
WARNING_LIST_PATH = Path("src/utils/warning_list.py")


def test_copy_controls_rendered():
    app_source = APP_PATH.read_text(encoding="utf-8")
    assert "render_copy_button(" in app_source
    assert 'copy_label="📋 Copy Snippet"' in app_source


def test_component_keys_are_unique():
    app_source = APP_PATH.read_text(encoding="utf-8")
    # Verify that different button_ids are generated, one for ca and one for cb
    assert 'button_id=f"copy_ca_{rank}"' in app_source
    assert 'button_id=f"copy_cb_{rank}"' in app_source


def test_plain_text_snippets_supplied_to_clipboard():
    app_source = APP_PATH.read_text(encoding="utf-8")
    # Make sure we pass plain 'ca' and 'cb' rather than 'highlighted_ca'
    assert "text_to_copy=ca" in app_source
    assert "text_to_copy=cb" in app_source


def test_render_copy_button_exists():
    warning_list_source = WARNING_LIST_PATH.read_text(encoding="utf-8")
    assert "def render_copy_button(" in warning_list_source
    assert "document.execCommand('copy')" in warning_list_source


def test_uses_modern_clipboard_api_as_primary_path():
    """navigator.clipboard.writeText() must be the primary copy mechanism,
    per the deprecation of document.execCommand('copy') (MDN 'Deprecated'
    since 2020; Safari/Firefox may silently fail)."""
    warning_list_source = WARNING_LIST_PATH.read_text(encoding="utf-8")
    assert "navigator.clipboard.writeText(text)" in warning_list_source
    assert "navigator.clipboard && navigator.clipboard.writeText" in warning_list_source


def test_clipboard_write_has_then_and_catch_feedback():
    """Success/failure UI feedback must be wired via .then()/.catch()."""
    warning_list_source = WARNING_LIST_PATH.read_text(encoding="utf-8")
    assert "navigator.clipboard.writeText(text).then(function()" in warning_list_source
    assert ".catch(function(err)" in warning_list_source
    # Both branches must actually drive the button's UI state.
    assert "showCopied();" in warning_list_source
    assert "showFailed();" in warning_list_source


def test_execcommand_is_only_used_as_a_fallback():
    """document.execCommand('copy') must remain solely as an older-browser
    fallback, not the primary copy path — it must live inside the fallback
    function and only be reached when the Clipboard API is unavailable or
    rejects."""
    warning_list_source = WARNING_LIST_PATH.read_text(encoding="utf-8")
    assert "function legacyCopyFallback()" in warning_list_source

    fallback_start = warning_list_source.index("function legacyCopyFallback()")
    fallback_end = warning_list_source.index(
        "if (navigator.clipboard && navigator.clipboard.writeText)"
    )
    fallback_body = warning_list_source[fallback_start:fallback_end]
    assert "document.execCommand('copy')" in fallback_body

    # The modern path must be tried first and only fall back on failure.
    clipboard_call_index = warning_list_source.index(
        "navigator.clipboard.writeText(text).then"
    )
    assert clipboard_call_index < warning_list_source.index(
        "legacyCopyFallback();", clipboard_call_index
    )


def test_hidden_textarea_only_created_inside_fallback():
    """The always-on hidden <textarea> workaround must be gone; a textarea
    should only be created inside the legacy fallback function, not
    unconditionally on every click."""
    warning_list_source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    textarea_occurrences = warning_list_source.count(
        'document.createElement("textarea")'
    )
    assert textarea_occurrences == 1

    fallback_start = warning_list_source.index("function legacyCopyFallback()")
    fallback_end = warning_list_source.index(
        "if (navigator.clipboard && navigator.clipboard.writeText)"
    )
    fallback_body = warning_list_source[fallback_start:fallback_end]
    assert 'document.createElement("textarea")' in fallback_body

    # And it must be defined only within the fallback function, not created
    # unconditionally before the modern Clipboard API path is even tried.
    clipboard_call_index = warning_list_source.index(
        "navigator.clipboard.writeText(text).then"
    )
    textarea_index = warning_list_source.index('document.createElement("textarea")')
    assert textarea_index < clipboard_call_index
    assert fallback_start < textarea_index < fallback_end
